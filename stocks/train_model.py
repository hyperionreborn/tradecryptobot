import json
import random
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .model import FocalLoss, ImprovedLSTMModel


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NumpyDataset(Dataset):
    def __init__(self, X_path: Path, y_path: Path, features_path: Path):
        self.X = torch.FloatTensor(np.load(X_path))
        self.y_prices = torch.FloatTensor(np.load(y_path))

        with open(features_path, "r", encoding="utf-8") as f:
            self.feature_names = json.load(f)

        self.close_idx = self.feature_names.index("Close") if "Close" in self.feature_names else 3
        current_prices = self.X[:, -1, self.close_idx]
        safe_current = torch.where(current_prices == 0, torch.ones_like(current_prices), current_prices)

        self.y_class = (self.y_prices > current_prices).float()
        self.y_change = (self.y_prices / safe_current) - 1.0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_class[idx].unsqueeze(0), self.y_change[idx].unsqueeze(0)


def _train_epoch(model, loader, class_loss_fn, price_loss_fn, optimizer, device, alpha=1.0, beta=1.0):
    model.train()
    total_loss = 0.0
    for xb, y_class, y_change in loader:
        xb = xb.to(device)
        y_class = y_class.to(device)
        y_change = y_change.to(device)

        optimizer.zero_grad()
        logits, pred_change = model(xb)
        loss_class = class_loss_fn(logits, y_class)
        loss_price = price_loss_fn(pred_change, y_change)
        loss = (alpha * loss_class) + (beta * loss_price)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / max(1, len(loader.dataset))


def _eval_epoch(model, loader, class_loss_fn, price_loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    mse = 0.0
    with torch.no_grad():
        for xb, y_class, y_change in loader:
            xb = xb.to(device)
            y_class = y_class.to(device)
            y_change = y_change.to(device)
            logits, pred_change = model(xb)
            loss_class = class_loss_fn(logits, y_class)
            loss_price = price_loss_fn(pred_change, y_change)
            loss = loss_class + loss_price
            total_loss += loss.item() * xb.size(0)
            probs = torch.sigmoid(logits)
            correct += ((probs > 0.5) == y_class).sum().item()
            mse += ((pred_change - y_change) ** 2).sum().item()

    n = max(1, len(loader.dataset))
    return total_loss / n, correct / n, mse / n


def TrainAll(dataset_dir: str, SEED: int = 42, EPOCHS: int = 80, BATCH: int = 32, LR: float = 1e-3) -> Tuple[float, float]:
    _set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training stock model on {dataset_dir} ({device})")

    outdir = Path(dataset_dir)
    x_path = outdir / "X.npy"
    y_path = outdir / "y.npy"
    features_path = outdir / "features.json"
    if not x_path.exists() or not y_path.exists() or not features_path.exists():
        raise FileNotFoundError(f"Dataset files missing in {dataset_dir}")

    dataset = NumpyDataset(x_path, y_path, features_path)
    n = len(dataset)
    if n < 200:
        print(f"Warning: dataset only has {n} windows. Results may be unstable.")

    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    # Purge a gap between splits so highly-overlapping neighboring windows
    # do not leak short-term context across train/val/test boundaries.
    gap = max(1, int(0.5 * dataset.X.shape[1]))
    val_start = min(train_end + gap, n)
    test_start = min(val_end + gap, n)

    train_ds = torch.utils.data.Subset(dataset, range(0, train_end))
    val_ds = torch.utils.data.Subset(dataset, range(val_start, val_end))
    test_ds = torch.utils.data.Subset(dataset, range(test_start, n))

    if len(val_ds) < 32 or len(test_ds) < 32:
        # Fallback to contiguous chronological splits if purge removed too much data.
        train_ds = torch.utils.data.Subset(dataset, range(0, train_end))
        val_ds = torch.utils.data.Subset(dataset, range(train_end, val_end))
        test_ds = torch.utils.data.Subset(dataset, range(val_end, n))

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise ValueError(
            f"Split too small for training (train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}). "
            "Increase years or reduce window_days."
        )

    X_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    X_val = torch.stack([val_ds[i][0] for i in range(len(val_ds))])
    X_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))])

    ntr, t_steps, n_features = X_train.shape
    nte = X_test.shape[0]
    nval = X_val.shape[0]

    print(f"Samples: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    scaler = StandardScaler()
    Xtr_2d = scaler.fit_transform(X_train.view(-1, n_features).numpy())
    Xval_2d = scaler.transform(X_val.view(-1, n_features).numpy())
    Xte_2d = scaler.transform(X_test.view(-1, n_features).numpy())
    joblib.dump(scaler, outdir / "scaler.pkl")

    train_ds.dataset.X[train_ds.indices] = torch.from_numpy(Xtr_2d).float().view(ntr, t_steps, n_features)
    val_ds.dataset.X[val_ds.indices] = torch.from_numpy(Xval_2d).float().view(nval, t_steps, n_features)
    test_ds.dataset.X[test_ds.indices] = torch.from_numpy(Xte_2d).float().view(nte, t_steps, n_features)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH)
    val_loader = DataLoader(val_ds, batch_size=BATCH)

    up_ratio = (torch.stack([train_ds[i][1] for i in range(len(train_ds))]).mean().item()) if len(train_ds) else 0.5
    class_loss_fn = FocalLoss(alpha=max(0.1, 1.0 - up_ratio), gamma=1.0)
    price_loss_fn = nn.MSELoss()

    model = ImprovedLSTMModel(
        input_size=n_features,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    model_config = {
        "input_size": n_features,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.3,
    }
    with open(outdir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    best_val_acc = 0.0
    best_val_mse = float("inf")
    best_path = outdir / f"{SEED}_stock_model.pt"
    patience = 30
    no_improve = 0

    for _epoch in range(1, EPOCHS + 1):
        _ = _train_epoch(model, train_loader, class_loss_fn, price_loss_fn, optimizer, device, alpha=1.6, beta=0.8)
        _val_loss, val_acc, val_mse = _eval_epoch(model, val_loader, class_loss_fn, price_loss_fn, device)
        scheduler.step(val_mse)

        improved = (val_acc > best_val_acc) or (val_mse < best_val_mse)
        if improved:
            best_val_acc = max(best_val_acc, val_acc)
            best_val_mse = min(best_val_mse, val_mse)
            torch.save(model.state_dict(), best_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    _, final_acc, final_mse = _eval_epoch(model, test_loader, class_loss_fn, price_loss_fn, device)
    rmse = float(np.sqrt(final_mse))
    print(f"Saved best model to {best_path}")
    print(f"Test accuracy: {final_acc:.2%}")
    print(f"Test RMSE (change): {rmse:.2%}")
    return final_acc, rmse
