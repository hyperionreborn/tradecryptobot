import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import json
import random
from pathlib import Path
from datetime import datetime
from pathlib import Path

from .model import LSTMModel, PriceModel
from .data_fetch import (
    make_dataset
)


def train(
        model,
        loader,
        class_criterion,
        price_criterion,
        optimizer,
        device,
        alpha=1.0,
        beta=1.0,
):
    model.train()
    total_loss = 0.0
    for xb, y_class, y_change in loader:
        xb, y_class, y_change = xb.to(device), y_class.to(device), y_change.to(device)
        optimizer.zero_grad()
        logits, pred_price = model(xb)

        # Use raw logits for Focal Loss (it applies sigmoid internally)
        loss_class = class_criterion(logits, y_class)
        loss_price = price_criterion(pred_price, y_change)
        loss = alpha * loss_class + beta * loss_price

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def train_price(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for xb, _, y_change in loader:
        xb = xb.to(device)
        # Ensure y_price has shape [batch_size, 1]
        y_change = y_change.to(device).view(-1, 1)

        optimizer.zero_grad()
        pred_price = model(xb)
        loss = criterion(pred_price, y_change)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def evaluate_price(model, loader, price_criterion, device):
    model.eval()
    total_loss = 0.0
    price_mse = 0.0
    all_predictions = []
    with torch.no_grad():
        for xb, _, y_change in loader:
            xb = xb.to(device)
            # Ensure y_price has shape [batch_size, 1]
            y_change = y_change.to(device).view(-1, 1)

            pred_price = model(xb)
            loss = price_criterion(pred_price, y_change)

            total_loss += loss.item() * xb.size(0)
            price_mse += ((pred_price - y_change) ** 2).sum().item()
            all_predictions.extend(pred_price.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        mse = price_mse / len(loader.dataset)
    return all_predictions, mse, avg_loss


def evaluate(model, loader, class_criterion, price_criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    price_mse = 0.0
    with torch.no_grad():
        for xb, y_class, y_change in loader:
            xb, y_class, y_change = (
                xb.to(device),
                y_class.to(device),
                y_change.to(device),
            )
            logits, pred_price = model(xb)

            # Use raw logits for Focal Loss (it applies sigmoid internally)
            loss_class = class_criterion(logits, y_class)
            loss_price = price_criterion(pred_price, y_change)
            loss = loss_class + loss_price
            total_loss += loss.item() * xb.size(0)

            # For accuracy calculation, apply sigmoid to logits
            pred_class = torch.sigmoid(logits)
            correct += ((pred_class > 0.5) == y_class).sum().item()
            price_mse += ((pred_price - y_change) ** 2).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    mse = price_mse / len(loader.dataset)

    return avg_loss, accuracy, mse


def evaluate_with_logits(model, loader, class_criterion, price_criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    price_mse = 0.0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for xb, y_class, y_change in loader:
            xb, y_class, y_change = (
                xb.to(device),
                y_class.to(device),
                y_change.to(device),
            )
            logits, pred_price = model(xb)

            # Use raw logits for Focal Loss (it applies sigmoid internally)
            loss_class = class_criterion(logits, y_class)
            loss_price = price_criterion(pred_price, y_change)
            loss = loss_class + loss_price
            total_loss += loss.item() * xb.size(0)

            # For accuracy calculation, apply sigmoid to logits
            pred_class = torch.sigmoid(logits)
            correct += ((pred_class > 0.5) == y_class).sum().item()
            price_mse += ((pred_price - y_change) ** 2).sum().item()

            all_logits.extend(
                logits.cpu().numpy()
            )  # Return raw logits for Platt scaling
            all_labels.extend(y_class.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    mse = price_mse / len(loader.dataset)

    return avg_loss, accuracy, mse, all_logits, all_labels


class NumpyDataset(Dataset):
    def __init__(self, X_path, y_path, features_path):
        self.X = torch.FloatTensor(np.load(X_path))
        self.y_prices = torch.FloatTensor(np.load(y_path))

        with open(features_path, 'r') as f:
            self.feature_names = json.load(f)

        try:
            self.close_idx = self.feature_names.index('Close')
        except ValueError:
            print("Warning: 'Close' not found in features, using index 3 as fallback")
            self.close_idx = 3

        # Generate binary labels (1 if future price > current price)
        # X shape: (N, T, F)
        # current_price is the Close price at the last time step of the window
        current_prices = self.X[:, -1, self.close_idx]
        self.y_class = (self.y_prices > current_prices).float()

        # Calculate price change ratio for regression target
        # Avoid division by zero
        safe_current_prices = torch.where(current_prices == 0, torch.ones_like(current_prices), current_prices)
        self.y_change = (self.y_prices / safe_current_prices) - 1

        print(f"NumpyDataset loaded: {len(self.X)} samples")
        print(f"Positive labels: {self.y_class.sum().item()}/{len(self.y_class)} ({self.y_class.mean().item():.2%})")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_class[idx].unsqueeze(0), self.y_change[idx].unsqueeze(0)


# # --- Main function to run the training and evaluation to call from main.py ---
def TrainAll(dataset_dir, EPOCHS=100, BATCH=64, LR=2e-3):
    TEST_SIZE = 0.2
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print("Getting initial training data...\n")

    outdir_path = Path(dataset_dir)
    x_path = outdir_path / "X.npy"
    y_path = outdir_path / "y.npy"
    features_path = outdir_path / "features.json"

    if not x_path.exists() or not y_path.exists():
        print(f"Dataset not found in {dataset_dir}")
        return

    dataset = NumpyDataset(x_path, y_path, features_path)

    if len(dataset) == 0:
        print("No valid training data\n")
        exit(1)
    pos_ratio = (dataset.y_class.sum() / len(dataset)).item()
    print(f"Label Distribution → Up: {pos_ratio:.2%}, Down: {1 - pos_ratio:.2%} across {len(dataset)} samples ")
    # Calculate class weights for better balancing

    # Split into train/test
    train_size = int((1 - TEST_SIZE) * len(dataset))
    test_size = len(dataset) - train_size
    split_idx = int(0.8 * len(dataset))
    train_ds = torch.utils.data.Subset(dataset, range(0, split_idx))
    test_ds = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))
    X_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    X_test = torch.stack([test_ds[i][0] for i in range(len(test_ds))])

    # ===== RESHAPE AND SCALE FEATURES =====
    Ntr, T, F = X_train.shape
    Nte = X_test.shape[0]

    # Flatten time steps for scaling
    Xtr_2d = X_train.view(-1, F).numpy()
    Xte_2d = X_test.view(-1, F).numpy()

    # Fit scaler on TRAIN only and transform both train and test
    scaler = StandardScaler()
    Xtr_2d = scaler.fit_transform(Xtr_2d)  # ✅ Fit on train only
    Xte_2d = scaler.transform(Xte_2d)  # ✅ Transform test
    scaler_path = Path(dataset_dir) / "scaler.pkl"
    # Save scaler for later use
    joblib.dump(scaler, scaler_path)
    print("Scaler saved to:", scaler_path)
    y_test = torch.stack([test_ds[i][1] for i in range(len(test_ds))])

    up_ratio = y_test.mean().item()

    print("\n===== TEST SET CLASS BALANCE =====")
    print(f"UP Ratio:   {up_ratio:.2%}")
    print(f"DOWN Ratio: {1 - up_ratio:.2%}")

    # ===== ALWAYS-UP BASELINE =====
    always_up_acc = (y_test == 1).float().mean().item()

    print("\n===== BASELINE CHECK =====")
    print(f"Always-UP Accuracy: {always_up_acc:.2%}")
    # Reshape back to original LSTM shape
    X_train = torch.from_numpy(Xtr_2d).float().view(Ntr, T, F)
    X_test = torch.from_numpy(Xte_2d).float().view(Nte, T, F)

    # ===== WRITE BACK INTO ORIGINAL DATASET STORAGE =====
    train_ds.dataset.X[train_ds.indices] = X_train
    test_ds.dataset.X[test_ds.indices] = X_test

    class_criterion = nn.BCEWithLogitsLoss()  # Increased gamma
    price_criterion = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH)

    # Get input size from the first item in dataset
    sample_x, _, _ = dataset[0]
    input_size = sample_x.shape[1]  # Number of features

    # Slightly larger model for better capacity
    model = LSTMModel(
        input_size=input_size, hidden_size=256, num_layers=3, dropout=0.2
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=1e-5
    )  # Add weight decay

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)

    model_config = {
        "input_size": input_size,
        "hidden_size": 256,  # Increased
        "num_layers": 3,  # Increased
        "dropout": 0.2,  # Increased
    }
    with open("model_config.json", "w") as f:
        json.dump(model_config, f)

    # Early stopping variables
    best_accuracy = 0.0
    patience = 200
    patience_counter = 0

    print("Starting training with improved class accuracy focus...\n")
    for epoch in range(1, EPOCHS + 1):
        # Emphasize classification more heavily (alpha=2.0 vs beta=0.5)
        train_loss = train(
            model,
            train_loader,
            class_criterion,
            price_criterion,
            optimizer,
            device,
            alpha=2.0,
            beta=0.0,
        )
        test_loss, test_acc, test_mse = evaluate(
            model, test_loader, class_criterion, price_criterion, device
        )

        # Step the scheduler
        scheduler.step()

        # Early stopping based on accuracy
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f"{dataset_dir}.pt")
        else:
            patience_counter += 1

        if (
                epoch % 50 == 0 or patience_counter == 0
        ):  # Print more frequently for best epochs
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Class Acc: {test_acc:.2%} | "
                f"Price MSE: {test_mse:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Best Acc: {best_accuracy:.2%}"
            )

        # Early stopping
        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch}. Best accuracy: {best_accuracy:.2%}\n"
            )
            break

    # Load best model for final save

    print(f"Model saved as {dataset_dir}.pt with best accuracy: {best_accuracy:.2%}\n")


def test_live(symbol,resample_hours,horizon,window_days):
    return





