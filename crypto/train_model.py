import time

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
import joblib
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,roc_auc_score
import numpy as np
import json
import random
from pathlib import Path
from datetime import datetime, timezone
from pathlib import Path

from .model import ImprovedLSTMModel,CryptoEnsemble
from .model import LSTMModel, PriceModel
from .data_fetch import (
    make_dataset,
    get_evaluate_window,
    scale_live_window
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
def TrainAll(dataset_dir,SEED = 42, EPOCHS=100, BATCH=64, LR=2e-3):
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

    N = len(dataset)

    train_end = int(0.85 * N)

    train_ds = torch.utils.data.Subset(dataset, range(0, train_end))
    test_ds = torch.utils.data.Subset(dataset, range(train_end, N))
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
    HIDDEN_SIZE = 32
    NUM_LAYERS = 2
    DROPOUT = 0.4
    # Slightly larger model for better capacity
    model = ImprovedLSTMModel(
        input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=1e-5
    )  # Add weight decay

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    model_config = {
        "input_size": input_size,
        "hidden_size": HIDDEN_SIZE,  # Increased
        "num_layers": NUM_LAYERS,  # Increased
        "dropout": DROPOUT,  # Increased
    }
    with open(f"{dataset_dir}/model_config.json", "w") as f:
        json.dump(model_config, f)

    # Early stopping variables
    best_accuracy = 0.0
    patience = 200
    patience_counter = 0

    print("Starting training with  class accuracy focus...\n")
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
        test_loss, test_acc, _ = evaluate(
            model, test_loader, class_criterion, price_criterion, device
        )

        # Step the scheduler
        scheduler.step()

        # Early stopping based on accuracy
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f"{dataset_dir}/{SEED}_binary_model.pt")
        else:
            patience_counter += 1

        if (
                epoch % 10 == 0 or patience_counter == 0
        ):  # Print more frequently for best epochs
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Class Acc: {test_acc:.2%} | "
                f"LR: {current_lr:.2e} | "
                f"Best Acc: {best_accuracy:.2%}"
            )

        # Early stopping
        if patience_counter >= patience:

            break

    # Load best model for final save

    print(f"Model saved as {dataset_dir}/{SEED}_binary_model.pt with best accuracy: {best_accuracy:.2%}\n")

    best_mse = 64.0
    patience = 200
    patience_counter = 0
    price_model = ImprovedLSTMModel(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    price_optimizer = torch.optim.Adam(
        price_model.parameters(), lr=LR, weight_decay=1e-5
    )

    price_scheduler = torch.optim.lr_scheduler.StepLR(price_optimizer, step_size=50, gamma=0.9)
    print("Starting training with price focus...\n")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(
            price_model,
            train_loader,
            class_criterion,
            price_criterion,
            price_optimizer,
            device,
            alpha=0.0,
            beta=2.0,
        )
        test_loss, test_acc, test_mse = evaluate(
           price_model, test_loader, class_criterion, price_criterion, device
        )

        # Step the scheduler
        price_scheduler.step()

        # Early stopping based on accuracy
        if test_mse < best_mse:
            best_mse = test_mse
            patience_counter = 0
            # Save best model
            torch.save(price_model.state_dict(), f"{dataset_dir}/{SEED}_price_model.pt")
        else:
            patience_counter += 1

        if (
                epoch % 50 == 0 or patience_counter == 0
        ):  # Print more frequently for best epochs
            current_lr = price_scheduler.get_last_lr()[0]
            rmse = np.sqrt(best_mse)  # 0.05 = 5% błąd

            print(
                f"Epoch {epoch:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Current mse: {test_mse} | "
                f"LR: {current_lr:.2e} | "
                f"Best mse: {best_mse} | "
                f"RMSE: {rmse:.2%}"
            )

        # Early stopping
        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch}. Best accuracy: {best_mse:.2%}\n"
            )
            break

    print("\n===== CALIBRATION EVALUATION (TEST SET) =====")
    model.load_state_dict(torch.load(f"{dataset_dir}/{SEED}_binary_model.pt"))
    model.eval()
    price_model.load_state_dict(torch.load(f"{dataset_dir}/{SEED}_price_model.pt"))
    price_model.eval()
    # Logity z TEST
    _, _, _, test_logits, test_labels = evaluate_with_logits(
        model, test_loader, class_criterion, price_criterion, device
    )

    test_logits = np.array(test_logits).reshape(-1,1)
    test_labels = np.array(test_labels).reshape(-1)

    # --- PRZED KALIBRACJĄ ---
    probs_raw = expit(test_logits)  # sigmoid(logits)
    brier_raw = brier_score_loss(test_labels, probs_raw)

    # --- PO KALIBRACJI ---

    print(f"Brier score: {brier_raw:.4f}")
    print("\nComputing ROC-AUC ")
    _, _, _, test_logits, test_labels = evaluate_with_logits(
        model, test_loader, class_criterion, price_criterion, device
    )

    # Convert to numpy arrays
    test_logits = np.array(test_logits).reshape(-1)
    test_labels = np.array(test_labels).reshape(-1)

    # Convert logits → probabilities
    probs_raw = expit(test_logits)  # sigmoid(logits)

    # Compute ROC-AUC
    roc_auc = roc_auc_score(test_labels, probs_raw)

    print(f"ROC-AUC (raw model): {roc_auc:.4f}")

    print("\n===== FILTERED METRICS (RAW SIGMOID) =====")

    probs_raw = expit(test_logits).reshape(-1)
    mask = (probs_raw >= 0.60) | (probs_raw <= 0.40)

    filtered_probs = probs_raw[mask]
    filtered_labels = test_labels[mask]

    directions = (filtered_probs > 0.5).astype(int)
    filtered_acc = (directions == filtered_labels).mean()

    print(f"Trades: {len(filtered_labels)} / {len(test_labels)}")
    print(f"Accuracy after filtering (RAW): {filtered_acc:.2%}")
    all_predictions = []
    with torch.no_grad():
        for xb, y_class, y_change in test_loader:
            xb = xb.to(device)

            # Predykcja z obu modeli
            logits, _ = model(xb)  # ignorujemy price z class_model
            prob_up = torch.sigmoid(logits)

            _, pred_change = price_model(xb)

            # Przenieś na CPU
            prob_up = prob_up.cpu().numpy().flatten()
            pred_change = pred_change.cpu().numpy().flatten()
            y_class = y_class.cpu().numpy().flatten()

            # Zapisz wszystko
            for i in range(len(prob_up)):
                all_predictions.append({
                    'prob_up': prob_up[i],
                    'pred_change': pred_change[i],
                    'actual_class': y_class[i],
                })
    print("Confidence trading accuracy")

    prop_threshold = 0.50
    trades = []
    correct_trades = []
    for p in all_predictions:

        predict_long = (p['prob_up'] > prop_threshold) and (p['pred_change'] > 0.0)

                # Short: oba modele przewidują spadek
        predict_short = (p['prob_up'] < prop_threshold) and (p['pred_change'] < 0.0)

        if predict_long or predict_short:
            trades.append(p)
            predicted_direction = 1 if predict_long else 0
            actual_direction = 1 if p['actual_class'] > 0 else 0
            correct_trades.append(predicted_direction == actual_direction)
    if trades:
        accuracy_confidence = sum(correct_trades)/len(correct_trades)
        print(f"Num trades:{len(correct_trades)}")
        print(f"Accuracy:{accuracy_confidence:.2%}")
    all_changes = dataset.y_change.numpy()
    avg_change = np.mean(np.abs(all_changes))
    print(f"average  change: {avg_change:.4%}")

    # Podział na train/test
    train_changes = dataset.y_change[train_ds.indices].numpy()
    test_changes = dataset.y_change[test_ds.indices].numpy()

    print(f"average train change: {np.mean(np.abs(train_changes)):.4%}")
    print(f"average test change: {np.mean(np.abs(test_changes)):.4%}")
    return best_accuracy,accuracy_confidence
def Ensemble_model_train(dataset,num_models = 3 ):
    seeds = [47]
    arr_acc = []
    arr_conf = []
    for i in range(1, num_models):
        seeds.append(seeds[i - 1] * 13)
    for seed in seeds:
        acc,conf =  TrainAll(dataset_dir=dataset,SEED=seed)
        arr_acc.append(acc)
        arr_conf.append(conf)
    accuracies = np.array(arr_acc)
    weights = accuracies / accuracies.sum()
    ensemble_info = {
        'seeds': seeds,
        'val_accuracies': arr_acc,
        'val_accuracies_confidence': arr_conf,
        'weights': weights.tolist(),
        'created_at': datetime.now().isoformat()
    }
    ensemble_path = Path(dataset) / "ensemble_info.json"
    with open(ensemble_path, 'w') as f:
        json.dump(ensemble_info, f, indent=2)

    print(f"\n✅ Ensemble training complete!")
    print(f"Model accuracies: {ensemble_info['val_accuracies']}")
    print(f"Ensemble weights: {ensemble_info['weights']}")
    print(f"Metadata saved to: {ensemble_path}")

    return ensemble_info
def Ensemble_model_evaluate(dataset):
    model = CryptoEnsemble(dataset)
    price_model = CryptoEnsemble(dataset,"price_model",device="cpu")
    all_predictions = []
    outdir_path = Path(dataset)
    datasetn = NumpyDataset(
        outdir_path / "X.npy",
        outdir_path / "y.npy",
        outdir_path / "features.json"
    )
    ensemble_path = outdir_path / "ensemble_info.json"

    with open(ensemble_path, 'r') as f:
        info = json.load(f)

    val_conf = info['val_accuracies_confidence']
    val_accs = info['val_accuracies']

    # Calculate statistics
    best_confidence = np.max(val_conf)
    best_individual_accuracy = np.max(val_accs)
    # Get test split
    N = len(datasetn)
    test_start = int(0.75 * N)
    test_indices = range(test_start, N)
    all_preds = []
    all_labels = []
    for idx in test_indices:
        xbn, y_class, y_change = datasetn[idx]
        xb = xbn.numpy()

        # Get predictions from ensemble model
        logits, pred_change = model.predict(xb)

        # Convert logits to tensor, apply sigmoid, get float value
        prob_up_tensor = torch.sigmoid(torch.tensor(logits, dtype=torch.float32))
        prob_up = prob_up_tensor.item()  # Get Python float - THIS IS A FLOAT, NOT TENSOR

        # pred_change is already a float from ensemble.predict()
        pred_change_val = float(pred_change)

        # Convert y_class tensor to float
        y_class_val = y_class.item() if isinstance(y_class, torch.Tensor) else float(y_class)

        # Get predicted class
        pred_class = 1 if prob_up > 0.5 else 0
        all_preds.append(pred_class)
        all_labels.append(y_class_val)

        # Store for confidence trading analysis
        all_predictions.append({
            'prob_up': prob_up,  # Already a float
            'pred_change': pred_change_val,  # Already a float
            'actual_class': y_class_val,  # Already a float
        })

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate accuracy
    accuracy = (all_preds == all_labels).mean()
    improvment = accuracy - best_individual_accuracy
    print(f"Ensemble model accuracy:{accuracy:.4%} improvement {improvment:.4%} ")

    prop_threshold = 0.50
    trades = []
    correct_trades = []

    for p in all_predictions:
        predict_long = (p['prob_up'] > prop_threshold) and (p['pred_change'] > 0.0)
        predict_short = (p['prob_up'] < prop_threshold) and (p['pred_change'] < 0.0)

        if predict_long or predict_short:
            trades.append(p)
            predicted_direction = 1 if predict_long else 0
            actual_direction = 1 if p['actual_class'] > 0 else 0
            correct_trades.append(predicted_direction == actual_direction)

    if trades:
        accuracy_confidence = sum(correct_trades) / len(correct_trades)
        print(f"Num trades: {len(trades)}")
        print(f"Accuracy: {accuracy_confidence:.2%}")

    if len(correct_trades) > 0:
        improvment_conf = accuracy_confidence - best_confidence
        print(f"Confidence trading accuracy {accuracy_confidence:.4%} improvement: {improvment_conf:.4%}")
    else:
        print("No trades made with confidence filtering")
def get_current_utc_time():
    """Get current hour and minute in UTC"""
    now_utc = datetime.now(timezone.utc)
    return now_utc.hour, now_utc.minute
def check_if_good_for_prediction(resample_hours=6, max_minutes_after=15):
    """
    Check if current time is good for prediction
    - Must be at resample hour (0, 6, 12, 18 for 6h)
    - Must be within X minutes after candle start
    """
    hour_utc, minute_utc = get_current_utc_time()

    # Check if it's a resample hour
    if hour_utc % resample_hours != 0:
        return False

    # Check if we're too late into the candle
    if minute_utc > max_minutes_after:
        return False

    return True


def test_live(symbol,resample_hours,window_days,horizon):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LTCUSDT"]
    modelinfo_path = "model_info.json"
    with open(modelinfo_path, "r") as f:
        model_info = json.load(f)

    # Convert to numpy arrays
    symbolss = np.array(model_info['symbols'])
    accuracies = np.array(model_info['accuracies'])
    conf_accuracies = np.array(model_info['confidence_accuracies'])
    best_conf_idx = np.argmax(conf_accuracies)
    best_acc_idx = np.argmax(accuracies)
    best_acc = np.max(accuracies)
    best_conf_acc = np.max(conf_accuracies)
    print(f"best symbol for confidence trading {symbolss[best_conf_idx]} with acc {best_conf_acc:.2%}")
    print(f"best symbol for trading {symbolss[best_acc_idx]} with acc {best_acc:.2%}  ")


    # Set thresholds
    min_acc = 0.6
    min_conf = 0.6

    # Create boolean masks
    acc_mask = accuracies >= min_acc  # Symbols with accuracy >= 0.6
    conf_mask = conf_accuracies >= min_conf  # Symbols with conf_accuracy >= 0.6
    combined_mask = acc_mask | conf_mask  # Symbols that meet EITHER threshold

    # Get arrays of qualified symbols
  # Array of symbols with good confidence accuracy
    qualified_symbols = symbolss[combined_mask]

    while True:
        if check_if_good_for_prediction(resample_hours=resample_hours,max_minutes_after=1):
            if symbol != "all":
                with open("model_config.json", "r") as f:
                    config = json.load(f)
                dataset_dir = f"{symbol}_{window_days}_{resample_hours}_{horizon}"
                model = ImprovedLSTMModel(
                    input_size=config["input_size"],
                    hidden_size=config["hidden_size"],
                    num_layers=config["num_layers"],
                    dropout=config["dropout"],
                ).to(device)
                model.load_state_dict(torch.load(f"{dataset_dir}/42_binary_model.pt"))
                model.eval()
                scaler_path = Path(dataset_dir) / "scaler.pkl"
                scaler = joblib.load(scaler_path)
                X = get_evaluate_window(symbol, window_days, resample_hours)
                X = scale_live_window(X, scaler)
                X = torch.tensor(X).to(device)

                with torch.no_grad():
                    logits, y_change = model(X)
                    prob_up = torch.sigmoid(logits).item()

                print(f"{symbol}: prob_up={prob_up:.3f}, ")
                del model
            else:
                for s in qualified_symbols:
                    with open("model_config.json", "r") as f:
                        config = json.load(f)
                    dataset_dir = f"{s}_{window_days}_{resample_hours}_{horizon}"
                    model = ImprovedLSTMModel(
                        input_size=config["input_size"],
                        hidden_size=config["hidden_size"],
                        num_layers=config["num_layers"],
                        dropout=config["dropout"],
                    ).to(device)
                    model.load_state_dict(torch.load(f"{dataset_dir}/42_binary_model.pt"))
                    model.eval()

                    scaler_path = Path(dataset_dir) / "scaler.pkl"
                    scaler = joblib.load(scaler_path)
                    X = get_evaluate_window(s,window_days,resample_hours)
                    X = scale_live_window(X,scaler)
                    X = torch.tensor(X).to(device)

                    with torch.no_grad():
                        logits, y_change = model(X)
                        prob_up = torch.sigmoid(logits).item()

                    print(f"{s}: prob_up={prob_up:.3f}, change={y_change.item():.4f}")
                    del model
            time.sleep(int(3550*resample_hours))
    return





