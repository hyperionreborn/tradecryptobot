import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import json
import random
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from .model import LSTMModel, PriceModel
from .data_fetch import (
    ,
make_dataset
)

class PriceDataset(Dataset):
    def __init__(self, data_dir="dataset", use_log_return=True, scale=True):
        data_dir = Path(data_dir)

        self.X = torch.from_numpy(np.load(data_dir / "X.npy")).float()
        y_future = torch.from_numpy(np.load(data_dir / "y.npy").astype(np.float32)).view(-1, 1)

        # Auto-detect Close index
        with open(data_dir / "features.json", "r") as f:
            features = json.load(f)
        close_idx = features.index("Close")

        current_price = self.X[:, -1, close_idx].view(-1, 1)

        # y_change: log return or percentage change

        self.y_change = torch.log(y_future / current_price)


        self.y_class = (self.y_change > 0).float()

        # Feature scaling (VERY important for LSTM)

        shape = self.X.shape
        self.scaler = StandardScaler()
        X2d = self.X.reshape(-1, shape[-1]).numpy()
        X2d = self.scaler.fit_transform(X2d)
        self.X = torch.from_numpy(X2d.reshape(shape)).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_class[idx], self.y_change[idx]

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs and targets should be in the format (B, 1)
        # Calculate BCE loss
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(inputs, targets)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        # Calculate p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Calculate alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate modulating factor
        modulating_factor = (1.0 - p_t) ** self.gamma

        # Combine all terms
        focal_loss = alpha_t * modulating_factor * bce_loss

        return focal_loss.mean()


class PlattScaler:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, logits, labels):
        logits = np.array(logits).reshape(-1, 1)
        labels = np.array(labels).ravel()
        self.model.fit(logits, labels)

    def predict_proba(self, logits):
        logits = np.array(logits).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]


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







# # --- Main function to run the training and evaluation ---
if __name__ == "__main__":
    # Hyperparameters
    WINDOW = 60
    BATCH = 64
    EPOCHS = 2000
    LR = 1e-3
    TEST_SIZE = 0.2
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print("Getting initial training data... \n")
    lstm_data, labels = data_get()
    if lstm_data is None or labels is None:
        print("Failed to get training training data \n")
        exit(1)

    print(f"Collected data for {len(lstm_data)} tokens \n")
    print(f"Have labels for {len(labels)} tokens \n")

    # Create dataset
    dataset = TokenDataset(lstm_data, labels)
    if len(dataset) == 0:
        print("No valid training data after filtering \n")
        exit(1)

    # Split into train/test
    train_size = int((1 - TEST_SIZE) * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(SEED)
    )
    class_criterion = FocalLoss(
        alpha=0.7, gamma=2.0
    )  # alpha=0.7 to focus more on positive class
    price_criterion = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH
    )  # Get input size from the first item in dataset
    sample_x, _, _ = dataset[0]
    input_size = sample_x.shape[1]  # Number of features

    model = LSTMModel(
        input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model_config = {
        "input_size": input_size,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "window_size": WINDOW,
    }
    with open("model_config.json", "w") as f:
        json.dump(model_config, f)

    print("\nStarting training... \n")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(
            model, train_loader, class_criterion, price_criterion, optimizer, device
        )
        test_loss, test_acc, test_mse = evaluate(
            model, test_loader, class_criterion, price_criterion, device
        )
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Class Acc: {test_acc:.2%} | "
            f"Price MSE: {test_mse:.4f}"
        )

    torch.save(model.state_dict(), "lstm_model.pt")
    print("Model saved as lstm_model.pt \n")

    # --- Train price model ---`
    print("Training dedicated price model... \n")
    price_model = PriceModel(
        input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2
    ).to(device)
    price_optimizer = torch.optim.Adam(price_model.parameters(), lr=LR)

    # Train price model
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_price(
            price_model, train_loader, price_criterion, price_optimizer, device
        )
        # You'll need to fix evaluate_price first (it has several bugs)
        # predictions, mse, avg_loss = evaluate_price(price_model, test_loader, price_criterion)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f}")

    torch.save(price_model.state_dict(), "price_model.pt")
    print("Price model saved as price_model.pt \n")

    # --- Evaluate price model ---
    print("Evaluating price model... \n")
    price_model.eval()
    with torch.no_grad():
        predictions, mse, avg_loss = evaluate_price(
            price_model, test_loader, price_criterion, device
        )  # Add device parameter
        print(f"Price Model MSE: {mse:.4f} | Avg Loss: {avg_loss:.4f} \n")
    # Save predictions and targets for further analysis
    np.savez("price_model_predictions.npz", predictions=predictions)
    print("Price model predictions saved as price_model_predictions.npz \n")

    # --- Fit and save Platt scaler ---
    print("Fitting Platt scaling... \n")
    _, _, _, val_logits, val_labels = evaluate_with_logits(
        model, test_loader, class_criterion, price_criterion, device
    )
    scaler = PlattScaler()
    scaler.fit(val_logits, val_labels)
    np.savez(
        "platt_scaler.npz",
        coef=scaler.model.coef_,
        intercept=scaler.model.intercept_,
        classes=scaler.model.classes_,
    )
    print("Platt scaler saved as platt_scaler.npz \n")

    print("Example usage: \n")
    print("from train_model import predict_next_hour \n")
    print("prediction, prob, price = predict_next_hour('lstm_model.pt') \n")
    print("print(f'Prediction: {prediction} | Prob: {prob:.2%} | Price: ${price:.2f}') \n")
