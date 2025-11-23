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
def TrainAll():

    BATCH = 64
    EPOCHS = 100
    LR = 2e-3  # Slightly higher learning rate
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

    dataset = PriceDataset("dataset")




    if len(dataset) == 0:
        print("No valid training data\n")
        exit(1)

    # Calculate class weights for better balancing


    # Split into train/test
    train_size = int((1 - TEST_SIZE) * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(SEED)
    )

    # More aggressive focal loss for hard examples
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
            torch.save(model.state_dict(), "lstm_model_best.pt")
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


    print(f"Model saved as lstm_model.pt with best accuracy: {best_accuracy:.2%}\n")









