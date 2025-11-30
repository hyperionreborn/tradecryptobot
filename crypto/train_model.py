import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import json
import random
from datetime import datetime
from pathlib import Path

from .model import LSTMModel, PriceModel
from .data_fetch import (
    data_get,
    json_get_merged_tokens,
    take_snapshot,
    convert,
    collect_data_for_tokens,
)


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
        self.y_change = self.y_prices / safe_current_prices

        print(f"NumpyDataset loaded: {len(self.X)} samples")
        print(f"Positive labels: {self.y_class.sum().item()}/{len(self.y_class)} ({self.y_class.mean().item():.2%})")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_class[idx].unsqueeze(0), self.y_change[idx].unsqueeze(0)


def train_from_dataset(outdir="./dataset", epochs=100, batch_size=64, lr=2e-3):
    outdir_path = Path(outdir)
    x_path = outdir_path / "X.npy"
    y_path = outdir_path / "y.npy"
    features_path = outdir_path / "features.json"
    
    if not x_path.exists() or not y_path.exists():
        print(f"Dataset not found in {outdir}")
        return
        
    dataset = NumpyDataset(x_path, y_path, features_path)
    
    # Split into train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model setup
    input_size = dataset.X.shape[2]
    model = LSTMModel(input_size=input_size, hidden_size=256, num_layers=3, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)
    
    class_criterion = FocalLoss(alpha=0.7, gamma=3.0)
    price_criterion = nn.MSELoss()
    
    best_accuracy = 0.0
    
    print("Starting training from numpy dataset...")
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, class_criterion, price_criterion, optimizer, device, alpha=2.0, beta=0.0)
        test_loss, test_acc, test_mse = evaluate(model, test_loader, class_criterion, price_criterion, device)
        
        scheduler.step()
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), "lstm_model_best.pt")
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.2%} | Best: {best_accuracy:.2%}")
            
    print(f"Training complete. Best accuracy: {best_accuracy:.2%}")
    # Save final model
    torch.save(model.state_dict(), "lstm_model.pt")


# # --- Main function to run the training and evaluation to call from main.py ---
def TrainAll(hours_collect=1, change=1):
    WINDOW = 60 * int(hours_collect)
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
    lstm_data = json_get_merged_tokens(f"lstm{hours_collect}h_{change}h")
    labels = json_get_merged_tokens(f"lstm_labels{hours_collect}h_{change}h")
    if lstm_data is None or labels is None:
        print("Failed to get training data. Please run data collection first:\n")
        print(
            f"python main.py --data_fetch --hours_collect {hours_collect} --change {change}\n"
        )
        return False

    print(f"Collected data for {len(lstm_data)} tokens\n")
    print(f"Have labels for {len(labels)} tokens\n")

    # Create dataset
    dataset = TokenDataset(lstm_data, labels)

    if len(dataset) == 0:
        print("No valid training data after filtering\n")
        exit(1)

    # Calculate class weights for better balancing
    all_labels = [dataset[i][1].item() for i in range(len(dataset))]
    positive_ratio = sum(all_labels) / len(all_labels)
    negative_ratio = 1 - positive_ratio
    print(
        f"Class distribution: {positive_ratio:.2%} positive, {negative_ratio:.2%} negative\n"
    )

    # Adjust alpha based on class imbalance - focus more on minority class
    focal_alpha = min(0.8, max(0.2, negative_ratio))
    print(f"Using focal loss alpha: {focal_alpha:.2f}\n")

    # Split into train/test
    train_size = int((1 - TEST_SIZE) * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(SEED)
    )

    # More aggressive focal loss for hard examples
    class_criterion = FocalLoss(alpha=focal_alpha, gamma=3.)  # Increased gamma
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
        "window_size": WINDOW,
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
    model.load_state_dict(torch.load("lstm_model_best.pt"))
    torch.save(model.state_dict(), f"{hours_collect}_{change}lstm_model.pt")
    print(f"Model saved as lstm_model.pt with best accuracy: {best_accuracy:.2%}\n")
    price_model_config = {
        "input_size": input_size,
        "hidden_size": 256,  # Increased
        "num_layers": 3,  # Increased
        "dropout": 0.2,  # Increased
        "window_size": WINDOW,
    }

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


def Predict(
    binarymodel_path: str = "lstm_model.pt",
    pricemodel_path: str = "price_model.pt",
    collect_time=1,
):  ## i will try to write the predicting function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("model_config.json", "r") as f:
        config = json.load(f)

    binary_model = LSTMModel(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    binary_model.load_state_dict(torch.load(binarymodel_path))
    binary_model.eval()
    with open("price_model_config.json", "r") as p:
        price_config = json.load(p)

    price_model = PriceModel(
        input_size=price_config["input_size"],
        hidden_size=price_config["hidden_size"],
        num_layers=price_config["num_layers"],
        dropout=price_config["dropout"],
    ).to(device)
    price_model.load_state_dict(torch.load(pricemodel_path))
    price_model.eval()
    tokens = take_snapshot()
    collected = collect_data_for_tokens(tokens, max_data_points=60 * collect_time)
    data = convert(collected)
    for token in tokens:
        current_time = datetime.now().strftime("%H:%M")
        token_address = token["tokenAddress"]
        X = data[token_address]
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = binary_model(X_tensor)
        raw_prob = torch.sigmoid(logits).item()
        prediction = "UP" if raw_prob > 0.5 else "DOWN"
        print(
            f"Token:{token_address} will go {prediction} with probability  to go up {raw_prob}  time:{current_time} \n"
        )


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
