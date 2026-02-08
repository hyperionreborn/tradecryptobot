import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timezone
from pathlib import Path
import json



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2,dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=dropout)
        self.fc_class = nn.Linear(hidden_size, 1)  # surowe logity
        self.fc_price = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # ostatni krok
        class_logits = self.fc_class(out)  # logity (nie sigmoid!)
        price_out = self.fc_price(out)
        return class_logits, price_out


class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Dodaj jedną warstwę pośrednią
        self.fc_mid = nn.Linear(hidden_size * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc_class = nn.Linear(64, 1)
        self.fc_price = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]

        # Dodaj non-linearity
        out = self.fc_mid(out)
        out = self.relu(out)
        out = self.dropout(out)

        class_logits = self.fc_class(out)
        price_out = self.fc_price(out)
        return class_logits, price_out

class BinaryModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc_class = nn.Linear(hidden_size, 1)  # surowe logity

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # ostatni krok
        class_logits = self.fc_class(out)  # logity (nie sigmoid!)
        return class_logits

class PriceModel(nn.Module):
    """
    Przewidywanie ceny 
    """
    def __init__(self,input_size=9,hidden_size=64,num_layers=2,dropout=0.2):
        super(PriceModel,self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size,1)

    def forward(self,x):
        out, _ = self.lstm(x)
        last_step = out[:,-1,:]
        return self.fc(last_step)


class CryptoEnsemble:
    """
    Load multiple trained models and make ensemble predictions
    """

    def __init__(self, dataset_dir,model_type="binary_model", device='cpu'):
        self.device = device
        self.models = []
        self.price_models = []
        self.scalers = []
        self.weights = []

        # Load ensemble info
        ensemble_path = Path(dataset_dir) / "ensemble_info.json"
        if not ensemble_path.exists():
            raise FileNotFoundError(f"No ensemble info found at {ensemble_path}")

        with open(ensemble_path, 'r') as f:
            self.info = json.load(f)

        # Load model config
        config_path = Path(dataset_dir) / "model_config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Load each model
        for i, seed in enumerate(self.info['seeds']):
            # Load scaler
            scaler_path = Path(dataset_dir) / "scaler.pkl"
            scaler = joblib.load(scaler_path)
            self.scalers.append(scaler)

            # Load model
            model = ImprovedLSTMModel(
                input_size=self.config['input_size'],
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout']
            ).to(device)

            model_path = Path(dataset_dir) / f"{seed}_{model_type}.pt"
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            self.models.append(model)

        # Get weights (use validation accuracy)
        self.weights = np.array(self.info.get('weights', [1.0 / len(self.models)] * len(self.models)))
        print(f"✅ Loaded ensemble with {len(self.models)} models")
        print(f"   Weights: {self.weights}")

    def predict(self, X_raw):
        """
        X_raw: numpy array of shape (T, F) - raw unscaled window
        Returns: ensemble probability and predicted change
        """
        all_probs = []
        all_changes = []

        for i, (model, scaler) in enumerate(zip(self.models, self.scalers)):
            # Scale using this model's scaler
            T, F = X_raw.shape
            X_scaled = scaler.transform(X_raw.reshape(-1, F)).reshape(1, T, F)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

            # Predict
            with torch.no_grad():
                logits, pred_change = model(X_tensor)
                prob = torch.sigmoid(logits).item()
                change = pred_change.item()

            all_probs.append(prob)
            all_changes.append(change)

        # Weighted average
        ensemble_prob = np.average(all_probs, weights=self.weights)
        ensemble_change = np.average(all_changes, weights=self.weights)

        # Calculate agreement (standard deviation of predictions) # Normalized 0-1

        return ensemble_prob,ensemble_change

# Test if code runs
