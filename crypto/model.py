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


import torch
import torch.nn as nn
import torch.nn.functional as F





# Version with both improvements but simpler combination
class ImprovedLSTMModel(nn.Module):
    """
    Simple enhanced LSTM with both improvements but cleaner architecture
    """

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()

        # 1. LSTM (keep as is)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # 2. FIXED Feature Gating - weight EACH feature separately
        # Input: [batch, seq_len, features] → Output: [batch, seq_len, features]
        self.feature_gate = nn.Sequential(
            nn.Linear(input_size, input_size),  # Same size for per-feature weights
            nn.Sigmoid()
        )

        # 3. Temporal Convolution (kernel_size=3 is better than 5 for crypto)
        self.temp_conv = nn.Conv1d(
            in_channels=hidden_size * 2,  # From bidirectional LSTM
            out_channels=hidden_size,
            kernel_size=3,  # Better for crypto patterns
            padding=1
        )

        # 4. FIXED: hidden_size * 4 (not *3)
        # lstm_last: hidden*2, conv_last: hidden, avg_pool: hidden = hidden*4
        self.combine_layer = nn.Linear(hidden_size * 4, 128)

        # 5. Processing in correct order
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 6. Output heads
        self.class_head = nn.Linear(128, 1)
        self.price_head = nn.Linear(128, 1)

    def forward(self, x):
        batch_size, seq_len, features = x.shape

        # ----- Feature Gating (FIXED) -----
        # Give EACH feature its own importance weight
        # x shape: [batch, seq_len, features]
        feature_importance = self.feature_gate(x)  # [batch, seq_len, features]
        weighted_input = x * feature_importance  # [batch, seq_len, features]

        # ----- LSTM Processing -----
        lstm_out, _ = self.lstm(weighted_input)  # [batch, seq_len, hidden*2]

        # ----- Temporal Convolution -----
        conv_input = lstm_out.permute(0, 2, 1)  # [batch, hidden*2, seq_len]
        conv_out = self.temp_conv(conv_input)  # [batch, hidden, seq_len]
        conv_out = conv_out.permute(0, 2, 1)  # [batch, seq_len, hidden]

        # ----- Combine Features (FIXED dimensions) -----
        lstm_last = lstm_out[:, -1, :]  # [batch, hidden*2]
        conv_last = conv_out[:, -1, :]  # [batch, hidden]
        avg_pool = conv_out.mean(dim=1)  # [batch, hidden]

        # CORRECT: hidden*2 + hidden + hidden = hidden*4
        combined = torch.cat([lstm_last, conv_last, avg_pool], dim=1)

        # ----- Final Processing -----
        out = self.combine_layer(combined)  # [batch, 128]
        out = self.relu(out)
        out = self.dropout(out)

        return self.class_head(out), self.price_head(out)


# Example usage in your training code:


    # Keep everything else the same!
class EImprovedLSTMModel(nn.Module):
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