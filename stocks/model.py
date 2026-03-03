import torch
import torch.nn as nn


class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.feature_gate = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Sigmoid(),
        )
        self.temp_conv = nn.Conv1d(
            in_channels=hidden_size * 2,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
        )
        self.combine_layer = nn.Linear(hidden_size * 4, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.class_head = nn.Linear(128, 1)
        self.price_head = nn.Linear(128, 1)

    def forward(self, x):
        feature_importance = self.feature_gate(x)
        weighted_input = x * feature_importance

        lstm_out, _ = self.lstm(weighted_input)
        conv_input = lstm_out.permute(0, 2, 1)
        conv_out = self.temp_conv(conv_input).permute(0, 2, 1)

        lstm_last = lstm_out[:, -1, :]
        conv_last = conv_out[:, -1, :]
        avg_pool = conv_out.mean(dim=1)
        combined = torch.cat([lstm_last, conv_last, avg_pool], dim=1)

        out = self.combine_layer(combined)
        out = self.relu(out)
        out = self.dropout(out)
        return self.class_head(out), self.price_head(out)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(inputs, targets)
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        modulating_factor = (1.0 - p_t) ** self.gamma
        focal_loss = alpha_t * modulating_factor * bce_loss
        return focal_loss.mean()
