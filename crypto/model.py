import torch
import torch.nn as nn
import pandas as pd
import numpy as np

def create_sequences(df: pd.DataFrame, window: int = 30):
    """
    Tworzy sekwencje danych z logarytmicznych zwrotów dla LSTM.
    Zwraca: X.shape=(n, window, 1), y.shape=(n,)
    """
    X, y = [], []
    returns = np.log(df['close'] / df['close'].shift(1)).fillna(0).values
    for i in range(len(df) - window - 1):
        seq = returns[i : i + window]
        label = 1 if returns[i + window + 1] > 0 else 0
        X.append(seq.reshape(-1, 1))  # (window, 1)
        y.append(label)
    return np.array(X), np.array(y)

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

# Test if code runs
if __name__ == "__main__":
    # Sztuczne dane: 200 próbek z 9 cechami (np. OHLC, wolumen, itp.)
    dummy_df = pd.DataFrame(np.random.rand(200, 9), columns=[f"f{i}" for i in range(9)])
    window = 60

    # Tworzenie sekwencji
    X = []
    y = []
    for i in range(len(dummy_df) - window - 1):
        seq = dummy_df.iloc[i : i + window].values
        label = 1 if dummy_df.iloc[i + window + 1, 0] > dummy_df.iloc[i + window, 0] else 0
        X.append(seq)
        y.append(label)
    X = np.array(X)
    y = np.array(y)

    print("X.shape:", X.shape, "y.mean:", y.mean())

    # Model
    model = LSTMModel(input_size=7, hidden_size=64)
    model.eval()  # tryb ewaluacyjny

    # Przykładowa predykcja na jednej próbce
    sample = torch.tensor(X[-1:], dtype=torch.float32)  # (1, window, features)
    with torch.no_grad():
        logits, pred_price = model(sample)
        prob = torch.sigmoid(logits).item()  # konwersja logit -> prawdopodobieństwo
        price_prediction = pred_price.item()
        prediction = "UP" if prob > 0.5 else "DOWN"

    print(f"Prediction: {prediction}, Prob: {prob:.4f}, Price_pred: {price_prediction:.2f}")
