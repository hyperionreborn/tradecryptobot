
from .model import LSTMModel, PriceModel
from .data_fetch import json_get
from .nlp import preprocess,generate_keywords,get_dominant_sentiment,contains_keyword
import torch
import json

def test_predictions(lstm_data_path, model_path='lstm_model.pt', price_model_path='price_model.pt'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data
    lstm_data = json_get(lstm_data_path)

    # Load model config
    with open('model_config.json', 'r') as f:
        config = json.load(f)

    # Initialize models
    binary_model = LSTMModel(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers']
    ).to(device)

    binary_model.load_state_dict(torch.load(model_path))
    binary_model.eval()

    price_model = PriceModel(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    price_model.load_state_dict(torch.load(price_model_path))
    price_model.eval()

    # Test predictions
    results = []
    with torch.no_grad():
        for token, data in lstm_data.items():
            X = torch.FloatTensor(data).unsqueeze(0).to(device)
            logits, pred_price = binary_model(X)
            predicted_price = price_model(X)

            prob = torch.sigmoid(logits).item()
            prediction = "UP" if prob > 0.5 else "DOWN"

            results.append({
                'token': token,
                'prediction': prediction,
                'probability': prob,
                'binary_price': pred_price.item(),
                'model_price': predicted_price.item()
            })

    return results


def test_nlp():
    test_tweets = ["DOGE to the moon! 🚀 #dogecoin", "DOGE is shit", "DOGE is scam", "beware of DOGE RUG PULL","for those who dont believe in  DOGE chart looks almost like bitcoin chart","DOGE has potential"]
    for tweet in test_tweets:
        preprocessed = preprocess(tweet)
        generate_keywords("DOGE")
        if contains_keyword(preprocessed, "DOGE"):
            sentiment = get_dominant_sentiment(preprocessed)
            print(f"{tweet} sentiment is {sentiment} ")
