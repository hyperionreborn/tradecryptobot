# tradecryptobot



A machine learning–based crypto trading system built with PyTorch and the Binance API. \

The project:  

Downloads historical market data from Binance  

Builds advanced technical features  

Trains LSTM-based deep learning models  

Performs backtesting   

Uses confidence filtering for trading  

# Usage:

python main.py --backtest --symbol all  

## Performs rolling retraining with cutoff shifts and prints statistics.  


  
python main.py --grid_search --symbol BTCUSDT  

## Performs grid search  


  
python main.py --train --symbol all  

## Train all symbols  


  
python main.py --data_fetch  --symbol BTCUSDT  --months 8 --window_days 28  --resample_hours 4  --horizon 3  

## Fetch the data

# Pre requisites

Python 3.8 or higher




# How to work with the code

1. Clone the repository
2. Install the requirements
3. `uv pip sync`
4. `./.venv/Scripts/activate`
And then run the code
