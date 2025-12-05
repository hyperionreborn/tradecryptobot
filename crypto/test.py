
from .model import LSTMModel, PriceModel
from .nlp import preprocess,generate_keywords,get_dominant_sentiment,contains_keyword
import torch
import json




def test_nlp():
    test_tweets = ["DOGE to the moon! 🚀 #dogecoin", "DOGE is shit", "DOGE is scam", "beware of DOGE RUG PULL","for those who dont believe in  DOGE chart looks almost like bitcoin chart","DOGE has potential"]
    for tweet in test_tweets:
        preprocessed = preprocess(tweet)
        generate_keywords("DOGE")
        if contains_keyword(preprocessed, "DOGE"):
            sentiment = get_dominant_sentiment(preprocessed)
            print(f"{tweet} sentiment is {sentiment} ")
