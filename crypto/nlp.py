import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

NLP_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
nlp_model = AutoModelForSequenceClassification.from_pretrained(NLP_MODEL)
tokenizer = AutoTokenizer.from_pretrained(NLP_MODEL)
nlp_model.eval()
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
def generate_keywords(coin_name,ticker = None):
    if ticker is None:
        ticker = coin_name[:4].upper()

    variants = [
        coin_name.lower(),
        coin_name.lower().capitalize(),
        coin_name.upper(),
        ticker.lower(),
        ticker.lower().capitalize(),
        ticker.upper(),
        f"${ticker.upper()}",
        f"${ticker.lower()}"
    ]
    return list(set(variants))
def preprocess(tweet):

    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)


    tweet = re.sub(r"@\w+", "", tweet)


    tweet = tweet.replace("#", "")


    tweet = re.sub(r"\s+", " ", tweet).strip()

    return tweet



def filter_tweets_by_keywords(tweets_df, keywords):

    pattern = "|".join([re.escape(k) for k in keywords])
    mask = tweets_df["content"].str.contains(pattern, case=False, na=False)
    return tweets_df[mask]

def contains_keyword(text, keywords):
    pattern = "|".join([re.escape(k) for k in keywords])
    return bool(re.search(pattern, text))
def get_sentiment_for_tweet(tweet,coin,ticker=None):
    tweet = preprocess(tweet)
    generate_keywords(coin)
    if contains_keyword(tweet,coin):
        sentiment = get_dominant_sentiment(tweet)
        print(f"{coin} sentiment is {sentiment} ")
    return

def get_dominant_sentiment(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = nlp_model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)


    probs_list = probs[0].tolist()
    max_index = probs_list.index(max(probs_list))
    return  SENTIMENT_LABELS[max_index]



def get_sentiments(tweets_df,coin,ticker=None):
    keywords = generate_keywords(coin,ticker)
    tweets_df["content"] = tweets_df["content"].apply(preprocess)
    tweets_df = filter_tweets_by_keywords(tweets_df, keywords)
    if tweets_df.empty:
        return None
    tweets_df["dominant_sentiment"] = tweets_df["content"].apply(get_dominant_sentiment)
    sentiment_counts = tweets_df["dominant_sentiment"].value_counts()
    num_positive = sentiment_counts.get("positive", 0)
    num_neutral = sentiment_counts.get("neutral", 0)
    num_negative = sentiment_counts.get("negative", 0)

    print(f"positive {num_positive} neutral {num_neutral} negative {num_negative}")
    return sentiment_counts
def analyze_coin_description(description):
    return





