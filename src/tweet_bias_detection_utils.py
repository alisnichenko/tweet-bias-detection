"""
This file contains utility functions used for the models and the
project overall. We don't have time to create a full-fledged structure,
so we will have a couple of main files and a helper file.
"""
import pandas as pd
import tweepy
import re
import os

from config import get_tweepy_config
from config import get_accounts_of_interest

def get_tweepy_api(conf: dict()) -> None:
    """
    Returns twitter api wrapper object by using the config object
    provided from a local function storing the api keys. Used for
    accessing the api and the object to interact with.
    """
    auth = tweepy.OAuthHandler(conf['api_key'], conf['api_key_secret'])
    auth.set_access_token(conf['access_token'], conf['access_token_secret'])
    return tweepy.API(auth)

def get_tweets_csv() -> None:
    """
    Creates files for each of the tweets in an appropriate folder. Folder names
    consist of the account name and the weight (label) assigned to it. File
    names consist of the account name and the id of the tweet, and the name.
    Located in ../data/.
    """
    # Gets api object using custom function with data config.
    api = get_tweepy_api(get_tweepy_config())
    accounts = get_accounts_of_interest()
    biases_col, tweets_col = list(), list()
    for account, bias in accounts.items():
        # Gets 200 tweets for the username account.
        tweets = api.user_timeline(screen_name=account, count=200,
            include_rts=False)
        for tweet in tweets:
            clean_tweet_text = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet.text,
                flags=re.MULTILINE)
            if len(clean_tweet_text) > 49:
                biases_col.append(bias)
                tweets_col.append(clean_tweet_text)
    # Create a dataframe using pandas.
    tweets_dict = dict({'biases': biases_col, 'tweets': tweets_col})
    tweets_df = pd.DataFrame(data=tweets_dict)
    tweets_df.to_csv('../data/tweets_biases.csv')

if __name__ == '__main__':
    get_tweets_csv()
