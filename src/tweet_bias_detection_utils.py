"""
This file contains utility functions used for the models and the
project overall. We don't have time to create a full-fledged structure,
so we will have a couple of main files and a helper file.
"""
import tweepy
import re
import os

from config import get_tweepy_config
from config import get_accounts_of_interest

def get_tweepy_api(conf: dict()) -> tweepy.API:
    """
    Returns twitter api wrapper object by using the config object
    provided from a local function storing the api keys. Used for
    accessing the api and the object to interact with.
    """
    auth = tweepy.OAuthHandler(conf['api_key'], conf['api_key_secret'])
    auth.set_access_token(conf['access_token'], conf['access_token_secret'])
    return tweepy.API(auth)

def get_tweets_users() -> None:
    """
    Creates files for each of the tweets in an appropriate folder. Folder names
    consist of the account name and the weight (label) assigned to it. File
    names consist of the account name and the id of the tweet, and the name.
    Located in ../data/.
    """
    # Gets api object using custom function with data config.
    api = get_tweepy_api(get_tweepy_config())
    accounts = get_accounts_of_interest()
    for account, bias in accounts.items():
        # Makes directory in ../data/.
        acc_dir = '../data/{0}_{1}'.format(account, bias)
        os.mkdir(acc_dir)
        # Gets 200 tweets for the username account.
        tweets = api.user_timeline(screen_name=account, count=200,
            include_rts=False)
        for tweet in tweets:
            clean_tweet_text = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet.text,
                flags=re.MULTILINE)
            if len(clean_tweet_text) > 49:
                acc_file_dir = '{0}/{1}_{2}_{3}'.format(acc_dir, account,
                    tweet.id_str, bias)
                with open(acc_file_dir, 'w') as f:
                    f.write(clean_tweet_text)

def show_data_dirs() -> None:
    """
    Prints out the folders, the files, and the information about them.
    Will be used to monitor the quality of the data and the amount of data.
    """
    for dirname in sorted(os.listdir('../data/')):
        dirpath = '../data/' + dirname
        fnames = os.listdir(dirpath)
        print("Processing {0}, {1} files found.".format(dirname, len(fnames)))
        # Other processing procedures should follow.

if __name__ == '__main__':
    show_data_dirs()