import config
import tweepy

auth = tweepy.OAuthHandler(config.api_key, config.api_key_secret)
auth.set_access_token(config.access_token, config.access_token_secret)
api = tweepy.API(auth)


# # search about a tweet by hashtag
# tweets = api.search("#Trump", count=100)
#
# for tweet in tweets:
#     print(tweet.text)