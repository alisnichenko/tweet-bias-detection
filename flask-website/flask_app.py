"""
This file contains the main logic for the flask app api.
The api will serve some html for presentation purposes, and
will send a request to the machine learning model for prediction.
"""
from flask import render_template
from flask import request
from flask import Flask
from flask import redirect, url_for
import tensorflow as tf
import numpy as np
import requests
import json
from requests_oauthlib import OAuth1
from datetime import datetime as dt

from config import get_tweepy_config

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/')
def index():
    """
    Default landing page. Will contain links to the remaining items of
    the website, in addition to some copyright and styling.
    """
    return render_template('index.html')

@app.route('/bias-detector', methods=['GET', 'POST'])
def bias_detector():
    """
    Bias detector page that will accept the tweet into a text box and
    then use the pretrained end-to-end model to predict the bias
    percentage.
    """

    if request.method == 'GET':
        return render_template('bias-detector.html',
                               message=request.args.get("message"),
                               bias=request.args.get("bias"),
                               tweet_url=request.args.get("tweet_url"),
                               user_message=request.args.get("user_message"),
                               user_datetime=request.args.get("user_datetime"),
                               user_name=request.args.get("user_name"),
                               user_handle=request.args.get("user_handle"),
                               user_profile_pic=request.args.get("user_profile_pic"),
                               user_prediction=request.args.get("user_prediction"),
                               message_error=request.args.get("message_error"),
                               tweet_error=request.args.get("tweet_error"))

    if request.method == 'POST':
        if(request.form['message'] != "" and request.form['tweetURL'] != ""):

            """:
            Predict Bias of Message
            """
            # Receive requested message
            text = request.form['message']

            # Run machine learning model
            model = tf.keras.models.load_model('../data', compile=False)
            probability = model.predict([[text]])
            prediction = np.argmax(probability[0])
            quote = text
            prediction = "Bias prediction: " + str(prediction * 10) + "%."

            """
            Predict Bias of Tweet
            """
            # Receive requested URL
            url = request.form['tweetURL']
            id = url.split("status/", 1)[1]

            # Authentication
            auth = OAuth1(get_tweepy_config()["api_key"],
                          get_tweepy_config()["api_key_secret"],
                          get_tweepy_config()["access_token"],
                          get_tweepy_config()["access_token_secret"])

            # Set up HTTP Get Request to the Twitter API
            url = "https://api.twitter.com/1.1/statuses/show.json?id=" + str(id)
            params = {"tweet_mode": "extended"}
            tweet = requests.get(url, auth=auth, params=params)
            tweet = json.loads(tweet.text)
            dtime = tweet["created_at"]

            # Choose needed information
            user_message = tweet["full_text"]
            user_datetime = dt.strftime(dt.strptime(dtime, '%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d')
            user_name = tweet["user"]["name"]
            user_handle = tweet["user"]["screen_name"]
            user_profile_pic = tweet["user"]["profile_image_url_https"]

            # Run machinel learning model
            tweet_probability = model.predict([[user_message]])
            tweet_prediction = np.argmax(tweet_probability[0])
            tweet_prediction = "Bias prediction: " + str(tweet_prediction * 10) + "%."

            # Send parameters to the client-sie
            return redirect(url_for('bias_detector',
                                    message=request.form['message'],
                                    bias=prediction,
                                    user_message=user_message,
                                    user_datetime=user_datetime,
                                    user_name=user_name,
                                    user_handle=user_handle,
                                    user_profile_pic=user_profile_pic,
                                    user_prediction=tweet_prediction,
                                    tweet_url=request.form['tweetURL']))

        else:
            # Check for errors (User did not input a field)
            if(request.form['message'] == "" and request.form['tweetURL'] == ""):
                return redirect(url_for('bias_detector',
                                        message_error="Please enter a message!",
                                        tweet_error="Please enter a tweet!"))
            elif(request.form['message'] == ""):
                return redirect(url_for('bias_detector', message_error="Please enter a message!", tweet_url=request.form['tweetURL']))
            elif(request.form['tweetURL'] == ""):
                return redirect(url_for('bias_detector', message=request.form['message'], tweet_error="Please enter a tweet"))

@app.route('/about')
def about():
    """
    About page that will introduce the team members, as well as provide
    some information about the project, machine learning, and research
    behind it.
    """
    return render_template('about.html')

if __name__ == '__main__':
    app.run()
