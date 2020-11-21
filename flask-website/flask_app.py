"""
This file contains the main logic for the flask app api.
The api will serve some html for presentation purposes, and
will send a request to the machine learning model for prediction.
"""
from flask import render_template
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    """
    Default landing page. Will contain links to the remaining items of
    the website, in addition to some copyright and styling.
    """
    return render_template('index.html')

@app.route('/bias-detector')
def bias_detector():
    """
    Bias detector page that will accept the tweet into a text box and
    then use the pretrained end-to-end model to predict the bias
    percentage.
    """
    return render_template('bias-detector.html')

@app.route('/about')
def about():
    """
    About page that will introduce the team members, as well as provide
    some information about the project, machine learning, and research
    behind it.
    """
    return render_template('about.html')
