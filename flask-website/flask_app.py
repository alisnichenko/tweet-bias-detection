"""
This file contains the main logic for the flask app api.
The api will serve some html for presentation purposes, and
will send a request to the machine learning model for prediction.
"""
from flask import render_template
from flask import request
from flask import Flask
import tensorflow as tf
import numpy as np

app = Flask(__name__)
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
        return render_template('bias-detector.html')
    if request.method == 'POST':
        text = request.form['message']
        model = tf.keras.models.load_model('../data', compile=False)
        probability = model.predict([[text]])
        prediction = np.argmax(probability[0])
        return text + " (prediction " + str(prediction * 10) + "%)."

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
