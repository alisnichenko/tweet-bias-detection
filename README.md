# tweet-bias-detection
Twitter's #Codechella submission for bias detection in tweets using Python and RNNs.  
The project was done over the course of 2 days by @alisnichenko and @enbattle.

# Overview
Twitter and social media are amazing. Misinformation and lack of trust are not. This is why we decided to tackle the problem of bias detection using machine learning, Python programming language, and Twitter API. Introducing Tweet Bias Detection: a tool used to detect bias in tweets and tweet-like messages by analyzing their word structures and assigning them a percentage from 0% to 100%, where 0% is not biased at all and 100% is extremely biased. We hope that this prototype will show some potential in application of this topic and will interest someone like Twitter in incorporating our idea into their products.

# Technologies used
Here is a list of technologies that were utilized in the project:

1. Python (programming language).
2. Flask (web framework for Python).
3. TensorFlow/Keras (machine learning framework for Python).
4. HTML/CSS (web-based presentation component).
5. Digital Ocean (web servers and hosting).
6. Tech domains (DNS and domains).
7. The Internet.

# Files overview
`data/` contains a `csv` file with relevant Twitter API data, a `saved_model.pb`, which is a trained end-to-end machine learning that is being loaded for prediction, and other directories that were supplied there during export of the model.  

`tweet-detection-bias/` contains the main code related to the development of the model and the machine learning part. `tweet_bias_detection_model.py` creates the pretrained model using our architecture and pretraining embeddings for word vectors. The pretraining dataset (which is Stanford's GLoVe dataset) **is not included** due to being too large. `tweet_bias_detection_utils.py` is used for data collection using Twitter API. It stores the results in `../data/`. `unnecessary_utils.py` - don't ask.

`flask-website/` contains the logic behind the website used for demonstration purposes. `flask_app.py` within this directory contains main `flask` logic that displays the content and routes requests.

`requirements.txt` contains all the modules used by existing Python code.

# Installation instructions
Here is the almost-universal step-by-step installation guide for testing our project. Keep in mind that some directions are specific to the Ubuntu Linux distribution.  

1. Install required modules.
    1. Install `python3` and `pip`.
    2. Install virtual environment module: `pip install virtualenv`.
    3. Create virtual environment in `tweet-bias-detection/`: `python3 -m venv venv`.
    4. Install required packages: `pip install -r requirements.txt`.
2. Run the flask app. Run from the `flask-website/` directory.
    1. Export flask variable for testing: `export FLASK_APP=flask_app.py`.
    2. Run the flask server: `flask run`.
    3. Enter the address provided in the browser (usually it's `127.0.0.1:5000`).
3. The pages `/about` and `/bias-detector` should be available for testing.

# Resources
Here is a list of resources that we heavily used during our development.

**Research**
- Political bias detection arxiv [paper](https://arxiv.org/pdf/2010.10652.pdf).
- COVID 19 bias and misinformation [paper](https://arxiv.org/pdf/2003.12309.pdf).

**Tutorials**
- Sentiment analysis on tweets using rnn [blog](https://medium.com/@gabriel.mayers/sentiment-analysis-from-tweets-using-recurrent-neural-networks-ebf6c202b9d5).
- Keras document page that described using [embeddings](https://keras.io/examples/nlp/pretrained_word_embeddings/).

**Information**
- Codechella's hacker map [document](https://www.notion.so/Codechella-Hacker-Map-1bc32d1fba4547ed98d81cc3ca31dfb3).

# Moving forward
- [ ] Better website
- [ ] More data
- [ ] Better models
- [ ] Sell to Twitter
