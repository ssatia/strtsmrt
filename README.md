# strtsmrt

strtsmrt is a machine learning system that predicts stock prices based on news articles.

##How it works:

* Gets sentiment analysis data on news associated to stocks in the S&P 500 (using Google's Natural Language Cloud API)
* Performs gradient descent on a neural network learning model using TensorFlow
* Uses historical stock data fetched from Yahoo! Finance for training and testing purposes

##Results:
Using a day's news sources, strtsmrt predicts the next day's stock price with an average percentage error of __0.5349%__ using the neural network model with two hidden layers.
