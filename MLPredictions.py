#!/usr/bin/env python3

import numpy as np
import logging
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot as plt
from oandapyV20.endpoints.instruments import InstrumentsCandles
from pandas.io.json import json_normalize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR


logging.basicConfig(
    filename='/var/log/Prediction.log',
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s : %(message)s',)

class MLAlgorithms(object):

    def radialBasis(symbol, ohlcd, api, prediction_days):
        candle = InstrumentsCandles(instrument=symbol,params=ohlcd)
        api.request(candle)

        price = float(candle.response['candles'][-1]['mid']['c'])
        df = pd.DataFrame.from_dict(json_normalize(candle.response['candles']))

        for column in 'mid.c','mid.h','mid.l','mid.o':
            df[column] = df[column].astype(float)

        dfi = df.copy()

        #A variable for predicting 'n' days out into the future
        prediction_days = 1 #n = 30 days

        #Create another column (the target or dependent variable) shifted 'n' units up
        df['Prediction'] = df[['mid.c']].shift(-prediction_days)

        #CREATE THE INDEPENDENT DATA SET (X)
        df.drop(columns=['complete','volume','time','mid.o','mid.h','mid.l'], inplace=True)

        # Convert the dataframe to a numpy array and drop the prediction column
        X = np.array(df.drop(['Prediction'],1))

        #Remove the last 'n' rows where 'n' is the prediction_days
        X= X[:len(df)-prediction_days]

        #CREATE THE DEPENDENT DATA SET (y)

        # Convert the dataframe to a numpy array (All of the values including the NaN's)
        y = np.array(df['Prediction'])

        # Get all of the y values except the last 'n' rows
        y = y[:-prediction_days]

        # Split the data into 80% training and 20% testing
    #   x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.16)

        # Set prediction_days_array equal to the last 30 rows of the original data set from the price column
        prediction_days_array = np.array(df.drop(['Prediction'],1))[-prediction_days:]

        # Create and train the Support Vector Machine (Regression) using the radial basis function
        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001)
#       svr_rbf = SVR(kernel='rbf', gamma=1, tol=1e-3, C=1.0, epsilon=0.1)#, verbose=True)
        svr_rbf.fit(x_train, y_train)

        # Testing Model: Score returns the accuracy of the prediction.
        # The best possible score is 1.0
        svr_rbf_confidence = svr_rbf.score(x_test, y_test)
#       print("svr_rbf accuracy: ", svr_rbf_confidence)

        # Print the predicted value
        svm_prediction = svr_rbf.predict(x_test)
    #   print(svm_prediction)


        #Print the actual values
    #   print(y_test)

        # Print the model predictions for the next 'n' days
        svm_prediction = svr_rbf.predict(prediction_days_array)
#       print(svm_prediction)

        #Print the actual price for the next 'n' days, n=prediction_days=30
    #   print(df)
        return svr_rbf_confidence, svm_prediction, price, dfi


    def lstm(symbol, ohlcd, api, prediction_days):
        min_max_scaler = MinMaxScaler()
        candle = InstrumentsCandles(instrument=symbol,params=ohlcd)
        api.request(candle)

        price = float(candle.response['candles'][-1]['mid']['c'])
        df = pd.DataFrame.from_dict(json_normalize(candle.response['candles']))

        for column in 'mid.c','mid.h','mid.l','mid.o':
            df[column] = df[column].astype(float)

        dfi = df.copy()
#           print(df)

        #A variable for predicting 'n' days out into the future
#           prediction_days = 30 #n = 30 days

        #Create another column (the target or dependent variable) shifted 'n' units up
#           df['Prediction'] = df[['mid.c']].shift(-prediction_days)

        #CREATE THE INDEPENDENT DATA SET (X)
        df_norm = df.drop(columns=['complete','volume','time','mid.o','mid.h','mid.l'], inplace=True)


#           df = pd.read_csv("BitcoinPrice.csv")
#           df_norm = df.drop(['Date'], 1, inplace=True)

#           prediction_days = 30

        df_train= df[:len(df)-prediction_days]
        df_test= df[len(df)-prediction_days:]

        training_set = df_train.values
        training_set = min_max_scaler.fit_transform(training_set)

        x_train = training_set[0:len(training_set)-1]
        y_train = training_set[1:len(training_set)]
        x_train = np.reshape(x_train, (len(x_train), 1, 1))

        num_units = 1
#       activation = 'sigmoid'
        activation='softmax'
        optimizer = 'adam'
        loss_function = 'mean_squared_error'
        batch_size = 5
        num_epochs = 100
        metrics=['accuracy']

# Initialize the RNN
        regressor = Sequential()

# Adding the input layer and the LSTM layer
        regressor.add(LSTM(units = num_units, activation = activation, input_shape=(None, 1)))

# Adding the output layer
        regressor.add(Dense(units = 1))

# Compiling the RNN
        regressor.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

# Using the training set to train the model
        regressor.fit(x_train, y_train, batch_size = batch_size, epochs = num_epochs, validation_split = 0.01, shuffle=False)

        test_set = df_test.values

        inputs = np.reshape(test_set, (len(test_set), 1))
        inputs = min_max_scaler.transform(inputs)
        inputs = np.reshape(inputs, (len(inputs), 1, 1))

        predicted_price = regressor.predict(inputs)
        predicted_price = min_max_scaler.inverse_transform(predicted_price)

        scores = regressor.evaluate(x_train, y_train, verbose=0)
#       print(scores[1])
#       print(regressor.summary())
#       print(symbol)
#       print(predicted_price)
#   plt.figure(figsize=(25, 25), dpi=80, facecolor = 'w', edgecolor = 'k')
#
#   plt.plot(test_set[:, 0], color='red', label='Real BTC Price')
#   plt.plot(predicted_price[:, 0], color = 'blue', label = 'Predicted BTC Price')
#
#   plt.title('BTC Price Prediction', fontsize = 40)
#   plt.xlabel('Time', fontsize=40)
#   plt.ylabel('BTC Price(USD)', fontsize = 40)
#   plt.legend(loc = 'best')
#   plt.show()

        return scores[1], predicted_price, price, dfi



