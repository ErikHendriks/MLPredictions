#!/usr/bin/env python3

#Description: This program predicts the price of Bitcoin for the next 30 days

#Data Source: https://www.blockchain.com/charts/market-price?
# https://towardsdatascience.com/bitcoin-price-prediction-using-lstm-9eb0938c22bd

import numpy as np 
import pandas as pd

#Load the data
#from google.colab import files # Use to load data on Google Colab
#uploaded = files.upload() # Use to load data on Google Colab

import datetime
import json
import numpy as np
import numpy as np
import logging
import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders as orders
import pandas as pd
import requests
import time
import urllib3

from indicators import indicator
from MLPredictions import MLAlgorithms
from oandapyV20 import API
from oandapyV20.contrib.requests import *
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.pricing import PricingStream
from oandapyV20.exceptions import V20Error, StreamTerminated
from pandas.io.json import json_normalize
from requests.exceptions import ConnectionError
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
#from sendInfo import sendEmail

logging.basicConfig(
    filename='/var/log/naturalResourcesPrediction.log',
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s : %(message)s',)

conf = [line.strip('\n') for line in open('/etc/oandaV20/naturalResourcesPrediction.conf')]
#conf = [line.strip('\n') for line in open('/etc/breakout/conf.v20')]
accountID = conf[0]
print(accountID)
api = API(access_token = conf[1],\
          environment = conf[2])

textList = []
textList.append('Oanda v20 test rapport at '+str(datetime.datetime.now()))
textList.append(' ')
#symbols = ['AUD_NZD']
#symbols = ['AUD_NZD','AUD_USD',\
#          'EUR_AUD','EUR_GBP','EUR_USD',\
#          'GBP_USD',\
#          'NZD_USD',\
#          'USD_CAD','USD_CHF']
#symbols = ['AUD_CAD','AUD_CHF','AUD_JPY','AUD_NZD','AUD_USD',\
#          'CAD_CHF','CAD_JPY',\
#          'CHF_JPY',\
#          'EUR_AUD','EUR_CAD','EUR_CHF','EUR_GBP','EUR_JPY','EUR_NZD','EUR_USD',\
#          'GBP_AUD','GBP_CAD','GBP_CHF','GBP_JPY','GBP_NZD','GBP_USD',\
#          'NZD_CAD','NZD_CHF','NZD_JPY','NZD_USD',\
#          'USD_CAD','USD_CHF','USD_JPY']

symbols = ['BCO_USD', 'CORN_USD', 'NATGAS_USD', 'SOYBN_USD', 'SUGAR_USD', 'WHEAT_USD', 'WTICO_USD']
params = {'instruments':'BCO_USD ,CORN_USD ,NATGAS_USD ,SOYBN_USD ,SUGAR_USD ,WHEAT_USD ,WTICO_USD'}

price = PricingStream(accountID=accountID,params=params)

ohlcd = {'count': 365,'granularity': 'D'}

for symbol in symbols:
#   r = MLAlgorithms.lstm(symbol, ohlcd, api, 1)
    r = MLAlgorithms.radialBasis(symbol, ohlcd, api, 1)
    accuracy = r[0]
    prediction = r[1][0]
    price = r[2]
    dfi = r[3]
    print(r[0])
    if r[0] > 0.9:
        if r[1][0] > r[2]:

            atr = indicator.atr(r[3],[14])
            ma30 = indicator.movingAverage(r[3],[30])
            ma50 = indicator.movingAverage(r[3],[50])
            ma100 = indicator.movingAverage(r[3],[100])
            buyRule1 = [[i for i in ma30.iloc[range(-5,-1)]] > [i for i in ma50.iloc[range(-5,-1)]]]
            buyRule2 = [[i for i in ma50.iloc[range(-5,-1)]] > [i for i in ma100.iloc[range(-5,-1)]]]
            if buyRule1[0] is True\
            and buyRule2[0] is True:
                print(symbol)
                candl = r[2]
                atr = float(atr)
                if 'CN50_USD' in symbol\
                or 'IN50_USD' in symbol\
                or 'AU200_AUD' in symbol\
                or 'EU50_EUR' in symbol\
                or 'FR40_EUR' in symbol\
                or 'DE30_EUR' in symbol\
                or 'XAG_JPY' in symbol\
                or 'HK33_HKD' in symbol\
                or 'UK100_GBP' in symbol\
                or 'TWIX_USD' in symbol\
                or 'JP225_USD' in symbol:
                    stopLoss = round(candl - (atr/2),1)
                    takeProfit = round(candl + (atr/2),1)

                elif 'NL25_EUR' in symbol\
                or 'SG30_SGD' in symbol:
                    stopLoss = round(candl - (atr/2),2)
                    takeProfit = round(candl + (atr/2),2)

                elif 'JPY' in symbol\
                or 'BCO_USD' in symbol\
                or 'UK10YB_GBP' in symbol\
                or 'NATGAS_USD' in symbol\
                or 'SOYBN_USD' in symbol\
                or 'CORN_USD' in symbol\
                or 'USB10Y_USD' in symbol\
                or 'USB05Y_USD' in symbol\
                or 'USB02Y_USD' in symbol\
                or 'WHEAT_USD' in symbol\
                or 'WTICO_USD' in symbol\
                or 'XPD_USD' in symbol\
                or 'XPT_USD' in symbol\
                or 'XAU' in symbol\
                or 'HUF' in symbol:
                    stopLoss = round(candl - (atr/2),3)
                    takeProfit = round(candl + (atr/2),3)

                else:
                    stopLoss = round(candl - (atr/2),5)
                    takeProfit = round(candl + (atr/2),5)


                buyOrder = MarketOrderRequest(instrument=symbol,\
                              units=25,\
                              takeProfitOnFill=TakeProfitDetails(price=float(takeProfit)).data,\
                              stopLossOnFill=StopLossDetails(price=float(stopLoss)).data)
                ro = orders.OrderCreate(accountID, data=buyOrder.data)
                rv = api.request(ro)
                print(rv)

        if r[1][0] < r[2]:
            atr = indicator.atr(r[3],[14])
            ma30 = indicator.movingAverage(r[3],[30])
            ma50 = indicator.movingAverage(r[3],[50])
            ma100 = indicator.movingAverage(r[3],[100])
            sellRule1 = [[i for i in ma30.iloc[range(-5,-1)]] < [i for i in ma50.iloc[range(-5,-1)]]]
            sellRule2 = [[i for i in ma50.iloc[range(-5,-1)]] < [i for i in ma100.iloc[range(-5,-1)]]]
            if sellRule1[0] is True\
            and sellRule2[0] is True:
                print(symbol)
                candl = r[2]
                atr = float(atr)
                if 'XAU_JPY' in symbol:
                    stopLoss = round(candl + (atr/2),0)
                    takeProfit = round(candl - (atr/2),0)

                if 'CN50_USD' in symbol\
                or 'IN50_USD' in symbol\
                or 'AU200_AUD' in symbol\
                or 'EU50_EUR' in symbol\
                or 'FR40_EUR' in symbol\
                or 'DE30_EUR' in symbol\
                or 'XAG_JPY' in symbol\
                or 'HK33_HKD' in symbol\
                or 'UK100_GBP' in symbol\
                or 'TWIX_USD' in symbol\
                or 'JP225_USD' in symbol:
                    stopLoss = round(candl + (atr/2),1)
                    takeProfit = round(candl - (atr/2),1)

                elif 'NL25_EUR' in symbol\
                or 'SG30_SGD' in symbol:
                    stopLoss = round(candl + (atr/2),2)
                    takeProfit = round(candl - (atr/2),2)

                elif 'JPY' in symbol\
                or 'BCO_USD' in symbol\
                or 'UK10YB_GBP' in symbol\
                or 'NATGAS_USD' in symbol\
                or 'SOYBN_USD' in symbol\
                or 'CORN_USD' in symbol\
                or 'USB10Y_USD' in symbol\
                or 'USB05Y_USD' in symbol\
                or 'USB02Y_USD' in symbol\
                or 'WHEAT_USD' in symbol\
                or 'WTICO_USD' in symbol\
                or 'XPD_USD' in symbol\
                or 'XPT_USD' in symbol\
                or 'XAU' in symbol\
                or 'HUF' in symbol:
                    stopLoss = round(candl + (atr/2),3)
                    takeProfit = round(candl - (atr/2),3)

                else:
                    stopLoss = round(candl + (atr/2),5)
                    takeProfit = round(candl - (atr/2),5)


                sellOrder = MarketOrderRequest(instrument=symbol,\
                              units=-25,\
                              takeProfitOnFill=TakeProfitDetails(price=float(takeProfit)).data,\
                              stopLossOnFill=StopLossDetails(price=float(stopLoss)).data)
                ro = orders.OrderCreate(accountID, data=sellOrder.data)
                rv = api.request(ro)
                print(rv)


#   candle = InstrumentsCandles(instrument=symbol,params=ohlcd)
#   api.request(candle)
#
#   df = pd.DataFrame.from_dict(json_normalize(candle.response['candles']))
#
#   for column in 'mid.c','mid.h','mid.l','mid.o':
#       df[column] = df[column].astype(float)
#
#   dfi = df.copy()
##  print(df)
#
#   #A variable for predicting 'n' days out into the future
#   prediction_days = 1 #n = 30 days
#
#   #Create another column (the target or dependent variable) shifted 'n' units up
#   df['Prediction'] = df[['mid.c']].shift(-prediction_days)
#
#   #CREATE THE INDEPENDENT DATA SET (X)
#   df.drop(columns=['complete','volume','time','mid.o','mid.h','mid.l'], inplace=True)
#
#   # Convert the dataframe to a numpy array and drop the prediction column
#   X = np.array(df.drop(['Prediction'],1))
#
#   #Remove the last 'n' rows where 'n' is the prediction_days
#   X= X[:len(df)-prediction_days]
#   #print('x ',X)
#
#   #CREATE THE DEPENDENT DATA SET (y)
#
#   # Convert the dataframe to a numpy array (All of the values including the NaN's)
#   y = np.array(df['Prediction'])
#
#   # Get all of the y values except the last 'n' rows
#   y = y[:-prediction_days]
#   #print('y ',y)
#
#   # Split the data into 80% training and 20% testing
##  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#   x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.16)
#
#   # Set prediction_days_array equal to the last 30 rows of the original data set from the price column
#   prediction_days_array = np.array(df.drop(['Prediction'],1))[-prediction_days:]
#   #print(prediction_days_array)
#
#   # Create and train the Support Vector Machine (Regression) using the radial basis function
##  svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001)
#   svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001)
#   svr_rbf.fit(x_train, y_train)
#
#   # Testing Model: Score returns the accuracy of the prediction.
#   # The best possible score is 1.0
#   svr_rbf_confidence = svr_rbf.score(x_test, y_test)
##  print("svr_rbf accuracy: ", svr_rbf_confidence)
#
#   # Print the predicted value
#   svm_prediction = svr_rbf.predict(x_test)
##  print(svm_prediction)
#
#
#   #Print the actual values
#   #print(y_test)
#
#   # Print the model predictions for the next 'n' days
#   svm_prediction = svr_rbf.predict(prediction_days_array)
##  print(svm_prediction)
#
#   #Print the actual price for the next 'n' days, n=prediction_days=30
##  print(df)


