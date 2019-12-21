#!/usr/bin/env python3

import datetime
import logging

from indicators import indicator
from MLPredictions import MLAlgorithms
from oandapyV20 import API
from tradeTools import TradeTools
from sendInfo import sendEmail

logging.basicConfig(
    filename='/var/log/currenciesPrediction.log',
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s : %(message)s',)

conf = [line.strip('\n') for line in open('/etc/oandaV20/currenciesPrediction.conf')]
accountID = conf[0]
api = API(access_token = conf[1], environment = conf[2])

TradeTools.closeTrades(accountID,api)

textList = []
textList.append('Oanda v20 currencies rapport at '+str(datetime.datetime.now()))
textList.append(' ')

symbols = ['AUD_CAD', 'AUD_CHF', 'AUD_HKD', 'AUD_JPY', 'AUD_NZD', 'AUD_SGD', 'AUD_USD', 'CAD_CHF', 'CAD_HKD', 'CAD_JPY', 'CAD_SGD', 'CHF_HKD', 'CHF_JPY', 'CHF_ZAR', 'EUR_AUD', 'EUR_CAD', 'EUR_CHF', 'EUR_CZK', 'EUR_DKK', 'EUR_GBP', 'EUR_HKD', 'EUR_HUF', 'EUR_JPY', 'EUR_NOK', 'EUR_NZD', 'EUR_PLN', 'EUR_SEK', 'EUR_SGD', 'EUR_TRY', 'EUR_USD', 'EUR_ZAR', 'FR40_EUR', 'GBP_AUD', 'GBP_CAD', 'GBP_CHF', 'GBP_HKD', 'GBP_JPY', 'GBP_NZD', 'GBP_PLN', 'GBP_SGD', 'GBP_USD', 'GBP_ZAR', 'HKD_JPY', 'NZD_CAD', 'NZD_CHF', 'NZD_HKD', 'NZD_JPY', 'NZD_SGD', 'NZD_USD', 'SGD_CHF', 'SGD_HKD', 'SGD_JPY', 'TRY_JPY', 'USD_CAD', 'USD_CHF', 'USD_CNH', 'USD_CZK', 'USD_DKK', 'USD_HKD', 'USD_HUF', 'USD_INR', 'USD_JPY', 'USD_MXN', 'USD_NOK', 'USD_PLN', 'USD_SAR', 'USD_SEK', 'USD_SGD', 'USD_THB', 'USD_TRY', 'USD_ZAR', 'ZAR_JPY']
ohlcd = {'count': 730,'granularity': 'D'}

for symbol in symbols:
#   r = MLAlgorithms.lstm(symbol, ohlcd, api, 1)
    r = MLAlgorithms.radialBasis(symbol, ohlcd, api, 1)
    accuracy = r[0]
    prediction = r[1][0]
    price = r[2]
    dfi = r[3]

    if accuracy > 0.9:
        if prediction > price:
            atr = indicator.atr(dfi,[14])
            ma30 = indicator.movingAverage(dfi,[30])
            ma50 = indicator.movingAverage(dfi,[50])
            ma100 = indicator.movingAverage(dfi,[100])

            buyRule1 = [[i for i in ma30.iloc[range(-5,-1)]] > [i for i in ma50.iloc[range(-5,-1)]]]
            buyRule2 = [[i for i in ma50.iloc[range(-5,-1)]] > [i for i in ma100.iloc[range(-5,-1)]]]
            if buyRule1[0] is True\
            and buyRule2[0] is True:
                atr = float(atr)

                if 'JPY' in symbol\
                or 'INR' in symbol\
                or 'THB' in symbol\
                or 'HUF' in symbol:
                    stopLoss = round(price - (atr/2),3)
                    takeProfit = round(price + (atr/2),3)

                else:
                    stopLoss = round(price - (atr/2),5)
                    takeProfit = round(price + (atr/2),5)

                response = TradeTools.marketOrder(accountID, api, symbol, 1000, takeProfit, stopLoss)
                textList.append('buy')
                textList.append(symbol)
                textList.append(response)
                textList.append(' ')

        if prediction < price:
            atr = indicator.atr(dfi,[14])
            ma30 = indicator.movingAverage(dfi,[30])
            ma50 = indicator.movingAverage(dfi,[50])
            ma100 = indicator.movingAverage(dfi,[100])

            sellRule1 = [[i for i in ma30.iloc[range(-5,-1)]] < [i for i in ma50.iloc[range(-5,-1)]]]
            sellRule2 = [[i for i in ma50.iloc[range(-5,-1)]] < [i for i in ma100.iloc[range(-5,-1)]]]
            if sellRule1[0] is True\
            and sellRule2[0] is True:
                atr = float(atr)

                if 'JPY' in symbol\
                or 'INR' in symbol\
                or 'THB' in symbol\
                or 'HUF' in symbol:
                    stopLoss = round(price + (atr/2),3)
                    takeProfit = round(price - (atr/2),3)

                else:
                    stopLoss = round(price + (atr/2),5)
                    takeProfit = round(price - (atr/2),5)

                response = TradeTools.marketOrder(accountID, api, symbol, -1000, takeProfit, stopLoss)
                textList.append('sell')
                textList.append(symbol)
                textList.append(response)
                textList.append(' ')

text = '\n'.join(map(str,textList))
subject = 'Weekly rapport currencies at '+str(datetime.datetime.now())
sendEmail(text,subject)


