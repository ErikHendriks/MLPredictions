#!/usr/bin/env python3

import datetime
import logging

from indicators import indicator
from MLPredictions import MLAlgorithms
from oandapyV20 import API
from tradeTools import TradeTools
from sendInfo import sendEmail

logging.basicConfig(
    filename='/var/log/metalsPrediction.log',
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s : %(message)s',)

conf = [line.strip('\n') for line in open('/etc/oandaV20/metalsPrediction.conf')]
accountID = conf[0]
api = API(access_token = conf[1], environment = conf[2])

TradeTools.closeTrades(accountID,api)

textList = []
textList.append('Oanda v20 metals rapport at '+str(datetime.datetime.now()))
textList.append(' ')

symbols = ['XAG_AUD', 'XAG_CAD', 'XAG_CHF', 'XAG_EUR', 'XAG_GBP', 'XAG_HKD', 'XAG_JPY', 'XAG_NZD', 'XAG_SGD', 'XAG_USD', 'XAU_AUD', 'XAU_CAD', 'XAU_CHF', 'XAU_EUR', 'XAU_GBP', 'XAU_HKD', 'XAU_JPY', 'XAU_NZD', 'XAU_SGD', 'XAU_USD', 'XAU_XAG', 'XCU_USD', 'XPD_USD', 'XPT_USD']
ohlcd = {'count': 208,'granularity': 'W'}

dec0 = ['XAU_JPY']
dec1 = ['XAG_JPY']
dec3 = ['XAU_AUD','XAU_CAD','XAU_CHF','XAU_EUR','XAU_GBP','XAU_HKD','XAU_NZD','XAU_SGD','XAU_USD','XAU_XAG','XPD_USD','XPT_USD','XAG_HKD','']
dec5 = ['XAG_EUR','XCU_USD','XAG_USD','XAG_SGD','XAG_NZD','XAG_HKD','XAG_GBP','XAG_CHF','XAG_CAD','XAG_AUD']

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

                if symbol in dec0:
                    stopLoss = round(price - (atr/2),0)
                    takeProfit = round(price + (atr/2),0)

                elif symbol in dec1:
                    stopLoss = round(price - (atr/2),1)
                    takeProfit = round(price + (atr/2),1)

                elif symbol in dec3:
                    stopLoss = round(price - (atr/2),3)
                    takeProfit = round(price + (atr/2),3)

                else:
                    stopLoss = round(price - (atr/2),5)
                    takeProfit = round(price + (atr/2),5)

                response = TradeTools.marketOrder(accountID, api, symbol, 1, takeProfit, stopLoss)
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

                if symbol in dec0:
                    stopLoss = round(price + (atr/2),0)
                    takeProfit = round(price - (atr/2),0)

                elif symbol in dec1:
                    stopLoss = round(price + (atr/2),1)
                    takeProfit = round(price - (atr/2),1)

                elif symbol in dec3:
                    stopLoss = round(price + (atr/2),3)
                    takeProfit = round(price - (atr/2),3)

                else:
                    stopLoss = round(price + (atr/2),5)
                    takeProfit = round(price - (atr/2),5)

                response = TradeTools.marketOrder(accountID, api, symbol, -1, takeProfit, stopLoss)
                textList.append('sell')
                textList.append(symbol)
                textList.append(response)
                textList.append(' ')

text = '\n'.join(map(str,textList))
subject = 'Weekly rapport metals at '+str(datetime.datetime.now())
sendEmail(text,subject)


