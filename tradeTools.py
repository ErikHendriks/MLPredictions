#!/usr/bin/env python3

import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades

from oandapyV20.contrib.requests import *
from oandapyV20 import API

conf = [line.strip('\n') for line in open('/etc/oandaV20/metalsPrediction.conf')]
accountID = conf[0]
api = API(access_token = conf[1],\
          environment = conf[2])

class TradeTools(object):

    def closeTrades(accountID, api):
        ot = trades.OpenTrades(accountID=accountID)
        api.request(ot)
        for i in range(0,len(ot.response['trades'])-1):
            tc = trades.TradeClose(accountID=accountID, tradeID=ot.response['trades'][i]['id'])
            api.request(tc)
#           print(api.request(tc))

    def marketOrder(accountID, api, symbol, units, takeProfit, stopLoss):
        marketOrder = MarketOrderRequest(instrument=symbol,\
                  units=1,\
                  takeProfitOnFill=TakeProfitDetails(price=float(takeProfit)).data,\
                  stopLossOnFill=StopLossDetails(price=float(stopLoss)).data)
        bo = orders.OrderCreate(accountID, data=marketOrder.data)
        api.request(bo)
#       rv = api.request(bo)
#       print(rv)


