#!/usr/bin/env python3

import oandapyV20.endpoints.trades as trades

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

    def marketOrder(symbol,units,stopLoss,takeProfit):


