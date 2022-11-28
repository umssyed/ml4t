""""""  		  	   		  	  		  		  		    	 		 		   		 		  
"""  		  	   		  	  		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  	  		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  	  		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  	  		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  	  		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  		  		  		    	 		 		   		 		  
or edited.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  	  		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  	  		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		  	  		  		  		    	 		 		   		 		  
"""  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		  	  		  		  		    	 		 		   		 		  
import random  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		  	  		  		  		    	 		 		   		 		  
import util as ut
import RTLearner as rt
import BagLearner as bl
import numpy as np


from indicators import simple_moving_average, bollinger_bands_percent, momentum
from marketsimcode import compute_portvals
  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
class StrategyLearner(object):  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		  	  		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		  	  		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		  	  		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    # constructor  		  	   		 impact=0.0, commission=0.0
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Constructor method  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		  	  		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   		  	  		  		  		    	 		 		   		 		  
        self.commission = commission
        self.N = 5
        self.YBUY = 0.01
        self.YSELL = -0.01

        self.numBags = 100
        self.leaf_size = 15
        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size":self.leaf_size, "verbose":False}, bags=self.numBags, verbose=False)
        #self.learner = rt.RTLearner(leaf_size=self.leaf_size, verbose=False)

    def author(self):
        return 'msyed46'

    def indicator_data(self, symbol, data, sd, ed):
        prices = pd.DataFrame(columns=[symbol, 'SMA', 'BBP', 'MMT'])
        prices[symbol] = data[symbol]
        prices[symbol] = prices[symbol]/prices[symbol][0]

        sma = simple_moving_average(sd, ed, n=20, symbol=symbol)
        prices['SMA'] = sma
        bbp = bollinger_bands_percent(sd, ed, n=20, symbol=symbol)
        prices['BBP'] = bbp
        mmt = momentum(sd, ed, n=10, symbol=symbol)
        prices['MMT'] = mmt

        #df_prices = prices.dropna()
        df_prices = prices
        return df_prices

    # this method should create a QLearner, and train it for trading
    def add_evidence(  		  	   		  	  		  		  		    	 		 		   		 		  
        self,  		  	   		  	  		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		  	  		  		  		    	 		 		   		 		  
        sd=dt.datetime(2008, 1, 1),  		  	   		  	  		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 1, 1),  		  	   		  	  		  		  		    	 		 		   		 		  
        sv=10000,
    ):  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		
  		  	   		  	  		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		  	  		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		  	  		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		  	  		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		  	  		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		  	  		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
        # add your code to do learning here  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
        # example usage of the old backward compatible util function  		  	   		  	  		  		  		    	 		 		   		 		  
        syms = [symbol]  		  	   		  	  		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed)  		  	   		  	  		  		  		    	 		 		   		 		  
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		  	   		  	  		  		  		    	 		 		   		 		  
        prices = prices_all[syms]  # only portfolio symbols  		  	   		  	  		  		  		    	 		 		   		 		  
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later

        # Get indicator data
        x_features = self.indicator_data(symbol, prices, sd, ed)

        # Append returns and Y
        dataset = x_features.assign(ret=0.00)
        dataset = dataset.assign(Y=0)

        # Calculate Y data
        for t in range(0, len(dataset) - self.N):
            trade_date = dataset.index[t]
            trade_date_N = dataset.index[t + self.N]

            norm_commission = self.commission / sv
            #print(f"norm_commission: {norm_commission}")

            # Attempt to incorporate adjusted returns with commisision
            unadjusted_returns = ((dataset.loc[trade_date_N][symbol]) / dataset.loc[trade_date][symbol]) - 1
            dollar_increased = sv*dataset.loc[trade_date_N][symbol]*unadjusted_returns
            adjusted_returns = dollar_increased - self.commission
            returns = adjusted_returns / (sv * dataset.loc[trade_date_N][symbol])

            # Buying or Selling a stock will push the price up or down. Include self.impact
            returns = ((dataset.loc[trade_date_N][symbol] * (1 + self.impact)) / dataset.loc[trade_date][symbol]) - 1

            '''
            if returns > 0:
                dataset.at[trade_date, 'ret'] = returns - self.impact
            else:
                dataset.at[trade_date, 'ret'] = returns + self.impact
            '''

            dataset.at[trade_date, 'ret'] = returns + self.impact

        for i in range(0, len(dataset)):
            trade_date = dataset.index[i]
            ret = dataset.loc[trade_date]['ret']

            if ret > self.YBUY:
                dataset.at[trade_date, 'Y'] = 1
            elif ret < self.YSELL:
                dataset.at[trade_date, 'Y'] = -1
            else:
                dataset.at[trade_date, 'Y'] = 0

        #print(dataset)
        dataset = dataset.dropna()
        x = dataset[[symbol, 'SMA', 'BBP', 'MMT']]
        y = dataset['Y']

        XTRAIN = np.array(x)
        YTRAIN = np.array(y)

        self.learner.add_evidence(XTRAIN,  YTRAIN)

        if self.verbose:
            #print(prices)
            pass

  		  	   		  	  		  		  		    	 		 		   		 		  

  		  	   		  	  		  		  		    	 		 		   		 		  
    # this method should use the existing policy and test it against new data  		  	   		  	  		  		  		    	 		 		   		 		  
    def testPolicy(  		  	   		  	  		  		  		    	 		 		   		 		  
        self,  		  	   		  	  		  		  		    	 		 		   		 		  
        symbol="IBM",  		  	   		  	  		  		  		    	 		 		   		 		  
        sd=dt.datetime(2009, 1, 1),  		  	   		  	  		  		  		    	 		 		   		 		  
        ed=dt.datetime(2010, 1, 1),  		  	   		  	  		  		  		    	 		 		   		 		  
        sv=100000,
    ):  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		  	  		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		  	  		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		  	  		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		  	  		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		  	  		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  	  		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  	  		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  	  		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  

        # here we build a fake set of trades
        # your code should return the same sort of data  		  	   		  	  		  		  		    	 		 		   		 		  
        dates = pd.date_range(sd, ed)  		  	   		  	  		  		  		    	 		 		   		 		  
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY  		  	   		  	  		  		  		    	 		 		   		 		  
        trades = prices_all[[symbol,]]  # only portfolio symbols
        trades_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		  	  		  		  		    	 		 		   		 		  
        trades.values[:, :] = 0  # set them all to nothing  		  	   		  	  		  		  		    	 		 		   		 		  

        # Get indicator data
        x_features = self.indicator_data(symbol, prices_all, sd, ed)

        # Append returns and Y
        dataset = x_features.assign(ret=0.00)
        dataset = dataset.assign(Y=0)

        # Find XTEST
        x = dataset[[symbol, 'SMA', 'BBP', 'MMT']]
        XTEST = np.array(x)

        YTEST = self.learner.query(XTEST)
        curr_shares = 0
        for i in range(1, len(trades)):
            trade_date = trades.index[i]

            if YTEST[i] > 0:
                # THIS IS A BUY CONDITION
                # If curr_shares = 0, buy 1000
                # If curr_shares = -1000, buy 2000
                # If curr_shares = 1000, buy 0 -> cant have more than 1000
                if curr_shares == 0:
                    trades.at[trade_date, symbol] = 1000
                    curr_shares = 1000
                elif curr_shares == -1000:
                    trades.at[trade_date, symbol] = 2000
                    curr_shares = 1000
                else:
                    trades.at[trade_date, symbol] = 0
                    curr_shares = 1000

            elif YTEST[i] < 0:
                # THIS IS A SELL CONDITION
                # If curr_shares = 0, sell 1000
                # If curr_shares = 1000, sell 2000
                # If curr_shares = -1000, sell 0 -> cant have more than -1000
                if curr_shares == 0:
                    trades.at[trade_date, symbol] = -1000
                    curr_shares = -1000
                elif curr_shares == 1000:
                    trades.at[trade_date, symbol] = -2000
                    curr_shares = -1000
                else:
                    trades.at[trade_date, symbol] = 0


            else:
                # HOLD NO POSITION IN THE STOCK
                trades.at[trade_date, symbol] = 0
                curr_shares = 0

        if self.verbose:
            pass
            #print(type(trades))  # it better be a DataFrame!
        if self.verbose:
            pass
            #print(trades)
        if self.verbose:
            pass
            #print(prices_all)
        return trades  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    def gen_trades(self, trades):
        #print("Generating trades...")
        trades.columns = ['Shares']
        trades = trades.assign(Symbol='JPM')
        trades = trades.assign(Order='HOLD')

        numBuys = 0
        numSells = 0

        curr_shares = 0
        for i in range(0, len(trades)):
            trade_date = trades.index[i]
            trade_shares = trades.loc[trade_date]['Shares']

            if trade_shares > 0:
                trades.at[trade_date, 'Order'] = 'BUY'
                numBuys += 1
            elif trade_shares < 0:
                trades.at[trade_date, 'Order'] = 'SELL'
                numSells += 1
            else:
                trades.at[trade_date, 'Order'] = 'HOLD'
        print(f"Number of Buys: {numBuys}")
        print(f"Number of Sells: {numSells}")
        return trades

if __name__ == "__main__":
    sl = StrategyLearner(verbose=True, impact=0.0125, commission=9.95)
    sl.add_evidence(symbol='JPM')
    df_trades_insample = sl.testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1))
    df_trades_outsample = sl.testPolicy(symbol='JPM', sd=dt.datetime(2009, 1, 1), ed=dt.datetime(2010, 1, 1))


    trades = sl.gen_trades(df_trades_insample)
    pv = compute_portvals(trades, start_val=100000, impact=0.0125, commission=9.95)
    print('\nFinal pv:\n')
    print(pv)
