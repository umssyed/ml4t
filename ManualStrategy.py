import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data
import matplotlib.pyplot as plt

from indicators import simple_moving_average, bollinger_bands_percent, momentum, stochastic_osc, commodity_channel_index
from marketsimcode import compute_portvals

class ManualStrategy(object):

    #constructor
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        """Constructor method"""
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def author():
        return 'msyed46'

    def assort_data(self, symbol, data, sd, ed):
        """
        The method takes normalized prices of the symbol, the SMA, Momentum and BBP
        All values with NaN are dropped
        """
        prices = pd.DataFrame(columns=[symbol, 'SMA', 'BBP', 'MMT'])
        prices[symbol] = data[symbol]
        prices[symbol] = data[symbol]/data[symbol][0]


        sma = simple_moving_average(sd, ed, n=20, symbol=symbol)
        prices['SMA'] = sma
        bbp = bollinger_bands_percent(sd, ed, n=20, symbol=symbol)
        prices['BBP'] = bbp
        mmt = momentum(sd, ed, n=10, symbol=symbol)
        prices['MMT'] = mmt

        df_prices = prices.dropna()
        df_prices = prices
        return df_prices

    def run_benchmark(self, symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        # Benchmark: Buy and hold 1000 shares at the start
        data = get_data([symbol], dates=pd.date_range(sd, ed), addSPY=False).dropna()
        df_assorted_data = self.assort_data(symbol=symbol, data=data, sd=sd, ed=ed)
        trades = df_assorted_data.copy(deep=True)
        trades.drop([symbol, 'SMA', 'BBP', 'MMT'], axis=1, inplace=True)
        df_trades = trades.assign(Symbol=symbol, Order='HOLD', Shares=0)
        first_trading_day = df_trades.index[0]
        df_trades.at[first_trading_day, 'Order'] = 'BUY'
        df_trades.at[first_trading_day, 'Shares'] = 1000
        df_trades.at[df_trades.index[1]:, 'Order'] = 'HOLD'
        df_trades.at[df_trades.index[1]:, 'Shares'] = 0
        return df_trades

    def testPolicy(self, symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        # Get data
        dates = pd.date_range(sd, ed)
        data = get_data([symbol], dates, addSPY=False).dropna()

        # ASSORT PRICES, SMA, BBP, MOMENTUM
        df_assorted_data = self.assort_data(symbol=symbol, data=data, sd=sd, ed=ed)
        #print(df_assorted_data)


        # CREATE TRADES FILE FOR TRACKING ORDERS AND SHARES
        trades = df_assorted_data.copy(deep=True)

        trades.drop([symbol, 'SMA', 'BBP', 'MMT'], axis=1, inplace=True)

        df_trades = trades.assign(Symbol=symbol, Order='HOLD', Shares=0)
        #print(df_trades)

        current_shares = 0
        # GENERATE ORDERS
        for i in range(0, len(df_assorted_data)):
            today = df_assorted_data.index[i]
            if i != 0:
                yesterday = df_assorted_data.index[i - 1]
            else:
                yesterday = today

            price = df_assorted_data.loc[today][symbol]
            sma = df_assorted_data.loc[today]
            bbp = df_assorted_data.loc[today]['BBP']
            mmt_today = df_assorted_data.loc[today]['MMT']
            mmt_yesterday = df_assorted_data.loc[yesterday]['MMT']
            mmt_diff = mmt_today - mmt_yesterday
            perc_mmt_change = mmt_diff/mmt_yesterday
            positive_mmt = mmt_today > 0 and perc_mmt_change > 0.5
            negative_mmt = mmt_today < 0 and perc_mmt_change < -0.5

            #print(positive_mmt)

            if bbp <= 0.2 and not self.upward_trend(symbol, today, df_assorted_data):
                current_shares = self._go_long(df_trades, today, current_shares)

            elif bbp >= 0.8 and self.upward_trend(symbol, today, df_assorted_data):
                current_shares = self._go_short(df_trades, today, current_shares)

            else:
                # UPWARD TREND
                # If we are in upward trend with heavy positive momentum, we want to go long
                # If we are in upward trend with slight positive momentum, and if we are currently
                # in short position, we want to buy to zero. If we are at 0, we want to buy to 1000, but not
                # 2000 shares to minimize risk.
                if self.upward_trend(symbol, today, df_assorted_data):
                    if positive_mmt:
                        current_shares = self._go_long(df_trades, today, current_shares)
                    else:
                        # If we have 1000 Shares long, we dont want to sell it yet because there is not enough
                        # indication that price will go down. But if we are 1000 Shares short, we may want to
                        # buy 1000 shares to be net 0 to minimize our losses
                        if mmt_today > 0 and current_shares == -1000 or current_shares == 0:
                            current_shares = self._buy_to_zero(df_trades, today, current_shares)

                # DOWNWARD TREND
                # If we are in downward trend with heavy negative moment, we want to go short
                # If we are in downward trend with slight negative momentum, and if we are current
                # in long position, we want to sell to zero.
                elif not self.upward_trend(symbol, today, df_assorted_data):
                    if negative_mmt:
                        current_shares = self._go_short(df_trades, today, current_shares)
                    else:
                        # If we have 1000 Shares short, we dont want to buy shares yet because there is not enough
                        # indication that price will go up. But if we are 1000 Shares long, we may want to
                        # sell 1000 shares to be net 0 to minimize our losses
                        if mmt_today < 0 and current_shares == 1000 or current_shares == 0:
                            current_shares = self._sell_to_zero(df_trades, today, current_shares)

        return df_trades

    # STRATEGIES
    def upward_trend(self, symbol, today, df_prices):
        # Price trend. Upward = True, Downward = False
        sma = df_prices.loc[today]['SMA']
        price = df_prices.loc[today][symbol]

        if price > sma:
            return True
        else:
            return False

    # EXECUTIONS
    def _buy(self, df_trades, trade_date, shares):
        df_trades.at[trade_date, 'Order'] = 'BUY'
        df_trades.at[trade_date, 'Shares'] = shares

    def _sell(self, df_trades, trade_date, shares):
        df_trades.at[trade_date, 'Order'] = 'SELL'
        df_trades.at[trade_date, 'Shares'] = shares

    def _hold(self, df_trades, trade_date, shares):
        df_trades.at[trade_date, 'Order'] = 'HOLD'
        df_trades.at[trade_date, 'Shares'] = shares

    # ACTIONS
    def _go_long(self, df_trades, trade_date, current_shares):
        if current_shares == 0:
            trade_shares = 1000
            self._buy(df_trades, trade_date, trade_shares)
            current_shares = current_shares + trade_shares

        elif current_shares == -1000:
            trade_shares = 2000
            self._buy(df_trades, trade_date, trade_shares)
            current_shares = current_shares + trade_shares

        elif current_shares == 1000:
            trade_shares = 0
            self._hold(df_trades, trade_date, trade_shares)
            current_shares = current_shares + trade_shares

        return current_shares

    def _go_short(self, df_trades, trade_date, current_shares):
        if current_shares == 0:
            trade_shares = 1000
            self._sell(df_trades, trade_date, trade_shares)
            current_shares = current_shares - trade_shares

        elif current_shares == 1000:
            trade_shares = 2000
            self._sell(df_trades, trade_date, trade_shares)
            current_shares = current_shares - trade_shares

        elif current_shares == -1000:
            trade_shares = 0
            self._hold(df_trades, trade_date, trade_shares)
            current_shares = current_shares - trade_shares

        return current_shares

    def _buy_to_zero(self, df_trades, trade_date, current_shares):
        if current_shares == -1000:
            trade_shares = 1000
            self._buy(df_trades, trade_date, trade_shares)
            current_shares = current_shares + trade_shares
        elif current_shares == 0:
            trade_shares = 0
            self._hold(df_trades, trade_date, trade_shares)
            current_shares = current_shares + trade_shares
        return current_shares


    def _sell_to_zero(self, df_trades, trade_date, current_shares):
        if current_shares == 1000:
            trade_shares = 1000
            self._sell(df_trades, trade_date, trade_shares)
            current_shares = current_shares - trade_shares

        else:
            trade_shares = 0
            self._hold(df_trades, trade_date, trade_shares)
            current_shares = current_shares - trade_shares

        return current_shares



def plot(start_date, end_date, benchmark, manual_strategy, benchmark_trades, manual_strategy_trades, data_sample):
    bm_trades = benchmark_trades.loc[benchmark_trades['Order'] == 'BUY']['Order']
    ms_trades = manual_strategy_trades.loc[(manual_strategy_trades['Order'] == 'BUY') | (manual_strategy_trades['Order'] == 'SELL')]['Order']

    plt.figure(figsize=(15, 9))
    plt.plot(benchmark, label="Benchmark", color="purple")
    plt.plot(manual_strategy, label="Manual Strategy", color="red")

    # Check ms_trades:
    for i in range(0, len(ms_trades)):
        date = ms_trades.index[i]
        # LONG - BLUE, SHORT - BLACK
        if ms_trades.loc[date]== 'BUY':
            plt.axvline(x=date, color='blue')
        elif ms_trades.loc[date] == 'SELL':
            plt.axvline(x=date, color='black')

    if data_sample == 'in_sample':
        plt.title("In-Sample Manual versus Benchmark Strategies")
        plt.xlabel('Date')
        plt.ylabel('Normalized Price ($)')
        plt.legend()
        plt.grid()
        plt.savefig('images/insample_manual_benchmark.png', facecolor='wheat')
    else:
        plt.title("Out-Sample Manual versus Benchmark Strategies")
        plt.xlabel('Date')
        plt.ylabel('Normalized Price ($)')
        plt.legend()
        plt.grid()
        plt.savefig('images/outsample_manual_benchmark.png', facecolor='wheat')
    plt.clf()

def generate_statistics(benchmark, manual_strategy):
    pd.set_option('display.float_format', '{:.6f}'.format)

    # Final Portfolio Value
    pv_benchmark = benchmark[-1]
    pv_manual = manual_strategy[-1]

    # Calculate Cummulative Return
    cr_benchmark = (benchmark[-1] / benchmark[0]) - 1
    cr_manual = (manual_strategy[-1] / manual_strategy[0]) - 1

    # Calculate Daily Returns
    dr_benchmark = ((benchmark / benchmark.shift(1)) - 1).iloc[1:]
    dr_manual = ((manual_strategy / manual_strategy.shift(1)) - 1).iloc[1:]

    # Calculate StDev of Daily Returns
    sd_benchmark = dr_benchmark.std()
    sd_manual = dr_manual.std()

    # Calculate Mean of Daily Returns
    mean_benchmark = dr_benchmark.mean()
    mean_manual = dr_manual.mean()

    stats = {
        'Statistics' : ['Final Portfolio Value', 'Cumulative Return', 'Standard Deviation of Daily Returns', 'Mean of Daily Returns'],
        'Benchmark Strategy': [pv_benchmark, cr_benchmark, sd_benchmark, mean_benchmark],
        'Manual Strategy': [pv_manual, cr_manual, sd_manual, mean_manual]
    }

    return stats

def author():
    return 'msyed46'

def run():
    # ================================== IN SAMPLE
    sym = 'JPM'
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    start_val = 100000

    ms = ManualStrategy()
    manual_strategy_trades = ms.testPolicy(symbol=sym, sd=start_date, ed=end_date, sv=start_val)
    benchmark_trades = ms.run_benchmark(symbol=sym, sd=start_date, ed=end_date, sv=start_val)


    # Compute Portfolio Value - manual_strategy and benchmark
    manual_strategy_portval = compute_portvals(manual_strategy_trades, start_val, commission=9.95, impact=0.005)
    manual_strategy_portval_norm = manual_strategy_portval['Values']/manual_strategy_portval['Values'][0]

    benchmark_portval = compute_portvals(benchmark_trades, start_val, commission=9.95, impact=0.005)
    benchmark_portval_norm = benchmark_portval['Values']/benchmark_portval['Values'][0]

    # Calculate Statistics
    insample_stats = generate_statistics(benchmark_portval['Values'], manual_strategy_portval['Values'])
    insample_table = pd.DataFrame(insample_stats)
    insample_table = insample_table.set_index('Statistics')
    print("In Sample Statistics")
    print(insample_table)
    print()

    # In-Sample Plots
    plot(start_date, end_date, benchmark_portval_norm, manual_strategy_portval_norm, benchmark_trades, manual_strategy_trades, "in_sample")


    # =============================================
    # ================================== OUT SAMPLE
    sym = 'JPM'
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2011, 12, 31)
    start_val = 100000

    ms = ManualStrategy()
    manual_strategy_trades = ms.testPolicy(symbol=sym, sd=start_date, ed=end_date, sv=start_val)
    benchmark_trades = ms.run_benchmark(symbol=sym, sd=start_date, ed=end_date, sv=start_val)

    # Compute Portfolio Value - manual_strategy and benchmark
    manual_strategy_portval = compute_portvals(manual_strategy_trades, start_val, commission=9.95, impact=0.005)
    manual_strategy_portval_norm = manual_strategy_portval['Values'] / manual_strategy_portval['Values'][0]

    benchmark_portval = compute_portvals(benchmark_trades, start_val, commission=9.95, impact=0.005)
    benchmark_portval_norm = benchmark_portval['Values'] / benchmark_portval['Values'][0]

    # Calculate Statistics
    outsample_stats = generate_statistics(benchmark_portval['Values'], manual_strategy_portval['Values'])
    outsample_stats = pd.DataFrame(outsample_stats)
    outsample_stats = outsample_stats.set_index('Statistics')
    print("Out of Sample Statistics")
    print(outsample_stats)
    print()

    # Out-Sample Plots
    plot(start_date, end_date, benchmark_portval_norm, manual_strategy_portval_norm, benchmark_trades, manual_strategy_trades, "out_sample")


if __name__ == '__main__':
    run()