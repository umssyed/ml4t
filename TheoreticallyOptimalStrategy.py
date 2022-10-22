import marketsimcode as ms
import pandas as pd
import datetime as dt
from util import get_data, plot_data
import matplotlib.pyplot as plt

def testPolicy(symbol, sd, ed, sv):
    print(f"TEST POLICY")
    # Generate data using util function and using the specified date range
    df = get_data([symbol], dates=pd.date_range(sd, ed))

    # Only grab the symbol information into a dataframe. This is the price each day for the symbol
    sym_price_df = df[[symbol]]

    # Forward fill and Backward fill dataframe
    sym_price_df = sym_price_df.ffill()
    sym_price_df = sym_price_df.bfill()

    #--------------------------TRADES DATAFRAME--------------------------#

    # Create a trades dataframe to store trade actions
    trades = pd.DataFrame(index=pd.date_range(sd, ed))

    # Add default symbol, order, shares
    trades = trades.assign(Symbol=symbol)
    trades = trades.assign(Curr_Pos=0)
    trades = trades.assign(Execute=0)
    # Perform trades
    len_df = len(sym_price_df)
    date = sym_price_df.index

    # Set current position in symbol to 0
    curr_pos = 0
    execute = 0
    for i in range(len_df-1):
        current_val = sym_price_df.loc[date[i]].loc[symbol]
        next_val = sym_price_df.loc[date[i+1]].loc[symbol]
        trade_date = date[i]

        if next_val > current_val:
            execute = 1000 - curr_pos

        else:
            execute = -1000 - curr_pos

        curr_pos = curr_pos + execute
        trades.loc[trade_date, 'Curr_Pos'] = curr_pos

        trades.loc[trade_date, 'Execute'] = execute

    df_trades = trades[["Symbol", "Execute"]]
    return df_trades

def run_theorectical(symbol, sd, ed, sv):
    # Generate trades
    generate_trades = testPolicy(symbol, sd=sd, ed=ed, sv=sv)
    # Return computed Portfolio Values
    return ms.compute_portvals(orders_file=generate_trades, start_val=sv)

def run_benchmark(symbol, sd, ed, sv):
    # Get dataframe for symbol
    df = get_data([symbol], dates=pd.date_range(sd, ed))
    sym_price_df = df[[symbol]]
    # Perform execution of $100,000 on the first trading day
    execute_date = sym_price_df.index[0]

    # Create trades dataframe to store trade actions
    benchmark_trades = pd.DataFrame(index=pd.date_range(sd, ed))

    # Add default symbol, order, shares
    benchmark_trades = benchmark_trades.assign(Symbol=symbol)
    benchmark_trades = benchmark_trades.assign(Execute=0)
    benchmark_trades.loc[execute_date, 'Execute'] = 1000

    # Return computed Portfolio Values
    return ms.compute_portvals(orders_file=benchmark_trades, start_val=sv)


def gen_results(benchmark, theoretical):
    pd.set_option('display.precision', 6)

    # Calculate Cumulative Return
    cr_benchmark = (benchmark[-1] / benchmark[0]) - 1
    cr_theoretical = (theoretical[-1] / theoretical[0]) - 1

    # Calculate Daily Returns
    dr_benchmark = ((benchmark / benchmark.shift(1)) - 1).iloc[1:]
    dr_theoretical = ((theoretical / theoretical.shift(1)) - 1).iloc[1:]

    # Calculate StDev of Daily Returns
    sd_benchmark = dr_benchmark.std()
    sd_theoretical = dr_theoretical.std()

    # Calculate Mean of Daily Returns
    mean_benchmark = dr_benchmark.mean()
    mean_theoretical = dr_theoretical.mean()


def gen_plots(benchmark, theoretical):

    # Normalize both datasets
    benchmark['Values'] = benchmark['Values'] / benchmark['Values'][0]
    theoretical['Values'] = theoretical['Values'] / theoretical['Values'][0]

    plt.figure(figsize=(15, 9))
    plt.plot(benchmark, label='Benchmark - Buy and Hold', color='Purple')
    plt.plot(theoretical, label='Theoretical Strategy', color='Red')
    plt.title('Theoretically Optimal Strategy against Buy and Hold Strategy')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.grid()
    plt.legend()
    plt.savefig('images/figure1.png')
    plt.clf()

def report():
    benchmark = run_benchmark('JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    theoretical = run_theorectical('JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000)

    # Generate benchmark and theoretical results for daily cumulative, stddev, mean returns of Values
    gen_results(benchmark['Values'], theoretical['Values'])

    # Generate Plots
    gen_plots(benchmark, theoretical)

def author():
    return 'msyed46'

if __name__ == "__main__":
    #of = pd.read_csv('../marketsim/orders/orders-01.csv', index_col=0, parse_dates=[0])
    #p = ms.compute_portvals(orders_file=of, start_val=1000000)
    report()





