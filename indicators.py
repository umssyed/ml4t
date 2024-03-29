import pandas as pd
import datetime as dt
from util import get_data, plot_data
import matplotlib.pyplot as plt

def simple_moving_average(sd, ed, n, symbol='JPM', gen_plot=False):
    # SIMPLE MOVING AVERAGE
    df = grab_data(sd, ed, n, symbol)
    # Get the rolling mean based on the above prices
    df['SMA'] = df[symbol].rolling(n).mean()
    # Remove earlier dates to start off with sd instead of sd_n
    df = df.truncate(before=sd)
    # Normalize both symbol and SMA values
    # For SMA, we need the first NaN day
    df[symbol] = df[symbol]/df[symbol][0]
    norm_sma_day = df['SMA'].notna().idxmax()
    df['SMA'] = df['SMA']/df['SMA'][norm_sma_day]

    # Separate out normalized symbol prices and indicators
    df_symbol = df[symbol]

    if gen_plot:
        plt.figure(figsize=(20, 9))
        plt.plot(df_symbol, label=f'{n}-Day SMA', color='Teal')
        plt.plot(df['SMA'], label=f'{symbol} Normalized Price', color='Black')
        plt.title(f'{n}-Day Simple Moving Average')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.grid()
        plt.savefig('images/sma.png', facecolor='wheat')


    df_indicator = df['SMA']
    return df_indicator

def bollinger_bands_percent(sd, ed, n, symbol='JPM', gen_plot=False):
    # BOLLINGER BANDS
    df = grab_data(sd, ed, n, symbol)
    # Get standard deviation and rolling mean (SMA) with n days (typically 20 days for medium term)
    df['stdev'] = df[symbol].rolling(n).std()
    df['SMA'] = df[symbol].rolling(n).mean()

    # Append upper and lower bands
    df['UB'] = df['SMA'] + 2*df['stdev']
    df['LB'] = df['SMA'] - 2 * df['stdev']

    # Remove earlier dates to start off with sd instead of sd_n
    df = df.truncate(before=sd)

    # GENERATE %BOLLINGER BAND
    #% B = (Price - Lower Band) / (Upper Band - Lower Band)
    df['perc_bb'] = (df[symbol] - df['LB']) / (df['UB'] - df['LB'])

    if gen_plot:
        plt.figure(figsize=(20, 9))
        plt.plot(df['perc_bb'], label='Percentage Bollinger Band', color='Teal')
        plt.hlines(0, sd, ed, color='Black')
        plt.hlines(1, sd, ed, color='Black')
        plt.title('Percentage Bollinger Bands')
        plt.xlabel('Date')
        plt.ylabel('%B')
        plt.legend()
        plt.grid()
        plt.savefig('images/bb.png', facecolor='wheat')
        plt.clf()

    df_indicator = df['perc_bb']
    return df_indicator

def momentum(sd, ed, n, symbol='JPM', gen_plot=False):
    # MOMENTUM
    df = grab_data(sd, ed, n, symbol)
    # Append momentum
    df['momentum'] = df[symbol]/df[symbol].shift(n) - 1
    # Remove earlier dates to start off with sd instead of sd_n
    df = df.truncate(before=sd)
    if gen_plot:
        plt.figure(figsize=(20, 9))
        plt.plot(df['momentum'], label=f'Momentum', color='Teal')
        plt.hlines(0, sd, ed, color='Red')
        plt.title(f'{n}-Days Period Momentum')
        plt.xlabel('Date')
        plt.ylabel('Momentum')
        plt.legend()
        plt.grid()
        plt.savefig('images/momentum.png', facecolor='wheat')
        plt.clf()

    df_indicator = df['momentum']
    return df_indicator

def stochastic_osc(sd, ed, n=14, symbol='JPM', gen_plot=False):
    # STOCHASTIC OSCILLATOR - 14 day period of look-back
    df = grab_data(sd, ed, n, symbol)

    df['stoch_osc'] = 100*(df[symbol] - df[symbol].rolling(n).min()) / (df[symbol].rolling(n).max() - df[symbol].rolling(n).min())
    # Remove earlier dates to start off with sd instead of sd_n
    df = df.truncate(before=sd)

    if gen_plot:
        plt.figure(figsize=(20, 9))
        plt.plot(df['stoch_osc'], label='Stochastic Oscillator', color='Teal')
        plt.hlines(20, sd, ed, color='Red', linestyles='dashed')
        plt.hlines(80, sd, ed, color='Red', linestyles='dashed')
        plt.title('Stochastic Oscillator - 14 Day Period')
        plt.xlabel('Date')
        plt.ylabel('Stochastic Oscillator (%K)')
        plt.legend()
        plt.grid()
        plt.savefig('images/stoch_osc.png', facecolor='wheat')
        plt.clf()

    df_indicator = df['stoch_osc']
    return df_indicator

def commodity_channel_index(sd, ed, n=20, symbol='JPM', gen_plot=False):
    # PARABOLIC SAR
    df = grab_data(sd, ed, n, symbol)
    df_other_data = grab_other_data(sd, ed, n, symbol)
    df['High'] = df_other_data['High']
    df['Low'] = df_other_data['Low']
    df['Close'] = df_other_data['Close']


    # Calculate Typical Price: Sum [ (High+Low+Close) / 3 ] i=0 -> i=n
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    #df['TP'] = (df[symbol].max() + df[symbol].min() + df[symbol]) / 3

    # Calculate Rolling SMA - Moving Average (MA)
    df['SMA_TP'] = df['TP'].rolling(n).mean()

    # Calculate Mean Absolute Deviation - |Typical Price - MA|
    df['MAD'] = (df['TP'].rolling(n).std()).rolling(n).mean()
    #df['MAD'] = df['TP'].rolling(n).apply(lambda m: pd.Series(m).mad())

    # Calculate CCI - (TP - MA) / (0.015 x MAD)
    df['CCI'] = (df['TP'] - df['SMA_TP']) / (0.015 * df['MAD'])
    #print(df)

    if gen_plot:
        plt.figure(figsize=(20, 9))
        plt.plot(df['CCI'], label='CCI', color='Teal')
        plt.hlines(100, sd, ed, color='Red', linestyles='dashed')
        plt.hlines(-100, sd, ed, color='Red', linestyles='dashed')
        plt.title('Commodity Channel Index')
        plt.xlabel('Date')
        plt.ylabel('CCI')
        plt.legend()
        plt.grid()
        plt.savefig('images/cci.png', facecolor='wheat')
        plt.clf()

    df_indicator = df['CCI']
    return df_indicator

def grab_data(sd, ed, n, symbol='JPM'):
    # GRAB DATA - PRICE
    # Get data n days before start date to avoid any NaN values
    # Window size based on n. Convert into timeDelta
    window = dt.timedelta(days=n)
    # Get start date for window size. Start date will be n days before actual start date
    # This is to avoid NaN values
    sd_n = sd - window
    # Generate data using util function and using the specified date range
    df_data = get_data([symbol], dates=pd.date_range(sd_n, ed))
    df_data.fillna(method='ffill', inplace=True)
    df_data.fillna(method='bfill', inplace=True)
    # Remove SPY index
    df = df_data[[symbol]]
    return df

def grab_other_data(sd, ed, n, symbol='JPM'):
    # GRAB OTHER DATA - HIGH, LOW, CLOSE
    # Get data n days before start date to avoid any NaN values
    # Window size based on n. Convert into timeDelta
    window = dt.timedelta(days=n)
    # Get start date for window size. Start date will be n days before actual start date
    # This is to avoid NaN values
    sd_n = sd - window

    # Generate "High" data using util function and using the specified date range
    df_high = get_data([symbol], dates=pd.date_range(sd_n, ed), colname="High")
    df_high.fillna(method='ffill', inplace=True)
    df_high.fillna(method='bfill', inplace=True)

    # Generate "Low" data using util function and using the specified date range
    df_low = get_data([symbol], dates=pd.date_range(sd_n, ed), colname="Low")
    df_low.fillna(method='ffill', inplace=True)
    df_low.fillna(method='bfill', inplace=True)

    # Generate "Close" data using util function and using the specified date range
    df_close = get_data([symbol], dates=pd.date_range(sd_n, ed), colname="Close")
    df_close.fillna(method='ffill', inplace=True)
    df_close.fillna(method='bfill', inplace=True)

    df_data = pd.DataFrame()
    df_data['High'] = df_high[symbol]
    df_data['Low'] = df_low[symbol]
    df_data['Close'] = df_close[symbol]
    return df_data

def author():
    return 'msyed46'

def run():
    simple_moving_average(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), n=20, symbol='JPM', gen_plot=True)
    bollinger_bands_percent(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), n=20, symbol='JPM', gen_plot=True)
    momentum(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), n=10, symbol='JPM', gen_plot=True)
    stochastic_osc(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), n=14, symbol='JPM', gen_plot=True)
    commodity_channel_index(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), n=20, symbol='JPM', gen_plot=True)

if __name__ == "__main__":
    run()

