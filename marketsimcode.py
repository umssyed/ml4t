import pandas as pd
import datetime as dt
from util import get_data

def compute_portvals(
        orders_file="./orders/orders-01.csv",
        start_val=1000000,
        commission=0,
        impact=0,
):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: Dataframe
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here
    end_date = orders_file.index[-1]
    # Using the dataframe of trades, construct a full order dataframe
    trades_of = orders_file
    trades_of = trades_of.assign(Order='HOLD')
    trades_of = trades_of.assign(Shares=0)

    # Start date should be

    date = trades_of.index
    shares = 0
    for i in range(0, len(trades_of)):
        exec = trades_of.loc[date[i], 'Execute']
        if exec > 0:
            trades_of.loc[date[i], 'Order'] = 'BUY'
            shares += exec
            trades_of.loc[date[i], 'Shares'] = exec
        else:
            trades_of.loc[date[i], 'Order'] = 'SELL'
            shares += exec
            trades_of.loc[date[i], 'Shares'] = exec
    orders_file = trades_of[trades_of['Execute'] != 0]
    orders_file = orders_file.drop('Execute', axis=1)

    # Set to 3 decimal places
    pd.options.display.float_format = '{:,.3f}'.format

    # Find Start Date and End date by scanning the orders_file
    df_len = len(orders_file)

    start_date = orders_file.index[0]
    dates = pd.date_range(start_date, end_date)
    # ---------------------PRICE DATAFRAME----------------------#
    # Set df_dates
    df_dates = pd.DataFrame(index=dates)

    # Loop through the orders file to find symbols
    sym = list(set(orders_file['Symbol'].values))

    # Create symbols dataframes with prices using util function get_data()
    df_sym = get_data(sym, dates)

    # Assign "Cash" to the df_sym
    df_price = df_sym.assign(Cash=start_val)
    #print(f"PRICE")
    #print(df_price)
    # ---------------------TRADES DATAFRAME----------------------#
    # Copy df_price to df_trades and drop SPY column
    df_trades = df_price
    df_trades = df_trades.drop('SPY', axis=1)

    # Initialize zero values for all symbols and Cash
    df_trades[sym] = 0
    df_trades['Cash'] = 0

    # Convert dataframe to float type object
    df_trades = df_trades.astype('float')
    # Go through orders file and fill in orders. Buy is positive, Sell is negative
    for i in range(0, df_len):
        trade_date = orders_file.index[i]
        trade_stock = orders_file['Symbol'][i]
        trade_order = orders_file['Order'][i]
        trade_shares = abs(orders_file['Shares'][i])
        actual_price = df_price.loc[trade_date][trade_stock]

        # Transaction Costs = Commission + Impact
        transaction_costs = commission + (impact * trade_shares * actual_price)
        df_trades.loc[trade_date]['Cash'] -= transaction_costs

        # For each order action, update shares and cash
        if trade_order == 'BUY':
            df_trades.loc[trade_date][trade_stock] += trade_shares
            df_trades.loc[trade_date]['Cash'] += -actual_price * trade_shares


        else:
            df_trades.loc[trade_date][trade_stock] += -trade_shares
            df_trades.loc[trade_date]['Cash'] += actual_price * trade_shares

    # ---------------------HOLDINGS DATAFRAME----------------------#
    # Copy df_trades to df_holdings
    df_holdings = df_trades
    st = pd.date_range(start_date, start_date)
    et = pd.date_range(end_date, end_date)

    # Start with initial cash and update initial cash only
    for index, values in df_holdings.iterrows():
        if index == st:
            values['Cash'] += start_val
        else:
            break

    # Go through the df_holdings and update holdings relative to previous day
    for col in df_holdings:
        df_holdings[col] = df_holdings[col].cumsum()

    # ---------------------VALUES DATAFRAME----------------------#
    # Copy df_holdings to df_values
    df_values = df_holdings

    # Update stock holding to values. Values = Prices x Holding
    for symbol in sym:
        df_values[symbol] = df_values[symbol] * df_price[symbol]

    pv = df_values.sum(axis=1)


    # ---------------------PORTVALS DATAFRAME----------------------#
    # Create portvals dataframe which is the sum of the values + cash
    portvals = pd.DataFrame(index=dates)
    portvals = portvals.assign(Values=pv).dropna(axis=0)

    return portvals


def author():
    return 'msyed46'
