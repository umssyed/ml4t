""""""
"""MC1-P2: Optimize a portfolio.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
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
  		  	   		  	  		  		  		    	 		 		   		 		  
Student Name: Muhammad Uzair Shahid Syed (replace with your name)  		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: msyed46 (replace with your User ID)  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		  	  		  		  		    	 		 		   		 		  
"""

import datetime as dt

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from util import get_data, plot_data
import scipy.optimize as spo


# This is the function that will be tested by the autograder  		  	   		  	  		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality  		  	   		  	  		  		  		    	 		 		   		 		  

def compute_statistics(allocs, prices):
    """
    This function computes the cumulative returns (cr), average daily returns (adr), standard deviation
    of daily returns (sddr), the sharpe ratio (sr) and portfolio values give the allocation and prices. The function returns
    cr, adr, sddr and sr.

    :param allocs: Allocation of stocks
    :type allocs: list
    :param prices: The prices of each stock in "allocs"
    :type prices: pandas.core.frame.DataFrame
    :return: Several values for portfolio allocations, cumulative return, average daily returns, standard deviation of daily returns, and Sharpe ratio
    :rtype: float
    """
    initial_investment = 1  # initial value of our investment ($10,000)
    rfr = 0.00  # risk-free return of 0.00 per day
    sf = 252.00  # sampling frequency - daily therefore 252.0

    # ----Calculate Daily Portfolio Value----

    # Step 1: Normalize the daily prices
    normalized_prices = prices / prices.ix[0, :]

    # Step 2: Find normalized allocated values
    alloced = normalized_prices * allocs

    # Step 3: Find the position value based on our initial investment value
    pos_vals = alloced * initial_investment

    # Step 4: Find total value of portfolio each day by summing across axis=1
    portfolio_val = pos_vals.sum(axis=1)

    # Step 5: Find daily returns using formula (price[t]/price[t-1]) - 1
    daily_returns = portfolio_val.copy()
    daily_returns[1:] = (portfolio_val[1:] / portfolio_val[:-1].values) - 1
    daily_returns[0] = 0  # Set initial daily return to 0 since no previous data

    # ----Calculate Portfolio Statistics----

    # Cummulative Return = (cum_return_final / cum_return_start) - 1
    cr = (portfolio_val[-1] / portfolio_val[0]) - 1

    # Average Daily Return = mean of daily returns
    adr = daily_returns[1:].mean()

    # Standard Deviation of daily return = std dev of daily return
    sddr = daily_returns[1:].std()

    # Sharpe Ratio = sqrt(sf) x mean(daily_returns - rfr)/std(daily_returns - rfr), daily_rf = 0.0
    top = (daily_returns[1:] - rfr).mean()
    bottom = (daily_returns[1:] - rfr).std()
    sr = (sf ** (1 / 2)) * (top / bottom)
    return cr, adr, sddr, sr, portfolio_val


def sr_function(allocs, prices):
    """
    This function is the objective function to compute the negative value of the sharpe ratio. The function calls
    the compute_statistics function to calculate portfolio statistics. The result of this function is used
    to optimize allocation based on the sharpe ratio.

    :param allocs: Allocation of stocks
    :type allocs: list
    :param prices: The prices of each stock in "allocs"
    :type prices: pandas.core.frame.DataFrame
    :return: Negative sharpe ratio
    :rtype: float
    """
    # Compute portfolio statistics to calculate sharpe ratio
    cr, adr, sddr, sr, portfolio_value = compute_statistics(allocs, prices)
    # Multiple sharpe ratio by -1.0 to enable optimization for allocation of stocks
    y = -1.00 * sr
    return y


def constraint(alloc):
    """
    This function defines the constraint used for the optimization function. The constraint is to ensure
    the sum of all allocation is equal to one. For example: sum(x) = 1. Therefore the return value
    is sum(x) - 1. The equality constraint should be set in another key.

    :param allocs: Allocation of stocks
    :type allocs: list
    :return: Return sum(x) - 1
    :rtype: float
    """
    return np.sum(alloc) - 1


def check_alloc_sum(alloc):
    """
    This is a test function to test new allocation of stocks equal to one
    """
    sum = 0
    for x in alloc:
        sum += x
    print(f"\nSum of new allocations: {sum}\n")


def optimize_portfolio(
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        syms=["GOOG", "AAPL", "GLD", "XOM"],
        gen_plot=False,
):
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		  	  		  		  		    	 		 		   		 		  
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		  	  		  		  		    	 		 		   		 		  
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		  	  		  		  		    	 		 		   		 		  
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		  	  		  		  		    	 		 		   		 		  
    statistics.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  		  		  		    	 		 		   		 		  
    :type sd: datetime  		  	   		  	  		  		  		    	 		 		   		 		  
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  		  		  		    	 		 		   		 		  
    :type ed: datetime  		  	   		  	  		  		  		    	 		 		   		 		  
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		  	  		  		  		    	 		 		   		 		  
        symbol in the data directory)  		  	   		  	  		  		  		    	 		 		   		 		  
    :type syms: list  		  	   		  	  		  		  		    	 		 		   		 		  
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		  	  		  		  		    	 		 		   		 		  
        code with gen_plot = False.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type gen_plot: bool  		  	   		  	  		  		  		    	 		 		   		 		  
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		  	  		  		  		    	 		 		   		 		  
        standard deviation of daily returns, and Sharpe ratio  		  	   		  	  		  		  		    	 		 		   		 		  
    :rtype: tuple  		  	   		  	  		  		  		    	 		 		   		 		  
    """

    # Read in adjusted closing prices for given symbols, date range  		  	   		  	  		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		  	  		  		  		    	 		 		   		 		  
    prices = prices_all[syms]  # only portfolio symbols  		  	   		  	  		  		  		    	 		 		   		 		  
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		  	  		  		  		    	 		 		   		 		  

    # find the allocations for the optimal portfolio  		  	   		  	  		  		  		    	 		 		   		 		  
    # note that the values here ARE NOT meant to be correct for a test case  		  	   		  	  		  		  		    	 		 		   		 		  

    # Get daily portfolio value  		  	   		  	  		  		  		    	 		 		   		 		  
    port_val = prices_SPY  # add code here to compute daily portfolio values

    # Find how many symbols are being passed in.
    total_symbols = len(syms)
    # Equally allocate based on number of stocks n therefore, (1/n)
    allocs = total_symbols * [1 / total_symbols]
    # Set bounds for allocation between 0 and 1 only
    b = (0, 1)
    # Set the bounds for each stock
    bounds = list(b for allocation in range(0, total_symbols))
    # Set constraints to equal to zero. The function to be called is constraint defined
    # in this program
    con = {
        'type': 'eq',
        'fun': constraint
    }
    # Run the minimize function with the appropriate arguments
    results = spo.minimize(sr_function, allocs, args=(prices,), bounds=bounds, method='SLSQP', constraints=(con))
    # Set computed allocations to the result
    computed_allocs = results.x
    # Compute statistics again with the new computed allocations
    cr, adr, sddr, sr, port_val = compute_statistics(computed_allocs, prices)
    # Set return value allocs to computed allocs
    allocs = computed_allocs

    #To test only:
    #print(f"\n\nNew Allocs: {computed_allocs}")
    check_alloc_sum(computed_allocs)
    print(f"Portfolio values:")
    print(port_val, prices_SPY)
    print("")

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here  		  	   		  	  		  		  		    	 		 		   		 		  
        df_temp = pd.concat(
            [port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1
        )
        # print(f"df_temp is: ")
        # print(df_temp)
        # print()
        # df_temp.plot(kind="scatter", x=, y="Portfolio")
        df_temp[['Portfolio', 'SPY']].plot()
        plt.show()
        pass

    return allocs, cr, adr, sddr, sr


def test_code():
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader.  	2012-09-12
    """

    start_date = dt.datetime(2009, 1, 1)
    end_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2012, 9, 12)
    symbols = ["NFLX", "GOOG", "MSFT"]

    # Assess the portfolio  		  	   		  	  		  		  		    	 		 		   		 		  
    allocations, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )

    # Print statistics  		  	   		  	  		  		  		    	 		 		   		 		  
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader  		  	   		  	  		  		  		    	 		 		   		 		  
    # Do not assume that it will be called  		  	   		  	  		  		  		    	 		 		   		 		  
    test_code()
