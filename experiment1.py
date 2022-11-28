import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data
import matplotlib.pyplot as plt

import ManualStrategy as ms
import StrategyLearner as sl
from marketsimcode import compute_portvals

def author():
    return 'msyed46'

def generate_statistics(manual_strategy, strategy_learner, benchmark, type):
    pd.set_option('display.float_format', '{:.6f}'.format)

    # Final Portfolio Value
    pv_benchmark = benchmark[-1]
    pv_manual = manual_strategy[-1]
    pv_strategy = strategy_learner[-1]

    # Calculate Cummulative Return
    cr_benchmark = (benchmark[-1] / benchmark[0]) - 1
    cr_manual = (manual_strategy[-1] / manual_strategy[0]) - 1
    cr_strategy = (strategy_learner[-1] / strategy_learner[0]) - 1

    # Calculate Daily Returns
    dr_benchmark = ((benchmark / benchmark.shift(1)) - 1).iloc[1:]
    dr_manual = ((manual_strategy / manual_strategy.shift(1)) - 1).iloc[1:]
    dr_strategy = ((strategy_learner / strategy_learner.shift(1)) - 1).iloc[1:]

    # Calculate StDev of Daily Returns
    sd_benchmark = dr_benchmark.std()
    sd_manual = dr_manual.std()
    sd_strategy = dr_strategy.std()

    # Calculate Mean of Daily Returns
    mean_benchmark = dr_benchmark.mean()
    mean_manual = dr_manual.mean()
    mean_strategy = dr_strategy.mean()

    stats = {
        'Statistics': ['Final Portfolio Value', 'Cumulative Return', 'Standard Deviation of Daily Returns',
                       'Mean of Daily Returns'],
        'Benchmark Strategy': [pv_benchmark, cr_benchmark, sd_benchmark, mean_benchmark],
        'Manual Strategy': [pv_manual, cr_manual, sd_manual, mean_manual],
        'Strategy Learner': [pv_strategy, cr_strategy, sd_strategy, mean_strategy]
    }
    table = pd.DataFrame(stats)
    stats_table = table.set_index('Statistics')

    if type == 'insample':
        print("\n EXPERIMENT 1: IN SAMPLE STATISTICS \n")
        print(stats_table)

    if type == 'outsample':
        print("\n EXPERIMENT 1: OUT OF SAMPLE STATISTICS \n")
        print(stats_table)


def plot_insample(manual_strategy, strategy_learner, benchmark):
    plt.figure(figsize=(15, 9))
    plt.plot(benchmark, label="Benchmark", color="blue")
    plt.plot(manual_strategy, label="Manual Strategy", color="orange")
    plt.plot(strategy_learner, label="Strategy Learner", color="green")

    plt.title("Experiment 1: In-Sample Trading Strategies Comparisons")
    plt.xlabel('Date')
    plt.ylabel('Normalized Price ($)')
    plt.legend()
    plt.grid()
    plt.savefig('images/experiment1_insample.png', facecolor='wheat')
    plt.clf()

def plot_outsample(manual_strategy, strategy_learner, benchmark):
    plt.figure(figsize=(15, 9))
    plt.plot(benchmark, label="Benchmark", color="blue")
    plt.plot(manual_strategy, label="Manual Strategy", color="orange")
    plt.plot(strategy_learner, label="Strategy Learner", color="green")

    plt.title("Experiment 1: Out-Sample Trading Strategies Comparisons")
    plt.xlabel('Date')
    plt.ylabel('Normalized Price ($)')
    plt.legend()
    plt.grid()
    plt.savefig('images/experiment1_outsample.png', facecolor='wheat')
    plt.clf()

def run():
    # Test conditions
    sym = 'JPM'
    start_val = 100000
    start_date_in = dt.datetime(2008, 1, 1)
    end_date_in = dt.datetime(2009, 12, 31)
    start_date_out = dt.datetime(2010, 1, 1)
    end_date_out = dt.datetime(2011, 12, 31)

    # TRAIN STRATEGIES
    ms_1 = ms.ManualStrategy(verbose=False, impact=0.005, commission=9.95)
    sl_in = sl.StrategyLearner(verbose=False, impact=0.005, commission=9.95)
    sl_out = sl.StrategyLearner(verbose=False, impact=0.005, commission=9.95)
    bm = ms.ManualStrategy(verbose=False, impact=0.005, commission=9.95)

    # ===================================IN SAMPLE=====================================#
    sl_in.add_evidence(symbol=sym, sd=start_date_in, ed=end_date_in, sv=start_val)

    # Generate Trades
    ms_insample_trades = ms_1.testPolicy(symbol=sym, sd=start_date_in, ed=end_date_in, sv=start_val)
    sl_insample_trades = sl_in.gen_trades(sl_in.testPolicy(symbol=sym, sd=start_date_in, ed=end_date_in, sv=start_val))
    bm_insample_trades = bm.run_benchmark(symbol=sym, sd=start_date_in, ed=end_date_in, sv=start_val)

    # Compute Portfolio Values
    ms_insample_pv = compute_portvals(ms_insample_trades, start_val, 9.95, 0.005)
    ms_insample_pv_norm = ms_insample_pv['Values'] / ms_insample_pv['Values'][0]

    sl_insample_pv = compute_portvals(sl_insample_trades, start_val, 9.95, 0.005)
    sl_insample_pv_norm = sl_insample_pv['Values'] / ms_insample_pv['Values'][0]

    bm_insample_pv = compute_portvals(bm_insample_trades, start_val, 9.95, 0.005)
    bm_insample_pv_norm = bm_insample_pv['Values'] / bm_insample_pv['Values'][0]

    plot_insample(ms_insample_pv_norm, sl_insample_pv_norm, bm_insample_pv_norm)
    generate_statistics(ms_insample_pv_norm, sl_insample_pv_norm, bm_insample_pv_norm, 'insample')

    # ===================================OUT SAMPLE=====================================#
    sl_out.add_evidence(symbol=sym, sd=start_date_out, ed=end_date_out, sv=start_val)

    # Generate Trades
    ms_outsample_trades = ms_1.testPolicy(symbol=sym, sd=start_date_out, ed=end_date_out, sv=start_val)
    sl_outsample_trades = sl_out.gen_trades(
        sl_out.testPolicy(symbol=sym, sd=start_date_out, ed=end_date_out, sv=start_val))
    bm_outsample_trades = bm.run_benchmark(symbol=sym, sd=start_date_out, ed=end_date_out, sv=start_val)

    # Compute Portfolio Values
    ms_outsample_pv = compute_portvals(ms_outsample_trades, start_val, 9.95, 0.005)
    ms_outsample_pv_norm = ms_outsample_pv['Values'] / ms_outsample_pv['Values'][0]

    sl_outsample_pv = compute_portvals(sl_outsample_trades, start_val, 9.95, 0.005)
    sl_outsample_pv_norm = sl_outsample_pv['Values'] / sl_outsample_pv['Values'][0]

    bm_outsample_pv = compute_portvals(bm_outsample_trades, start_val, 9.95, 0.005)
    bm_outsample_pv_norm = bm_outsample_pv['Values'] / bm_outsample_pv['Values'][0]

    plot_outsample(ms_outsample_pv_norm, sl_outsample_pv_norm, bm_outsample_pv_norm)
    generate_statistics(ms_outsample_pv_norm, sl_outsample_pv_norm, bm_outsample_pv_norm, 'outsample')

if __name__ == "__main__":
    run()

