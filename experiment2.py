import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data
import matplotlib.pyplot as plt

import StrategyLearner as sl
from marketsimcode import compute_portvals


def author():
    return 'msyed46'

def generate_statistics(sl1, sl2, sl3, sl4, impact):
    pd.set_option('display.float_format', '{:.6f}'.format)

    # Final Portfolio Value
    pv_sl1 = sl1[-1]
    pv_sl2= sl2[-1]
    pv_sl3 = sl3[-1]
    pv_sl4 = sl4[-1]

    # Calculate Cummulative Return
    cr_sl1 = (sl1[-1] / sl1[0]) - 1
    cr_sl2 = (sl2[-1] / sl2[0]) - 1
    cr_sl3 = (sl3[-1] / sl3[0]) - 1
    cr_sl4 = (sl4[-1] / sl4[0]) - 1

    # Calculate Daily Returns
    dr_sl1 = ((sl1 / sl1.shift(1)) - 1).iloc[1:]
    dr_sl2 = ((sl2 / sl2.shift(1)) - 1).iloc[1:]
    dr_sl3 = ((sl3 / sl3.shift(1)) - 1).iloc[1:]
    dr_sl4 = ((sl4 / sl4.shift(1)) - 1).iloc[1:]

    # Calculate StDev of Daily Returns
    sd_sl1 = dr_sl1.std()
    sd_sl2 = dr_sl2.std()
    sd_sl3 = dr_sl3.std()
    sd_sl4 = dr_sl4.std()

    # Calculate Mean of Daily Returns
    mean_sl1 = dr_sl1.mean()
    mean_sl2 = dr_sl2.mean()
    mean_sl3 = dr_sl3.mean()
    mean_sl4 = dr_sl4.mean()

    impact_1 = f'Impact {impact[0]}'
    impact_2 = f'Impact {impact[1]}'
    impact_3 = f'Impact {impact[2]}'
    impact_4 = f'Impact {impact[3]}'

    stats = {
        'Statistics': ['Final Portfolio Value', 'Cumulative Return', 'Standard Deviation of Daily Returns',
                       'Mean of Daily Returns'],
        impact_1: [pv_sl1, cr_sl1, sd_sl1, mean_sl1],
        impact_2: [pv_sl2, cr_sl2, sd_sl2, mean_sl2],
        impact_3: [pv_sl3, cr_sl3, sd_sl3, mean_sl3],
        impact_4: [pv_sl4, cr_sl4, sd_sl4, mean_sl4]
    }

    table = pd.DataFrame(stats)
    stats_table = table.set_index('Statistics')

    print("\n EXPERIMENT 2: IN SAMPLE STATISTICS FOR DIFFERENT IMPACT VALUES \n")
    print(stats_table)

def plot(sl1, sl2, sl3, sl4, impact):
    plt.figure(figsize=(15, 9))
    plt.plot(sl1, label=f"Impact: {impact[0]}", color="blue")
    plt.plot(sl2, label=f"Impact: {impact[1]}", color="black")
    plt.plot(sl3, label=f"Impact: {impact[2]}", color="green")
    plt.plot(sl4, label=f"Impact: {impact[3]}", color="orange")

    plt.title("Experiment 2: In-Sample Strategy Learner with different Impact values")
    plt.xlabel('Date')
    plt.ylabel('Normalized Price ($)')
    plt.legend()
    plt.grid()
    plt.savefig('images/experiment2.png', facecolor='wheat')
    plt.clf()


def run():
    # Test conditions
    sym = 'JPM'
    start_val = 100000
    start_date_in = dt.datetime(2008, 1, 1)
    end_date_in = dt.datetime(2009, 12, 31)
    impact = [0.000, 0.005, 0.010, 0.015]
    #impact = [0.01, 0.02, 0.03, 0.04]

    # Initiate SL with different impact values
    sl_1 = sl.StrategyLearner(verbose=False, impact=impact[0], commission=0.00)
    sl_2 = sl.StrategyLearner(verbose=False, impact=impact[1], commission=0.00)
    sl_3 = sl.StrategyLearner(verbose=False, impact=impact[2], commission=0.00)
    sl_4 = sl.StrategyLearner(verbose=False, impact=impact[3], commission=0.00)

    # Add evidence for each SL instance
    sl_1.add_evidence(symbol=sym, sd=start_date_in, ed=end_date_in, sv=start_val)
    sl_2.add_evidence(symbol=sym, sd=start_date_in, ed=end_date_in, sv=start_val)
    sl_3.add_evidence(symbol=sym, sd=start_date_in, ed=end_date_in, sv=start_val)
    sl_4.add_evidence(symbol=sym, sd=start_date_in, ed=end_date_in, sv=start_val)

    # Test policies for each SL instance and generate trades
    sl_1_trades = sl_1.gen_trades(sl_1.testPolicy(symbol=sym, sd=start_date_in, ed=end_date_in, sv=start_val))
    sl_2_trades = sl_2.gen_trades(sl_2.testPolicy(symbol=sym, sd=start_date_in, ed=end_date_in, sv=start_val))
    sl_3_trades = sl_3.gen_trades(sl_3.testPolicy(symbol=sym, sd=start_date_in, ed=end_date_in, sv=start_val))
    sl_4_trades = sl_4.gen_trades(sl_4.testPolicy(symbol=sym, sd=start_date_in, ed=end_date_in, sv=start_val))

    # Compute normalized values for each SL trades
    sl_1_pv = compute_portvals(sl_1_trades, start_val, impact=impact[0], commission=0.0)
    sl_1_pv_norm = sl_1_pv['Values']/sl_1_pv['Values'][0]

    sl_2_pv = compute_portvals(sl_2_trades, start_val, impact=impact[0], commission=0.0)
    sl_2_pv_norm = sl_2_pv['Values'] / sl_2_pv['Values'][0]

    sl_3_pv = compute_portvals(sl_3_trades, start_val, impact=impact[0], commission=0.0)
    sl_3_pv_norm = sl_3_pv['Values'] / sl_3_pv['Values'][0]

    sl_4_pv = compute_portvals(sl_4_trades, start_val, impact=impact[0], commission=0.0)
    sl_4_pv_norm = sl_4_pv['Values'] / sl_4_pv['Values'][0]

    plot(sl_1_pv_norm, sl_2_pv_norm, sl_3_pv_norm, sl_4_pv_norm, impact)
    generate_statistics(sl_1_pv_norm, sl_2_pv_norm, sl_3_pv_norm, sl_4_pv_norm, impact)

if __name__ == '__main__':
    run()
