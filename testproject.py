import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data
import matplotlib.pyplot as plt

from ManualStrategy import ManualStrategy, run as ms_run
from StrategyLearner import StrategyLearner
import experiment1
import experiment2

def author():
    return 'msyed46'

# Manual Strategy
ms_run()

# Experiments
experiment1.run()
experiment2.run()






