""""""
"""  		  	   		  	  		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
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
"""

import math
import sys

import numpy as np
import csv

import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it
import matplotlib.pyplot as plt
import time

def experiment_1(train_x, train_y, test_x, test_y, max_leaf_size):
    rmse_inSample = []
    rmse_outSample = []

    #Evaluate for leaf size 1 to max_leaf size
    for leaf_size in range(1, max_leaf_size+1):
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        learner.add_evidence(data_x=train_x, data_y=train_y)

        #In-sample
        predY_inSample = learner.query(train_x)
        rmse_inSample.append(math.sqrt(((train_y - predY_inSample) ** 2).sum() / train_y.shape[0]))

        #Out-sample
        predY_outSample = learner.query(test_x)
        rmse_outSample.append(math.sqrt(((test_y - predY_outSample) ** 2).sum() / test_y.shape[0]))

    x_axis = range(1, max_leaf_size+1)
    y_inSample = rmse_inSample
    y_outSample = rmse_outSample

    plt.plot(x_axis, y_inSample, label="In-Sample predictions")
    plt.plot(x_axis, y_outSample, label="Out-Sample predictions")
    plt.title("Figure 1: DTLearner with 100 Leaf Size")
    plt.legend()
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    #plt.show()


def experiment_2(train_x, train_y, test_x, test_y, max_leaf_size, numBags):
    rmse_inSample = []
    rmse_outSample = []

    for leaf_size in range(1, max_leaf_size+1):
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size":leaf_size, "verbose":False}, bags=numBags, verbose=False)
        learner.add_evidence(data_x=train_x, data_y=train_y)

        #In-Sample
        predY_inSample = learner.query(train_x)
        rmse_inSample.append(math.sqrt(((train_y - predY_inSample) ** 2).sum() / train_y.shape[0]))

        #Out-Sample
        predY_outSample = learner.query(test_x)
        rmse_outSample.append(math.sqrt(((test_y - predY_outSample) ** 2).sum() / test_y.shape[0]))


    x_axis = range(1, max_leaf_size + 1)
    y_inSample = rmse_inSample
    y_outSample = rmse_outSample


    plt.plot(x_axis, y_inSample, label="In-Sample predictions")
    plt.plot(x_axis, y_outSample, label="Out-Sample predictions")
    plt.title("Figure 2: BagLearner with 20 bags of DTLearners and with 100 Leaf Size")
    plt.legend()
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    #plt.show()

def experiment_3_rsquared(train_x, train_y, test_x, test_y, max_leaf_size):
    dt_outsample = []
    rt_outsample = []

    for leaf_size in range(1, max_leaf_size+1):
        #Create DT Learner
        dt_learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        dt_learner.add_evidence(data_x=train_x, data_y=train_y)

        #Get out-sample
        dt_predY_outSample = dt_learner.query(train_x)
        dt_corr_matrix = np.corrcoef(x=dt_predY_outSample, y=train_y)
        dt_corr = dt_corr_matrix[0, 1]
        dt_outsample.append(dt_corr)

        #Create RT Learner
        rt_learner = rt.RTLearner(leaf_size=leaf_size, verbose=False)
        rt_learner.add_evidence(data_x=train_x, data_y=train_y)

        #Get out-sample
        rt_predY_outSample = rt_learner.query(train_x)
        rt_corr_matrix = np.corrcoef(x=rt_predY_outSample, y=train_y)
        rt_corr = rt_corr_matrix[0, 1]
        rt_outsample.append(rt_corr)

    x_axis = range(1, max_leaf_size+1)
    y_axis_dt = dt_outsample
    y_axis_rt = rt_outsample

    plt.plot(x_axis, y_axis_dt, label="Decision Tree r-squared")
    plt.plot(x_axis, y_axis_rt, label="Random Tree r-squared")
    plt.title("Figure 3: Coefficient of Determination between DT and RT learners")
    plt.legend()
    plt.xlabel("Leaf Size")
    plt.ylabel("Coefficient of Determination (r-squared)")
    plt.show()


def mae(actual, predicted):
    mae = np.mean(np.abs(actual-predicted))
    return mae

def experiment_3_mae(train_x, train_y, test_x, test_y, max_leaf_size):
    dt_mae = []
    rt_mae = []

    for leaf_size in range(1, max_leaf_size+1):
        #Create DT Learner
        dt_learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        dt_learner.add_evidence(data_x=train_x, data_y=train_y)

        #Get out-sample
        dt_mae_outsample = dt_learner.query(test_x)
        mean_abs_error = mae(dt_mae_outsample, test_y)
        dt_mae.append(mean_abs_error)

        #Create RT Learner
        rt_learner = rt.RTLearner(leaf_size=leaf_size, verbose=False)
        rt_learner.add_evidence(data_x=train_x, data_y=train_y)

        #Get out-sample
        rt_mae_outsample = rt_learner.query(test_x)
        mean_abs_error = mae(rt_mae_outsample, test_y)
        rt_mae.append(mean_abs_error)

    x_axis = range(1, max_leaf_size + 1)
    y_axis_dt = dt_mae
    y_axis_rt = rt_mae

    plt.plot(x_axis, y_axis_dt, label="Decision Tree MAE")
    plt.plot(x_axis, y_axis_rt, label="Random Tree MAE")
    plt.title("Figure 4: MAE between DT and RT learners")
    plt.legend()
    plt.xlabel("Leaf Size")
    plt.ylabel("Mean Absolute Error")
    plt.show()

def experiment_3_build_time(train_x, train_y, max_leaf_size):
    dt_build_time = []
    rt_build_time = []

    for leaf_size in range(1, max_leaf_size+1):
        #Create DT Learner and time it
        dt_learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        dt_start = time.time()
        dt_learner.add_evidence(data_x=train_x, data_y=train_y)
        dt_stop = time.time()
        dt_build_time.append(dt_stop - dt_start)

        # Create RT Learner and time it
        rt_learner = rt.RTLearner(leaf_size=leaf_size, verbose=False)
        rt_start = time.time()
        rt_learner.add_evidence(data_x=train_x, data_y=train_y)
        rt_stop = time.time()
        rt_build_time.append(rt_stop - rt_start)

    x_axis = range(1, max_leaf_size + 1)
    y_axis_dt = dt_build_time
    y_axis_rt = rt_build_time

    plt.plot(x_axis, y_axis_dt, label="Decision Tree Build Time")
    plt.plot(x_axis, y_axis_rt, label="Random Tree Build time")
    plt.title("Figure 5: Build time for DT and RT learners")
    plt.legend()
    plt.xlabel("Leaf Size")
    plt.ylabel("Build Time (s)")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])

    if 'Istanbul' in inf.name:
        # Perform first column ignoring here
        myFile = np.genfromtxt('Data/Istanbul.csv', delimiter=',', skip_header=1, usecols=range(1,10))
        data = np.array(
            [list(s) for s in myFile]
        )

    else:
        data = np.array(
            [list(map(float, s.strip().split(","))) for s in inf.readlines()]
        )


    # compute how much of the data is training and testing  		  	   		  	  		  		  		    	 		 		   		 		  
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data  		  	   		  	  		  		  		    	 		 		   		 		  
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    #experiment_1(train_x, train_y, test_x, test_y, 100)
    #experiment_2(train_x, train_y, test_x, test_y, 100, 20)
    #experiment_3_rsquared(train_x, train_y, test_x, test_y, 200)
    #experiment_3_mae(train_x, train_y, test_x, test_y, 100)
    experiment_3_build_time(train_x, train_y, 50)

    # create a learner and train it  		  	   		  	  		  		  		    	 		 		   		 		  
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner  		  	   		  	  		  		  		    	 		 		   		 		  
    learner.add_evidence(train_x, train_y)  # train it  		  	   		  	  		  		  		    	 		 		   		 		  
    print(learner.author())  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    # evaluate in sample  		  	   		  	  		  		  		    	 		 		   		 		  
    pred_y = learner.query(train_x)  # get the predictions  		  	   		  	  		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		  	  		  		  		    	 		 		   		 		  
    print()  		  	   		  	  		  		  		    	 		 		   		 		  
    print("In sample results")  		  	   		  	  		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		  	   		  	  		  		  		    	 		 		   		 		  
    c = np.corrcoef(pred_y, y=train_y)  		  	   		  	  		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    # evaluate out of sample  		  	   		  	  		  		  		    	 		 		   		 		  
    pred_y = learner.query(test_x)  # get the predictions
    print(pred_y, test_y)
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		  	  		  		  		    	 		 		   		 		  
    print()  		  	   		  	  		  		  		    	 		 		   		 		  
    print("Out of sample results")  		  	   		  	  		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		  	   		  	  		  		  		    	 		 		   		 		  
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0,1]}")  
