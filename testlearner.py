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

    print("Test")
    '''
    train_x = np.array(
        [[2, 3, 2, 4, 1],
         [3, 4, 2, 4, 2],
         [1, 3, 1, 4, 3],
         [3, 3, 1, 2, 1],
         [1, 3, 2, 3, 3],
         [4, 3, 2, 3, 3],
         [3, 4, 4, 4, 1],
         [3, 1, 2, 2, 1]])

    train_y = np.array([2, 1, 4, 5, 8, 5, 6, 2])
    print(train_x)
    print(train_y)
    '''

    #learner = dt.DTLearner(leaf_size=1, verbose=False)
    #learner.add_evidence(data_x=train_x, data_y=train_y)
    #pred_dt = learner.query(test_x)
    #print(f"prediction:\n")
    #print(pred_dt)

    print("------------------------")
    #learner = rt.RTLearner(leaf_size=1, verbose=False)
    #learner.add_evidence(data_x=train_x, data_y=train_y)
    #pred_rt = learner.query(train_x)
    #print(f"prediction:\n")
    #print(pred_rt)

    print('\n------------------------')
    #print('Bag Learner')
    #learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size":50, "verbose":False}, bags=100, verbose=True)
    #learner.add_evidence(data_x=train_x, data_y=train_y)
    #learner.query(train_x)

    learner = it.InsaneLearner(verbose=False)
    learner.add_evidence(data_x=train_x, data_y=train_y)
    r = learner.query(train_x)
    print(r)


    '''
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
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		  	  		  		  		    	 		 		   		 		  
    print()  		  	   		  	  		  		  		    	 		 		   		 		  
    print("Out of sample results")  		  	   		  	  		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		  	   		  	  		  		  		    	 		 		   		 		  
    c = np.corrcoef(pred_y, y=test_y)  		  	   		  	  		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  
    '''
