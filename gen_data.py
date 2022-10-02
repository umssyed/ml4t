""""""  		  	   		  	  		  		  		    	 		 		   		 		  
"""  		  	   		  	  		  		  		    	 		 		   		 		  
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		  	  		  		  		    	 		 		   		 		  
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
  		  	   		  	  		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		  	  		  		  		    	 		 		   		 		  
"""  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
import math  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
# this function should return a dataset (X and Y) that will work  		  	   		  	  		  		  		    	 		 		   		 		  
# better for linear regression than decision trees  		  	   		  	  		  		  		    	 		 		   		 		  
def best_4_lin_reg(seed=1489683273):  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		  	  		  		  		    	 		 		   		 		  
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		  	  		  		  		    	 		 		   		 		  
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		  	  		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    np.random.seed(seed)

    #Set min/max for rows and columns as defined per requirements
    rows = np.random.randint(low=10, high=1001, size=1)[0]
    cols = np.random.randint(low=2, high=11, size=1)[0]
    print(f"rows: {rows}, cols: {cols}")

    # Initialize empty numpy arrays for x_datasets and y_datasets filled with zeros
    x_dataset = np.random.random((rows, cols))
    y_dataset = np.zeros((rows,))

    # x_dataset and y_datasets need to be filled
    # Y = 1x_1 + 2x_2 + 3x_3 + 4x_4 +...+ Nx_n where x_n values are the feature columns
    for i in range(0, cols):
        y_dataset += (i+1)*x_dataset[:, i]


    x = x_dataset
    y = y_dataset
    return x, y  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
def best_4_dt(seed=1489683273):  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		  	  		  		  		    	 		 		   		 		  
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		  	  		  		  		    	 		 		   		 		  
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		  	  		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    np.random.seed(seed)

    rows = np.random.randint(low=10, high=1001, size=1)[0]
    cols = np.random.randint(low=2, high=11, size=1)[0]
    print(f"rows: {rows}, cols: {cols}")

    #rows = 2
    #cols = 4
    x_dataset = np.random.random((rows, cols))*500
    #x_dataset = np.random.random_integers(1, 400, (rows, cols))
    y_dataset = np.zeros((rows,))

    for i in range(0, cols):
        multiplier = (i + 1) * 10
        if i == 0 or i == 1 or i == 2:
            y_dataset += multiplier * (x_dataset[:, i] ** (i + 2))
        else:
            y_dataset += multiplier * np.sin(x_dataset[:, i])


    #print(f"x = \n{x_dataset}")
    #print(f"y = \n{y_dataset}")
    x = x_dataset
    y = y_dataset
    return x, y  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
def author():  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		  	  		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    return "msyed46"  # Change this to your user ID
  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  	  		  		  		    	 		 		   		 		  
    print("they call me Tim.")
    best_4_dt()
    #best_4_lin_reg()
