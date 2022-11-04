""""""  		  	   		  	  		  		  		    	 		 		   		 		  
"""  		  	   		  	  		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
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
GT User ID: msyed46 (replace with your User ID)  		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903760502 (replace with your GT ID)  	

CHANGE THIS AT THE END	  	   		  	  		  		  		    	 		 		   		 		  
"""  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
import random as rand  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
import numpy as np


class QLearner(object):
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    This is a Q learner object.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    :param num_states: The number of states to consider.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type num_states: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :param num_actions: The number of actions available..  		  	   		  	  		  		  		    	 		 		   		 		  
    :type num_actions: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type alpha: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type gamma: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type rar: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type radr: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type dyna: int  		  	   		  	  		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		  	  		  		  		    	 		 		   		 		  
    """

    def author(self):
        return 'msyed46'

    def __init__(
        self,  		  	   		  	  		  		  		    	 		 		   		 		  
        num_states=100,  		  	   		  	  		  		  		    	 		 		   		 		  
        num_actions=4,  		  	   		  	  		  		  		    	 		 		   		 		  
        alpha=0.2,  		  	   		  	  		  		  		    	 		 		   		 		  
        gamma=0.9,  		  	   		  	  		  		  		    	 		 		   		 		  
        rar=0.5,  		  	   		  	  		  		  		    	 		 		   		 		  
        radr=0.99,  		  	   		  	  		  		  		    	 		 		   		 		  
        dyna=0,  		  	   		  	  		  		  		    	 		 		   		 		  
        verbose=False,  		  	   		  	  		  		  		    	 		 		   		 		  
    ):  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Constructor method  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		  	  		  		  		    	 		 		   		 		  
        self.num_actions = num_actions  		  	   		  	  		  		  		    	 		 		   		 		  
        self.s = 0  		  	   		  	  		  		  		    	 		 		   		 		  
        self.a = 0

        # INITIALIZE
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.qtable = np.zeros(shape=(num_states, num_actions))
        self.experience = [] # this is the experience list used for Dyna
        self.num_states = num_states


        #if self.dyna != 0:
        #    self.model = {}
        #    self.R = np.zeros(shape=(num_states, num_actions))
        #    self.T = np.full(shape=(num_states, num_actions, num_states), fill_value=0.00001)
        #    self.T_model = {}

  		  	   		  	  		  		  		    	 		 		   		 		  
    def querysetstate(self, s):  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Update the state without updating the Q-table  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
        :param s: The new state  		  	   		  	  		  		  		    	 		 		   		 		  
        :type s: int  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		  	  		  		  		    	 		 		   		 		  
        """

        # UPDATE STATES AND ACTION
        self.s = s  		  	   		  	  		  		  		    	 		 		   		 		  
        action = np.argmax(self.qtable[s])

        if self.verbose:  		  	   		  	  		  		  		    	 		 		   		 		  
            print(f"s = {s}, a = {action}")  		  	   		  	  		  		  		    	 		 		   		 		  
        return action  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    def query(self, s_prime, r):  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Update the Q table and return an action  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
        :param s_prime: The new state  		  	   		  	  		  		  		    	 		 		   		 		  
        :type s_prime: int  		  	   		  	  		  		  		    	 		 		   		 		  
        :param r: The immediate reward  		  	   		  	  		  		  		    	 		 		   		 		  
        :type r: float  		  	   		  	  		  		  		    	 		 		   		 		  
        :return: The selected action  		  	   		  	  		  		  		    	 		 		   		 		  
        :rtype: int  		  	   		  	  		  		  		    	 		 		   		 		  
        """

        # UPDATE Q-TABLE
        self.update_qtable(state=self.s, action=self.a, next_state=s_prime, reward=r)

        # ADD TO EXPERIENCE TUPLE
        exp_instance = [self.s, self.a, s_prime, r]
        self.experience.append(exp_instance)

        if self.dyna != 0:
            #self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + r * self.alpha

            # PERFORM HALLUCINATIONS WITH KNOWN EXPERIENCES FROM THE EXPERIENCE LIST
            for i in range(self.dyna):
                elm = np.random.randint(low=0, high=len(self.experience))
                exp = self.experience[elm]
                s_dyna = exp[0]
                a_dyna = exp[1]
                s_prime_dyna = exp[2]
                r_dyna = exp[3]  # Using this method for r_dyna is faster
                #r_dyna = self.R[s_dyna, a_dyna]

                # UPDATE Q-TABLE WITH HALLUCINATED EXPERIENCES
                self.update_qtable(state=s_dyna, action=a_dyna, next_state=s_prime_dyna, reward=r_dyna)

        # CHOSE NEXT ACTION
        random_int = np.random.uniform(low=0, high=1)
        if random_int < self.rar:
            action = np.random.randint(0, self.num_actions)
        else:
            action = np.argmax(self.qtable[s_prime])

        # UPDATING STATES
        self.s = s_prime
        self.a = action

        # UPDATE RAR = RAR * RADR
        self.rar = self.rar * self.radr

        if not self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")  		  	   		  	  		  		  		    	 		 		   		 		  
        return action

    #def next_action(self, num_actions, next_state):    takes longer to run. append later to query()
    #    random_int = np.random.uniform(low=0, high=1)
    #    if random_int < self.rar:
    #        action = np.random.randint(0, num_actions)
    #    else:
    #        action = np.argmax(self.qtable[next_state])
    #    return action

    def update_qtable(self, state, action, next_state, reward):
        # UPDATE Q-TABLE
        # Q'[s,a] = immediate reward + discounted reward
        immediate_reward = (1 - self.alpha) * self.qtable[state, action]
        later_rewards = self.qtable[next_state, np.argmax(self.qtable[next_state])]
        discounted_reward = self.alpha * (reward + self.gamma * later_rewards)
        self.qtable[state, action] = immediate_reward + discounted_reward


if __name__ == "__main__":  		  	   		  	  		  		  		    	 		 		   		 		  
    print("Remember Q from Star Trek? Well, this isn't him")  		  	   		  	  		  		  		    	 		 		   		 		  
