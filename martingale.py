""""""  		  	   		  	  		  		  		    	 		 		   		 		  
"""Assess a betting strategy.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
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
GT ID: ########################################### (replace with your GT ID)  		  	   		  	  		  		  		    	 		 		   		 		  
"""  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
def author():  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		  	  		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    return "msyed46"  # replace tb34 with your Georgia Tech username.
  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
def gtid():  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		  	  		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    return 903760502  # replace with your GT ID number
  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
def get_spin_result(win_prob):  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		  	  		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		  	  		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		  	  		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    result = False
    if np.random.random() <= win_prob:
        result = True  		  	   		  	  		  		  		    	 		 		   		 		  
    return result  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
def test_code():  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		  	  		  		  		    	 		 		   		 		  
    """  		  	   		  	  		  		  		    	 		 		   		 		  
    win_prob = 18/38  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once
    print(get_spin_result(win_prob))  # test the roulette spin

    # add your code here to implement the experiments
    #For Experiment 1, Figure 1
    figure1_exp1(win_prob, episodes=10)

    #For Experiment 1, Figure 2 and Figure 3
    figure2_3_exp1(win_prob, episodes=1000)

    #For Experiment 2, Figure 4 and Figure 5
    figure4_5_exp2(win_prob, episodes=1000)

def figure1_exp1(win_prob, episodes):
    #Generate Figure 1:
    #Run simple simulator 10 episodes and track winnings, starting
    #from 0 each time. Plot all 10 episodes on one chart.
    #X-Axis ranges ->  0 - 300
    #Y-Axis ranges -> -256 - 100

    experiment = np.zeros((episodes, 1001))

    for row in range(0, episodes):
        run_simulator_roulette_per_episode(experiment, row, win_prob)

    x_axis = [n for n in range(0, 1001)]
    y_axis_0 = experiment[0, :]
    y_axis_1 = experiment[1, :]
    y_axis_2 = experiment[2, :]
    y_axis_3 = experiment[3, :]
    y_axis_4 = experiment[4, :]
    y_axis_5 = experiment[5, :]
    y_axis_6 = experiment[6, :]
    y_axis_7 = experiment[7, :]
    y_axis_8 = experiment[8, :]
    y_axis_9 = experiment[9, :]

    fig, fig1 = plt.subplots()
    fig1.plot(x_axis, y_axis_0, label="Episode 1 Winnings")
    fig1.plot(x_axis, y_axis_1, label="Episode 2 Winnings")
    fig1.plot(x_axis, y_axis_2, label="Episode 3 Winnings")
    fig1.plot(x_axis, y_axis_3, label="Episode 4 Winnings")
    fig1.plot(x_axis, y_axis_4, label="Episode 5 Winnings")
    fig1.plot(x_axis, y_axis_5, label="Episode 6 Winnings")
    fig1.plot(x_axis, y_axis_6, label="Episode 7 Winnings")
    fig1.plot(x_axis, y_axis_7, label="Episode 8 Winnings")
    fig1.plot(x_axis, y_axis_8, label="Episode 9 Winnings")
    fig1.plot(x_axis, y_axis_9, label="Episode 10 Winnings")

    plt.title("Figure 1: Episode Winnings For Each Spin (Simple Simulator)")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend()
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings")
    plt.savefig('images/figure1.png')
    plt.close(fig)

def figure2_3_exp1(win_prob, episodes):
    #Generate Figure 2 and 3:
    #Run simple simulator 1000 episodes.

    experiment = np.zeros((episodes, 1001))
    for row in range(0, episodes):
        run_simulator_roulette_per_episode(experiment, row, win_prob)

    x_axis = [n for n in range(0, 1001)]
    mean = np.mean(experiment, axis=0) #axis = 0 for mean of a particular spin across all episodes
    median = np.median(experiment, axis=0) #axis = 0 for median of a particular spin across all episodes
    std = np.std(experiment, axis=0, ddof=0) # ddof=0 for "population std dev"

    #FIGURE 2
    #Plot the MEAN value of winnings
    #for each spin round using same axis bounds as Figure 1.
    #Plot mean+std and mean-std on the same plot
    fig, fig2 = plt.subplots()
    fig2.plot(x_axis, mean, label="Mean")
    fig2.plot(x_axis, mean+std, label="Mean + standard deviation")
    fig2.plot(x_axis, mean-std, label="Mean - standard deviation")

    plt.title("Figure 2: Mean and standard deviation (Simple Simulator)")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend()
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings")
    plt.savefig('images/figure2.png')
    plt.close(fig)


    #FIGURE 3
    #Plot the MEDIAN value of winnings
    #for each spin round using same axis bounds as Figure 1.
    #Plot median+std and median-std on the same plot
    fig, fig3 = plt.subplots()
    fig3.plot(x_axis, median, label="Median")
    fig3.plot(x_axis, median + std, label="Median + standard deviation")
    fig3.plot(x_axis, median - std, label="Median - standard deviation")

    plt.title("Figure 3: Median and standard deviation (Simple Simulator)")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend()
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings")
    plt.savefig('images/figure3.png')
    plt.close(fig)


def figure4_5_exp2(win_prob, episodes):
    #Generate Figure 4 and 5:
    #Run "realistic" simulator 1000 episodes.

    experiment = np.zeros((episodes, 1001))
    for row in range(0, episodes):
        run_realistic_roulette_per_episode(experiment, row, win_prob)

    x_axis = [n for n in range(0, 1001)]
    mean = np.mean(experiment, axis=0) #axis = 0 for mean of a particular spin across all episodes
    median = np.median(experiment, axis=0) #axis = 0 for median of a particular spin across all episodes
    std = np.std(experiment, axis=0, ddof=0) # ddof=0 for "population std dev"

    #FIGURE 4
    #Plot the MEAN value of winnings
    #for each spin round using same axis bounds as Figure 1.
    #Plot mean+std and mean-std on the same plot
    fig, fig4 = plt.subplots()
    fig4.plot(x_axis, mean, label="Mean")
    fig4.plot(x_axis, mean+std, label="Mean + standard deviation")
    fig4.plot(x_axis, mean-std, label="Mean - standard deviation")

    plt.title("Figure 4: Mean and standard deviation (Realistic Simulator)")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend()
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings")
    plt.savefig('images/figure4.png')
    plt.close(fig)

    #FIGURE 5
    #Plot the MEDIAN value of winnings
    #for each spin round using same axis bounds as Figure 1.
    #Plot median+std and median-std on the same plot
    fig, fig5 = plt.subplots()
    fig5.plot(x_axis, median, label="Median")
    fig5.plot(x_axis, median + std, label="Median + standard deviation")
    fig5.plot(x_axis, median - std, label="Median - standard deviation")

    plt.title("Figure 5: Median and standard deviation (Realistic Simulator)")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.legend()
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings")
    plt.savefig('images/figure5.png')
    plt.close(fig)


#3.2 - Formula fuction for "Explore the strategy and create some charts"
def run_simulator_roulette_per_episode(array, episode_num, win_prob):
    bet_winnings = 0
    spin_number = 1
    max_spins = 1001
    bet_amount = 1

    while spin_number < max_spins and bet_winnings <= 80:
        #If bet winnings has already reached $80, fill in all other spins
        #with $80. Do not spin again.
        if bet_winnings >= 80:
            array[episode_num, spin_number+1:] = 80
        else:
            won = get_spin_result(win_prob)
            if won is True:
                bet_winnings = bet_winnings + bet_amount
                array[episode_num, spin_number] = bet_winnings
                bet_amount = 1
            else:
                bet_winnings = bet_winnings - bet_amount
                array[episode_num, spin_number] = bet_winnings
                bet_amount = bet_amount * 2

        #After each spin, enter the bet_winnings of the spin in its appropriate episode
        #Go to the next bet number
        #array[episode_num, spin_number] = bet_winnings
        spin_number = spin_number + 1

#3.3 - Formula function for "A more realistic gambling simulator"
def run_realistic_roulette_per_episode(array, episode_num, win_prob):
    bet_winnings = 256 #start bet winnings with bank roll of $256
    max_bet_winnings = 336 #max bet winnings is $bet_winnings + $80 = $336
    spin_number = 1
    max_spins = 1001
    bet_amount = 1

    #while spins are less than 1000 and total bet winnings is neither above 336 or reached 0
    while spin_number < max_spins and bet_winnings <= 336:
        # If my bet winnings is larger than $336, forward fill $80
        if bet_winnings >= max_bet_winnings:
            array[episode_num, spin_number:] = 80
        # Else if my bet winnings reaches $0, forward fill -$256
        elif bet_winnings <= 0:
            array[episode_num, spin_number:] = -256
        # Else continue spinning
        else:
            won = get_spin_result(win_prob)
            if won is True:
                bet_winnings = bet_winnings + bet_amount
                array[episode_num, spin_number] = bet_winnings - 256
                bet_amount = 1
            else:
                bet_winnings = bet_winnings - bet_amount
                array[episode_num, spin_number] = bet_winnings - 256
                bet_amount = bet_amount * 2

                if bet_winnings < 256 and bet_winnings < bet_amount:
                    bet_amount = bet_winnings

        #After each spin, enter the bet_winnings of the spin in its appropriate episode
        #Go to the next bet number
        spin_number = spin_number + 1


if __name__ == "__main__":  		  	   		  	  		  		  		    	 		 		   		 		  
    test_code()  		  	   		  	  		  		  		    	 		 		   		 		  

    #TIPS
    #a.shape -> returns total dimension
    #a.shape[0] -> return number of rows
    #a.shape[1] -> return number of columns

