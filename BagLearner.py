import numpy as np
import DTLearner as dt
import RTLearner as rt

class BagLearner:

    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.bag_learners = []

        print(f"learner: {self.learner}")


    def printData(self):
        if self.verbose == True:
            print(f"\n================BAG-LEARNER=================")
            print(f"Implementation incomplete.....:")

            print(f"\n{self.author()}\n")
            print(f"==========================================")

    def add_evidence(self, data_x, data_y):
        print(f"Evidence added into BagLearner")

        #Add evidence will call the createBag function multiple times depending on
        #number of bags required


        #Test Code - Lets create for 1 bag
        self.createBagLearners(data_x, data_y)


    def createBagLearners(self, data_x, data_y):
        for i in range(0, self.bags):
            one_bag = self.createBag(data_x, data_y)
            self.bag_learners.append(one_bag)
        print(self.bag_learners)

    def createLearner(self, data_x, data_y):
        #this is basically calling the self.learner and inputting n_prime dataset
        #Think about kwargs here i.e. leaf_size

        # Step 1. Randomize data set.
        random_data_x, random_data_y = self.randomTrainingDataSet(data_x, data_y)
        print(f"random_data_x:\n{random_data_x}")
        print(f"random_data_y:\n{random_data_y}")
        bag = self.learner(**self.kwargs)
        bag.add_evidence(random_data_x, random_data_y)

        return bag

    def randomTrainingDataSet(self, data_x, data_y):
        #This will return n items where n is the number of items in data_x and data_y
        #The return value may have repeated items.
        cols = int(data_x.shape[1])
        random_x = np.zeros((0, cols))
        random_y = np.empty(0)

        n_data_items = data_x.shape[0] #Number of n data items in the training set\
        #Randomly index out n_items with replacement from n_data_items shape
        indices = np.random.choice(n_data_items, size=n_data_items, replace=True)

        #Use the above indices to append into random_x and random_y data sets
        for i in indices:
            random_x = np.vstack([random_x, data_x[i]])
            random_y = np.append(random_y, data_y[i])

        return random_x, random_y



    def author(self):
        return 'msyed46'