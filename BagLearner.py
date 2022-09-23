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
        self.bag_learners = [] #Collection of learners in one bag
        self.numRows = 0
        self.predictions = None

    def printData(self):
        if self.verbose == True:
            print(f"\n================BAG-LEARNER=================")
            print(self.predictions)
            print(f"\n{self.author()}\n")
            print(f"==========================================")

    def add_evidence(self, data_x, data_y):
        self.numRows = data_y.shape[0]

        #Call the function to start creating bag learners
        for i in range(0, self.bags):
            one_learner = self.createLearner(data_x, data_y)
            self.bag_learners.append(one_learner)

    def createLearner(self, data_x, data_y):
        #For each learner, randomize the data_x and data_y dataset
        #Create one learner from this randomized dataset
        random_data_x, random_data_y = self.randomizeDataSet(data_x, data_y)
        oneLearner = self.learner(**self.kwargs)
        oneLearner.add_evidence(random_data_x, random_data_y)
        return oneLearner

    def randomizeDataSet(self, data_x, data_y):
        #This will return n items where n is the number of items in data_x and data_y
        #The return value may have repeated items.
        cols = int(data_x.shape[1])
        random_x = np.zeros((0, cols))
        random_y = np.empty(0)

        n_data_items = data_x.shape[0] #Number of n data items in the training set
        #Randomly index out n_items with replacement from n_data_items shape
        indices = np.random.choice(n_data_items, size=n_data_items, replace=True)

        #Use the above indices to append into random_x and random_y data sets
        for i in indices:
            random_x = np.vstack([random_x, data_x[i]])
            random_y = np.append(random_y, data_y[i])

        return random_x, random_y

    def query(self, points):
        #Query each learner in the bag and then take the mean
        predictions_stack = None

        for i in range(0, len(self.bag_learners)):
            one_learner = self.bag_learners[i]
            pred_from_one_learner = one_learner.query(points)
            pred_from_one_learner = np.atleast_2d(pred_from_one_learner).T

            if predictions_stack is None:
                predictions_stack = pred_from_one_learner
            else:
                predictions_stack = np.column_stack((predictions_stack, pred_from_one_learner))
        print("")
        #print(predictions_stack)
        #print(predictions_stack.shape)

        #if self.bags == 1:
        #    predictions = predictions_stack
        #else:
        #    predictions = np.mean(predictions_stack, axis=1)

        predictions = np.mean(predictions_stack, axis=1)
        print(predictions)
        print(predictions.shape)
        self.predictions = predictions
        #self.printData()
        return predictions

    def author(self):
        return 'msyed46'