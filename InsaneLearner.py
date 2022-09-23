import numpy as np
import BagLearner as bl
import LinRegLearner as lrl

class InsaneLearner:

    def __init__(self, verbose=False):
        self.verbose, self.result = verbose, []
        self.lrlBagLearners = [bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, verbose=self.verbose) for i in range(0, 20)]

    def add_evidence(self, data_x, data_y):
        #add evidence
        for bag in self.lrlBagLearners:
            bag.add_evidence(data_x, data_y)

    def query(self, points):
        for bag in self.lrlBagLearners:
            self.result.append(bag.query(points))
        print(self.result)
        return sum(self.result)/len(self.result)

    def author(self):
        return 'msyed46'