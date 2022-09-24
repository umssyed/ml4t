import numpy as np
import BagLearner as bl
import LinRegLearner as lrl
class InsaneLearner:
    def __init__(self, verbose=False):
        self.verbose, self.result = verbose, None
        self.lrlBagLearners = [bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, verbose=self.verbose) for i in range(0, 20)]
    def add_evidence(self, data_x, data_y):
        #add evidence
        for bag in self.lrlBagLearners:
            bag.add_evidence(data_x, data_y)
    def query(self, points):
        for bag in self.lrlBagLearners:
            q = bag.query(points).T
            if self.result is None: self.result = q
            else: self.result = np.vstack((self.result, q))
        return np.mean(self.result, axis=0)
    def author(self):
        return 'msyed46'