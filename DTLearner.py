import numpy as np


class DTLearner:

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def add_evidence(self, data_x, data_y):
        print(f"x: {data_x}")
        print(f"y: {data_y}")

        if data_x.shape[0] <= self.leaf_size:
            return np.array([-1, np.mean(data_y), -1, -1])

        if data_x.shape[0] == 1:
            return np.array([-1, data_y[0], -1, -1])

        if np.all(data_y == data_y[0]):
            return np.array([-1, data_y[0], -1, -1])

        else:

            # Find the correlation
            numCols = data_x.shape[1]
            correlation = np.empty(shape=[0, numCols])
            for i in range(0, numCols):
                c = np.around(np.corrcoef(data_x[:, i], data_y)[0, 1], decimals=3)
                correlation = np.append(correlation, abs(c))
            print(f"The correlation is: {correlation}")

            # Find the max
            if np.around(np.max(correlation), decimals=3) == np.around(np.min(correlation), decimals=3):
                maxCorrIndex = 0
            else:
                maxCorrIndex = correlation.argmax()
            factor = maxCorrIndex
            print(f"maxCorrIndex: {maxCorrIndex}")

            # Sort data_x and data_y with median reference with maxCorrIndex
            sorted_data_x = data_x[data_x[:, maxCorrIndex].argsort()]
            sorted_data_y = data_y[data_x[:, maxCorrIndex].argsort()]
            print(sorted_data_x)
            print(sorted_data_y)
            # find split value by using the median of the maxCorrIndex sorted column in data_x
            splitVal = np.median(data_x[:, maxCorrIndex])

            # find row index where value is less than or equal to splitVal
            index = 0
            for values in sorted_data_x[:, maxCorrIndex]:
                if values <= splitVal:
                    index += 1
                else:
                    break
            #Left tree data x and data y
            lt_data_x = np.array(sorted_data_x[0:index, :])
            lt_data_y = np.array(sorted_data_y[0:index])

            #Right tree data x and data y
            rt_data_x = np.array(sorted_data_x[index:, :])
            rt_data_y = np.array(sorted_data_y[index:])

            #Build left tree
            leftTree = self.add_evidence(lt_data_x, lt_data_y)

            #Build right tree
            rightTree = self.add_evidence(rt_data_x, rt_data_y)

            if leftTree.ndim == 1:
                lt_shape = 1
            else:
                lt_shape = leftTree.shape[0]

            root = np.array([factor+1, splitVal, 1, lt_shape + 1])
            tree = np.vstack((root, leftTree, rightTree))
            return tree



    def query(self, points):

        pass

    def auth0r(self):
        return 'msyed46'
