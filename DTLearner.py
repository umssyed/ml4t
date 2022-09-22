import numpy as np


class DTLearner:

    def __init__(self, leaf_size=1, verbose=False):
        self.tree = None
        self.leaf_size = leaf_size
        self.verbose = verbose

    def add_evidence(self, data_x, data_y):
        self.tree = self.build_tree(data_x, data_y)
        return self.tree

    def searchTree(self, point):
        print("\n\nIN SEARCH TREE\n")
        predicted_value = -1
        node_index = 0 #starting node_index
        root_node = self.tree[node_index]

        current_node = root_node
        factor = int(current_node[0])

        #'''
        while factor != -1:
            # Identify the value of factor in point
            point_val = point[factor-1]

            # Identify the value of factor in tree
            tree_val = current_node[1]

            # If point_val is less than nodes split Value, go left
            if point_val <= tree_val:
                print(f"Going left")
                node_index += 1
                current_node = self.tree[node_index]
            else:
                print(f"Going right")
                node_index += int(current_node[-1]) #move down the right tree
                current_node = self.tree[node_index]

            factor = int(current_node[0])
        #'''

        #double check to see we are at leaf node
        if current_node[0] == -1:
            predicted_value = current_node[1]
        else:
            print(f"We ran into an error! Completed the entire search in the tree and not at a leaf node!")
        return predicted_value

    def query(self, points):
        print(self.tree)
        print(f"\nStarting to search tree...")
        predDataPoints_y = np.empty(0)
        for data_point in points:
            pred_y_point = self.searchTree(data_point)
            predDataPoints_y = np.append(predDataPoints_y, pred_y_point)

        print(f"After searching...\n")
        print(predDataPoints_y)
        return predDataPoints_y


    def build_tree(self, data_x, data_y):
        print(f"\nINSIDE BUILD TREE FUNCTION\n")
        print(f"x: \n{data_x}")
        print(f"{data_x.shape}, {data_x.shape[0]}, {self.leaf_size}")
        print(f"y: \n{data_y}")

        if data_x.shape[0] == 1:
            print(f"We are at a leaf node due to data_x.shape[0] == 1")
            return np.array([-1, data_y[0], -1, -1])

        if np.all(data_y == data_y[0]):
            print(f"Our y values are all the same.")
            return np.array([-1, data_y[0], -1, -1])

        if data_x.shape[0] <= self.leaf_size:
            print(f"Our data_x.shape[0] has less values than the leaf_size")
            return np.array([-1, np.mean(data_y), -1, -1])

        else:
            print(f"\nWe are in the else statement...\n")

            # Step 1. Find the correlation of data_x to data_y
            numCols = data_x.shape[1]
            correlation = np.empty(shape=[0, numCols])
            for i in range(0, numCols):
                c = np.around(np.corrcoef(data_x[:, i], data_y)[0, 1], decimals=3)
                if np.isnan(c):
                    correlation = np.append(correlation, 0)
                else:
                    correlation = np.append(correlation, abs(c))
            print(f"The correlation is: {correlation}")

            # Step 2. Find the max correlation x_y. This is the column (factor) we will use
            if np.around(np.max(correlation), decimals=3) == np.around(np.min(correlation), decimals=3):
                maxCorrIndex = 0
            else:
                maxCorrIndex = correlation.argmax()
            factor = maxCorrIndex + 1 #Index the factors from 1 -> N
            print(f"Chosen maxCorrIndex: {maxCorrIndex}")

            # Step 3. Sort the data_x, data_y with reference to the factor column
            # Sort data_x and data_y with median reference with maxCorrIndex
            sorted_data_x = data_x[data_x[:, maxCorrIndex].argsort()]
            sorted_data_y = data_y[data_x[:, maxCorrIndex].argsort()]

            print(f"SORTED DATA_X: \n{sorted_data_x}")
            print(f"SORTED DATA_Y: \n{sorted_data_y}")

            # Step 4. Find the split value using the median on the factor column
            # Step 4. (i) find split value by using the median of the maxCorrIndex sorted column in data_x
            splitVal = np.median(data_x[:, maxCorrIndex])
            print(f"\nSplit value chosen: {splitVal}")

            index = 0
            #Handle edge case - where all values of factor are less or equal than splitVal
            print("\nhandle edge case")
            print(sorted_data_x[:, maxCorrIndex])
            print(np.all(sorted_data_x[:, maxCorrIndex] <= 2.2))
            if np.all(sorted_data_x[:, maxCorrIndex] <= splitVal) or np.all(sorted_data_x[:, maxCorrIndex] > splitVal):
                print("We have left or right edge case!")
                m = np.mean(sorted_data_x[:, maxCorrIndex])
                for value in sorted_data_x[:, maxCorrIndex]:
                    if value <= m:
                        index += 1
                    else:
                        break

                lt_data_x = np.array(sorted_data_x[sorted_data_x[:, maxCorrIndex] <= m])
                lt_data_y = np.array(sorted_data_y[0:index])

                rt_data_x = np.array(sorted_data_x[index:, :])
                rt_data_y = np.array(sorted_data_y[index:])


            else:
                # Step 4. (ii) find row index where value is less than or equal to splitVal
                print(sorted_data_x[:, maxCorrIndex])
                for value in sorted_data_x[:, maxCorrIndex]:
                    if value <= splitVal:
                        index += 1
                    else:
                        break
                print(f"The row index to split on: {index}")
                # Step 5. Create the left tree for data_x and data_y
                # Left tree data x and data y
                lt_data_x = np.array(sorted_data_x[0:index, :])
                lt_data_y = np.array(sorted_data_y[0:index])

                # Step 6. Create the right tree for data_x and data_y
                # Right tree data x and data y
                rt_data_x = np.array(sorted_data_x[index:, :])
                rt_data_y = np.array(sorted_data_y[index:])

            # Step 7. Build the left and right tree
            # Build left tree
            leftTree = self.add_evidence(lt_data_x, lt_data_y)

            # Build right tree
            rightTree = self.add_evidence(rt_data_x, rt_data_y)

            # Final Step 8. Add to the tree
            if leftTree.ndim == 1:
                lt_shape = 1
            else:
                lt_shape = leftTree.shape[0]

            root = np.array([factor, splitVal, 1, lt_shape + 1])
            tree = np.vstack((root, leftTree, rightTree))
            return tree

    def author(self):
        return 'msyed46'
