import numpy as np


class RTLearner:

    def __init__(self, leaf_size=1, verbose=False):
        self.tree = None
        self.leaf_size = leaf_size
        self.verbose = verbose

    def printData(self):
        if self.verbose == True:
            print(f"\n================RTLEARNER=================")
            print(f"Final tree:")
            print(self.tree)
            print(f"\n{self.author()}\n")
            print(f"==========================================")

    def add_evidence(self, data_x, data_y):
        self.tree = self.build_tree(data_x, data_y)
        self.printData()
        return self.tree


    def build_tree(self, data_x, data_y):
        if data_x.shape[0] == 1:
            return np.array([-1, data_y[0], -1, -1])

        if np.all(data_y == data_y[0]):
            return np.array([-1, data_y[0], -1, -1])

        if data_x.shape[0] <= self.leaf_size:
            return np.array([-1, np.mean(data_y), -1, -1])

        else:
            # Step 1. Find the correlation of data_x to data_y
            numCols = data_x.shape[1]
            #print(f"numCols is: {numCols}, {data_x.shape[0]}")
            correlation = np.empty(shape=[0, numCols])
            for i in range(0, numCols):
                if np.std(data_x[:, i]) > 0:
                    c = np.around(np.corrcoef(data_x[:, i], data_y)[0, 1], decimals=3)
                    if np.isnan(c):
                        correlation = np.append(correlation, 0)
                    else:
                        correlation = np.append(correlation, abs(c))
                else:
                    correlation = np.append(correlation, 0)

            # Step 2. Find the max correlation x_y. This is the column (factor) we will use
            if np.around(np.max(correlation), decimals=3) == np.around(np.min(correlation), decimals=3):
                best_feature = 0
            else:
                best_feature = correlation.argmax()

            best_feature = np.random.randint(numCols)
            factor = best_feature + 1 #Index the factors from 1 -> N

            # Step 3. Sort the data_x, data_y with reference to the factor column
            # Sort data_x and data_y with median reference with best_feature
            sorted_data_x = data_x[data_x[:, best_feature].argsort()]
            sorted_data_y = data_y[data_x[:, best_feature].argsort()]


            # Step 4. Find the split value using the median on the factor column
            # Step 4. (i) find split value by using the median of the best_feature sorted column in data_x
            splitVal = np.median(data_x[:, best_feature])

            index = 0
            #Handle edge case - where all values of factor are less or equal than splitVal
            if np.all(sorted_data_x[:, best_feature] <= splitVal) or np.all(sorted_data_x[:, best_feature] > splitVal):
                m = np.mean(sorted_data_x[:, best_feature])
                for value in sorted_data_x[:, best_feature]:
                    if value <= m:
                        index += 1
                    else:
                        break

                lt_data_x = np.array(sorted_data_x[sorted_data_x[:, best_feature] <= m])
                lt_data_y = np.array(sorted_data_y[0:index])

                rt_data_x = np.array(sorted_data_x[index:, :])
                rt_data_y = np.array(sorted_data_y[index:])

                # Edge Case - If the split causes all values to the left with nothing on right,
                # Append leaf node with mean values of left tree's Y data
                if rt_data_x.shape[0] == 0 and rt_data_y.shape[0] == 0:
                    return np.array([-1, np.mean(lt_data_y), -1, -1])

                # Edge Case - If the split causes all values to the right with nothing on left,
                # Append leaf node with mean values of right tree's Y data
                if lt_data_x.shape[0] == 0 and lt_data_y.shape[0] == 0:
                    return np.array([-1, np.mean(rt_data_y), -1, -1])

            else:
                # Step 4. (ii) find row index where value is less than or equal to splitVal
                for value in sorted_data_x[:, best_feature]:
                    if value <= splitVal:
                        index += 1
                    else:
                        break
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
            leftTree = self.build_tree(lt_data_x, lt_data_y)

            # Build right tree
            rightTree = self.build_tree(rt_data_x, rt_data_y)

            # Final Step 8. Add to the tree
            if leftTree.ndim == 1:
                lt_shape = 1
            else:
                lt_shape = leftTree.shape[0]

            root = np.array([factor, splitVal, 1, lt_shape + 1])
            tree = np.vstack((root, leftTree, rightTree))
            return tree

    def searchTree(self, point):
        # Set predicted value to a negative one
        # Start with node index 0 - root
        # Identify root_node to be the tree's first (0) node
        predicted_value = -1
        node_index = 0  # starting node_index
        root_node = self.tree[node_index]

        # Set the current node to the root node and the factor
        # to the first node's factor value
        current_node = root_node
        factor = int(current_node[0])

        while factor != -1:
            # Identify the value of factor in point
            point_val = point[factor - 1]

            # Identify the value of factor in tree
            tree_val = current_node[1]

            # If point_val is less than nodes split Value, go left
            if point_val <= tree_val:
                node_index += 1
                current_node = self.tree[node_index]
            else:
                node_index += int(current_node[-1])  # move down the right tree
                current_node = self.tree[node_index]

            factor = int(current_node[0])

        # double check to see we are at leaf node
        if current_node[0] == -1:
            predicted_value = current_node[1]
        else:
            # Set the predicted value to a random negative value.
            # DELETE PRINT STATEMENT LATER
            print(f"We ran into an error! Completed the entire search in the tree and not at a leaf node!")
            predicted_value = -1000000
        return predicted_value

    def query(self, points):
        predDataPoints_y = np.empty(0)
        for data_point in points:
            pred_y_point = self.searchTree(data_point)
            predDataPoints_y = np.append(predDataPoints_y, pred_y_point)

        return predDataPoints_y

    def author(self):
        return 'msyed46'
