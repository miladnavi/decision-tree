import numpy as np
import pandas as pd

'''
Data to make examples with
'''
data = np.array(pd.read_csv('dataset_61_iris.csv'))
# :----------- metric
# X_train = np.array(data[:, 0:4])
# X_train = np.vstack((np.array([1, 1, 1, 1]), X_train))
# Y_train = np.array(data[:, 4])
# :----------- ordinal
X_train = np.array(data[:, 0:4])
X_train, indices = np.unique(np.round(X_train.astype(np.double)), axis=0, return_index=True)
X_train = np.vstack((np.array([2, 2, 2, 2]), X_train))
Y_train = data[indices, 4]


class Node:
    def __init__(self):
        self.leaf = False
        self.attr = None
        self.data_type = None
        self.split_criterion = None
        self.children = []
        self.result = None


    @staticmethod
    def _entropy(label_counts):
        """
        In:  Label counts in np.array
        Out: Entropy (0s and 1s respectively)
        """
        tmp = label_counts / np.sum(label_counts)
        etrp = tmp * np.log2(tmp, out=np.zeros_like(tmp), where=(label_counts != 0))

        return - np.sum(etrp)

    def _information_gain(self, cross_tab):
        """
         In:  Cross tab of X_train and Y_train
         Out: The information gain IG(A,Y) = E(Y) - SUM (|Di| / |D| * E(Di))
        """
        etrp_before = self._entropy(np.bincount(np.sum(cross_tab, axis=0)))
        etrp_after = sum([np.sum(row) * self._entropy(row) / np.sum(cross_tab) for row in cross_tab])

        return etrp_before - etrp_after

    def _find_attr(self, x_train, y_train):
        """
        In:  Data
        Out: 1. Index of attribute to split by
             2. Threshold to split by if found attribute is metric
             3. Classes if found attribute is ordinal
        """
        ig = []  # List of Information Gain Values for each Attribute
        sp = []  # List of Thresholds to split by if Attribute is metric

        for attribute in x_train.T:
            attr = attribute[1:]  # First row always contains Information about the Attributes themselves, no data

            if attribute[0] == 1:  # Attribute is metric
                thresholds = np.convolve(np.sort(attr), np.array([.5, .5]), mode='valid')
                information_gains = [self._information_gain(np.array(pd.crosstab([attr > thresholds[i]], y_train)))
                                     for i in range(0, (len(attr) - 1))]

                ig.append(max(information_gains))
                sp.append(thresholds[information_gains.index(max(information_gains))])

            if attribute[0] == 2:  # Attribute is ordinal
                ig.append(self._information_gain(np.array(pd.crosstab(attr, y_train))))
                sp.append(np.unique(attr))  # No Threshold can be calculated

        return ig.index(max(ig)), sp[ig.index(max(ig))]

    @staticmethod
    def _make_children(number):
        """
        In:  Number of children
        Out: List of nodes (children)
        """
        return [Node() for _ in range(number)]

    def fit(self, x_train, y_train):
        """
        Fit the model
        """
        x_data = x_train[1:, :]
        unique, counts = np.unique(y_train, return_counts=True)
        self.result = dict(zip(unique, counts))

        if len(self.result) == 1:
            self.leaf = True
            print(self.result)

        else:
            self.attr, self.split_criterion = self._find_attr(x_train, y_train)
            self.data_type = x_train[0, self.attr]

            if self.data_type == 1:
                # only two Nodes, one for each side of the best split we found
                self.children = self._make_children(2)
                indices_left = x_data[:, self.attr] > self.split_criterion  # left split, w/o first row
                indices_right = x_data[:, self.attr] <= self.split_criterion  # right split, w/o first row

                # concatenate the Information about the data_type from the first row
                self.children[0].fit(np.vstack((x_train[0, :], x_data[indices_left, :])), y_train[indices_left])
                self.children[1].fit(np.vstack((x_train[0, :], x_data[indices_right, :])), y_train[indices_right])

            if self.data_type == 2:
                # one child for every unique label in attr
                self.children = self._make_children(len(self.split_criterion))  # Node for every class

                # Iterate over all Nodes in children and call fit with data corresponding to the class label in attr
                for node, clss in zip(self.children, self.split_criterion):
                    node.fit(np.vstack((x_train[0, :], x_data[x_data[:, self.attr] == clss, :])),
                             y_train[x_data[:, self.attr] == clss])

    def predict(self, x):
        """
        Predict label for single instance
        """
        if not self.leaf:

            if self.data_type == 1:

                if x[self.attr] > self.split_criterion:
                    self.children[0].predict(x)
                else:
                    self.children[1].predict(x)

            if self.data_type == 2:
                self.children[self.split_criterion.index(x)].predict(x)


tree = Node()
tree.fit(X_train, Y_train)
