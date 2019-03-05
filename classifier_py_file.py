import numpy as np
import pandas as pd
from timeit import default_timer as timer


class Node:

    def __init__(self):
        self.leaf = False
        self.attr = None
        self.data_type = None
        self.split_criterion = None
        self.children = []
        self.result = None
        self.pruned = False

    @staticmethod
    def _make_children(number):
        """
        :param number: number of children for given node
        :return: list of children
        """
        return [Node() for _ in range(number)]

    @staticmethod
    def _entropy(label_counts):
        """
        :param label_counts: vector of label counts for any partition
        :return: entropy (https://en.wikipedia.org/wiki/Entropy_(information_theory))
        """
        tmp = label_counts / np.sum(label_counts)
        etrp = tmp * np.log2(tmp, out=np.zeros_like(tmp), where=(label_counts != 0))

        return - np.sum(etrp)

    def _information_gain(self, cross_tab):
        """
        :param cross_tab: cross table for any subset in [x_train, y_train]
        :return: information gain (https://en.wikipedia.org/wiki/Information_gain_ratio)
        """
        etrp_before = self._entropy(np.bincount(np.sum(cross_tab, axis=0)))
        etrp_after = sum([np.sum(row) * self._entropy(row) / np.sum(cross_tab) for row in cross_tab])

        return etrp_before - etrp_after

    def _find_attr(self, x, y):
        """
        :param x: training set with data types in first row
        :param y: training labels
        :return: index of attribute to split by and information about the split
                    1 (numerical): threshold to split by
                    2 (categorical): class labels for children
        """
        ig = []  # List of Information Gain Values for each Attribute
        sp = []  # List of Thresholds to split by if Attribute is numerical
        for attribute in x.T:
            attr = attribute[1:]  # First row always contains Information about the Attributes themselves, no data
            if attribute[0] == 1:  # Attribute is numerical
                thresholds = np.convolve(np.sort(attr), np.array([.5, .5]), mode='valid')
                information_gains = [self._information_gain(np.array(pd.crosstab([attr > thresholds[i]], y)))
                                     for i in range(len(thresholds))]

                ig.append(max(information_gains))
                sp.append(thresholds[information_gains.index(max(information_gains))])

            if attribute[0] == 2:  # Attribute is categorical
                ig.append(self._information_gain(np.array(pd.crosstab(attr, y))))
                sp.append(np.unique(attr))

        return ig.index(max(ig)), sp[ig.index(max(ig))]

    def fit(self, x, y, data_types):
        """
        :param data_types: vector indicating data type of attributes
        :param x: training set with data types in first row
        :param y: training labels
        :return: Fit all trees in trees. No return value
        """
        unique, counts  = np.unique(y, return_counts=True)

        self.result = dict(zip(unique, counts))

        if len(self.result) == 1:
            self.leaf = True
            # print(self.result)

        else:
            self.attr, self.split_criterion = self._find_attr(np.row_stack((data_types, x)), y)
            self.data_type = data_types[self.attr]

            if self.data_type == 1:
                # only two Nodes, one for each side of the best split we found
                self.children = self._make_children(2)
                indices_left  = x[:, self.attr] > self.split_criterion  # left split, w/o first row
                indices_right = x[:, self.attr] <= self.split_criterion  # right split, w/o first row

                # concatenate the Information about the data_type from the first row
                self.children[0].fit(x[indices_left, :], y[indices_left], data_types)
                self.children[1].fit(x[indices_right, :], y[indices_right], data_types)

            if self.data_type == 2:
                # one child for every unique label in attr
                self.children = self._make_children(len(self.split_criterion))  # Node for every class

                # Iterate over all Nodes in children and call fit with data corresponding to the class label in attr
                for node, clss in zip(self.children, self.split_criterion):
                    node.fit(x[x[:, self.attr] == clss, :], y[x[:, self.attr] == clss], data_types)

    def predict(self, instance):
        """
        :param instance: training instance
        :return: result dict
        """
        instance = instance.reshape(len(instance))
        if not self.leaf:

            if self.data_type == 1:

                if instance[self.attr] > self.split_criterion:
                    return self.children[0].predict(instance)
                else:
                    return self.children[1].predict(instance)

            if self.data_type == 2:
                if 0 < len(np.where(instance[self.attr] == self.split_criterion)[0]):
                    clss = np.int(np.where(instance[self.attr] == self.split_criterion)[0])
                    return self.children[clss].predict(instance)
                else:
                    return self.result
        else:
            return self.result

    def evaluate(self, x, y, data_types):

        val10      = []
        x10, y10   = ten_folds(x, y)
        cmp        = [True for _ in range(10)]

        start = timer()

        for i in range(10):
            time_while = timer()
            print("Iteration ", i + 1, ". Time taken so far: ", int(time_while - start), " seconds.")

            tmp    = cmp.copy()
            tmp[i] = False
            xtrain = np.row_stack([x10[j] for j in range(10) if tmp[j]])
            xtest  = np.array(x10[i])
            ytrain = np.hstack([y10[j] for j in range(10) if tmp[j]])
            ytest  = np.array(y10[i])

            self.fit(xtrain, ytrain, data_types)
            predictions = [list(self.predict(row).keys())[0] for row in xtest]
            print("ACC in this batch:", sum(predictions == ytest) / len(ytest))
            val10.append(sum(predictions == ytest) / len(ytest))

        end = timer()
        return val10, end - start


def pre_process(x, y, data_types):
    """
    :param x: Training data
    :param y: Training labels
    :param data_types: vector with data types
    :return: X and Y without contradictions
    """
    tmp = np.column_stack((x, y))
    length = len(tmp[:, 0])
    m = np.empty((length, length))

    if np.int64 == type(data_types):
        x = x.reshape(len(y), 1)
        bmp = [True] + [False]

    else:
        bmp = [True for _ in data_types] + [False]

    for it in range(len(list(data_types)) + 1):
        if any(isinstance(k, str) for k in tmp[:, it]):
            tmp[:, it] = np.array(pd.factorize(tmp[:, it])[0])

    for i in range(length):
        for j in range(length):
            m[i, j] = all((tmp[i, :] == tmp[j, :]) == bmp)

    correct_indices = [ele == 0 for ele in sum(m)]
    return np.array(tmp[correct_indices, :-1]), np.array(y[correct_indices])


def ten_folds(x, y):
    """
    :param x: Training data
    :param y: Training labels
    :return: list with 10 subsets
    """
    length = len(y)

    x_list = [x[i:i + int(length / 10), :] for i in range(0, length, int(length / 10))]
    y_list = [y[i:i + int(length / 10)] for i in range(0, length, int(length / 10))]

    return x_list, y_list
