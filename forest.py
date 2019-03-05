import random
import numpy as np
from timeit import default_timer as timer
from classifier_py_file import Node
from classifier_py_file import pre_process
from classifier_py_file import ten_folds


class Forest:

    def __init__(self):
        self.trees = []
        self.feature_sets = []

    def _power_set(self, seq):
        """
        :param seq: some sequence in form of an iterable
        :return: power set of the sequence
        """
        if len(seq) <= 0:
            yield []
        else:
            for item in self._power_set(seq[1:]):
                yield [seq[0]] + item
                yield item

    @ staticmethod
    def _seed_trees(n_trees):
        """
        :param n_trees: number of trees in the forest
        :return: list containing number_of_trees trees
        """
        return [Node() for _ in range(n_trees)]

    def fit(self, x, y, data_types, n_trees):
        """
        :param data_types: vector indicating attribute data types
        :param x: training set with data types in first row
        :param y: training labels
        :param n_trees: number of trees to make
        :return: Fit all trees in trees. No return value
        """
        self.trees    = self._seed_trees(n_trees)
        all_attr_sets = list(self._power_set(range(0, len(x[0, :]))))[:-1]  # no empty sets
        random.seed(0)
        self.feature_sets  = random.sample(all_attr_sets, n_trees)
        # print(self.feature_sets)

        for tree, feature_set in zip(self.trees, self.feature_sets):
            if len(feature_set) == 1:
                xtmp, ytmp = pre_process(x[:, feature_set], y, data_types[[feature_set]])
                # print(x[:, feature_set].shape)
                # print(y)
                # print(data_types[[feature_set]])
                tree.fit(xtmp, ytmp, data_types[[feature_set]])
            else:
                xtmp, ytmp = pre_process(x[:, feature_set], y, data_types[feature_set])
                tree.fit(xtmp, ytmp, data_types[feature_set])

    def predict(self, instance):
        """
        :param instance: instance w/o label
        :return: predicted label
        """
        ############################
        # 1. For each tree and the feature set that was used, we predict the classes given the relevant features of the
        #    instance
        # 2. We take the majority vote for each tree
        # 3. We take the majority vote over all trees to give the same weight to each tree
        ############################
        votes = [max(tree.predict(instance[feature_set]), key=tree.predict(instance[feature_set]).get)
                 for tree, feature_set in zip(self.trees, self.feature_sets)]

        # only return the vote with the highest count
        return max(votes, key=votes.count)

        # TODO: return a dict with counts for every class like Node.predict does

    def evaluate(self, x, y, data_types):

        val10 = []
        x10, y10 = ten_folds(x, y)
        cmp = [True for _ in range(10)]

        start = timer()

        for i in range(10):
            tmp    = cmp.copy()
            tmp[i] = False
            xtrain = np.row_stack([x10[j] for j in range(10) if tmp[j]])
            xtest  = np.array(x10[i])
            ytrain = np.hstack([y10[j] for j in range(10) if tmp[j]])
            ytest  = np.array(y10[i])

            self.fit(xtrain, ytrain, data_types, 2)
            predictions = [self.predict(row) for row in xtest]
            val10.append(sum(predictions == ytest) / len(ytest))

        end = timer()
        return np.mean(val10), end - start
