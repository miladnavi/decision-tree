import random
from classifier_py_file import Node


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

    def nurture_forest(self, x_train, y_train, n_trees):
        """
        :param x_train: training set with data types in first row
        :param y_train: training labels
        :param n_trees: number of trees to make
        :return: Fit all trees in trees. No return value
        """
        self.trees    = self._seed_trees(n_trees)
        all_attr_sets = list(self._power_set(range(0, len(x_train[0, :]))))[:-1]  # no empty sets
        self.feature_sets  = random.sample(all_attr_sets, n_trees)

        for tree, feature_set in zip(self.trees, self.feature_sets):
            if len(feature_set) == 1:
                tree.fit(x_train[:, feature_set].reshape(len(x_train), 1), y_train)
            else:
                tree.fit(x_train[:, feature_set], y_train)

    def ask_forest_for_guidance(self, instance):
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
