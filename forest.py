import random
from classifier_py_file import Node


class Forest:

    def __init__(self):
        self.trees = []

    @staticmethod
    def _seed_trees(number_of_trees):
        """
        set self.trees to list of trees with length number_of_trees
        """
        return [Node() for _ in range(number_of_trees)]

    @staticmethod
    def _power_set(self, seq):
        """
        Returns all the subsets of this set. This is a generator.
        """
        if len(seq) <= 0:
            yield []
        else:
            for item in self._power_set(seq[1:]):
                yield [seq[0]] + item
                yield item

    def nurture_forest(self, x_train, y_train, size_of_forest):
        """
        train all trees in self.trees
        """
        self.trees     = self._seed_trees(size_of_forest)
        all_attr_sets  = list(self._power_set(range(0, len(x_train[0, :]))))[:-1]  # no empty sets
        forest_sets    = random.sample(range(1, len(all_attr_sets)), size_of_forest)

        for node, feature_set in zip(self.trees, forest_sets):
            node.fit(x_train[:, feature_set], y_train)
