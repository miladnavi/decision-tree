import numpy as np
import pandas as pd
from classifier_py_file import Node

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


tree = Node()
tree.fit(X_train, Y_train)

# print(X_train[4, :])
# print(tree.attr)
# print(tree.split_criterion)
#
# print(np.int(np.where(X_train[4, tree.attr] == tree.split_criterion)[0]))
#
# print(tree.children[3].attr)


print(tree.predict(X_train[4, :]))
