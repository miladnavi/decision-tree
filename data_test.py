import numpy as np
import pandas as pd
from classifier_py_file import Node
from classifier_py_file import pre_process

from forest import Forest

# :-------------------------------------------
data = np.array(pd.read_csv('dataset_61_iris.csv'))
# data = np.array(pd.read_csv('dataset_13_breast-cancer.csv'))
'''
Two examples. One with the original numerical data from the iris data set. The second one is a discretization of the 
former without duplicates. Un-comment to use either one.
'''
# :----------- metric
# x_train = np.array(data[:, 0:4])
# x_train = np.vstack((np.array([1, 1, 1, 1]), x_train))
# y_train = np.array(data[:, 4])
# data_types = np.array([1, 1, 1, 1])

# :----------- ordinal
x_train = np.array(data[:, 0:4])
x_train = np.round(x_train.astype(np.int))
# x_train = np.vstack((np.array([2, 2, 2, 2]), x_train))
y_train = data[:, 4]
data_types = np.array([2, 2, 2, 2])


# x_train = np.array(data[:, 0:9])
# x_train = np.vstack((np.array([2, 2, 2, 2, 2, 2, 2, 2, 2]), x_train))
# y_train = data[:, 9]


x, y = pre_process(x_train, y_train, data_types)



'''
Initialize the model with a Node from the Node class. Node contains a method to fit the model with training data X_train
and the corresponding label vector Y_train. Note that X_train contains information about the data type in the first row.
'''
# tree = Node()
# tree.fit(x, y, data_types[[0]])

'''
After fitting, the model can be used to predict the label of a given instance. 
'''
instance = x_train[4, :]
# print(tree.predict(instance))

'''
Access instance attributes
'''
# print(vars(tree))


'''
Decision Forest Test
'''

forest = Forest()
forest.fit(x_train, y_train, data_types, 2)
print(forest.predict(instance))

# print(forest.ask_forest_for_guidance(instance))

# print(forest.trees[0].predict(instance))
# print(forest.trees[1].predict(instance))

# votes = [max(tree.predict(instance), key=tree.predict(instance).get)]


# print(max(votes, key=votes.count))

# print(forest.ask_forest_for_guidance(instance))

# print(x_train[:, [0]].reshape(len(x_train[:, [0]]), 1))

