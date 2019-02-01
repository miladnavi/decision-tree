import numpy as np
import pandas as pd
from classifier_py_file import Node

# :-------------------------------------------
data = np.array(pd.read_csv('dataset_61_iris.csv'))

'''
Two examples. One with the original numerical data from the iris data set. The second one is a discretization of the 
former without duplicates. Un-comment to use either one.
'''
# :----------- metric
# x_train = np.array(data[:, 0:4])
# x_train = np.vstack((np.array([1, 1, 1, 1]), x_train))
# y_train = np.array(data[:, 4])


# :----------- ordinal
x_train = np.array(data[:, 0:4])
x_train, indices = np.unique(np.round(x_train.astype(np.double)), axis=0, return_index=True)
x_train = np.vstack((np.array([1, 1, 1, 1]), x_train))
y_train = data[indices, 4]


'''
Initialize the model with a Node from the Node class. Node contains a method to fit the model with training data X_train
and the corresponding label vector Y_train. Note that X_train contains information about the data type in the first row.
'''
tree = Node()
tree.fit(x_train, y_train)

print(tree.predict(x_train[4, :]))

'''
After fitting, the model can be used to predict the label of a given instance. 
'''
instance = x_train[4, :]
print(tree.predict(instance))

'''
Access instance attributes
'''
print(vars(tree))
