import numpy as np
import pandas as pd
import time
import sys
from classifier_py_file import Node
from without_pruning import tree_without_pruning
from min_err_pruning import min_err_pruning
from err_comp_pruning import err_complexity_pruning
from pessim_err_pruning import pessim_err_pruning

argv = int(sys.argv[1])
if argv is 1 or argv is None:

    # Data to make examples with
    data = np.array(pd.read_csv('../dataset_61_iris.csv'))
    np.random.shuffle(data)
    X_train = np.array(data[:99, 0:4])
    X_train = np.vstack((np.array([1, 1, 1, 1]), X_train))
    X_test = np.array(data[101:150, 0:4])
    Y_train = data[:99, 4]
    Y_test = data[101:150, 4]
    test_data = np.array(data[100:, :])

    # Training model
    tree1 = Node()
    start = time.time()
    tree1.fit(X_train, Y_train)
    end = time.time()
    print('Training Model: ', end - start)
    tree2 = Node()
    tree2.fit(X_train, Y_train)
    tree3 = Node()
    tree3.fit(X_train, Y_train)

    # Number of all labels which we data set have
    label_number = len(set(Y_train))

elif argv is 2:
    # Data to make examples with
    data = np.array(pd.read_csv('../dataset_pump-status.csv'))
    np.random.shuffle(data)
    X_train = np.array(data[:320, 0:5])
    X_train = np.vstack((np.array([1, 1, 1, 1, 1]), X_train))
    X_test = np.array(data[321:384, 0:5])
    Y_train = data[:320, 5]
    Y_test = data[321:384, 5]
    test_data = np.array(data[301:, :])

    # Training model
    tree1 = Node()
    start = time.time()
    tree1.fit(X_train, Y_train)
    end = time.time()
    print('Training model time complexity: ', end - start)
    tree2 = Node()
    tree2.fit(X_train, Y_train)
    tree3 = Node()
    tree3.fit(X_train, Y_train)

    # Number of all labels which we data set have
    label_number = len(set(Y_train))

elif argv is 3:
    # Data to make examples with
    data = np.array(pd.read_csv('../dataset_13_breast-cancer.csv'))
    np.random.shuffle(data)
    X_train = np.array(data[:200, 0:9])
    X_train = np.vstack((np.array([2, 2, 2, 2, 2, 2, 2, 2, 2]), X_train))
    X_test = np.array(data[200:278, 0:9])
    Y_train = data[:200, 9]
    Y_test = data[201:278, 9]
    test_data = np.array(data[201:, :])

    # Training model
    tree1 = Node()
    start = time.time()
    tree1.fit(X_train, Y_train)
    end = time.time()
    print('Training model time complexity: ', end - start)
    tree2 = Node()
    tree2.fit(X_train, Y_train)
    tree3 = Node()
    tree3.fit(X_train, Y_train)

    # Number of all labels which we data set have
    label_number = len(set(Y_train))

elif argv is 4:
    # Data to make examples with
    data = np.array(pd.read_csv('../dataset_LED-display-domain-7digit.csv'))
    np.random.shuffle(data)
    X_train = np.array(data[:460, 0:7])
    X_train = np.vstack((np.array([2, 2, 2, 2, 2, 2, 2]), X_train))
    X_test = np.array(data[461:501, 0:7])
    Y_train = data[:460, 7]
    Y_test = data[461:501, 7]
    test_data = np.array(data[461:, :])

    # Training model
    tree1 = Node()
    start = time.time()
    tree1.fit(X_train, Y_train)
    end = time.time()
    print('Training model time complexity: ', end - start)
    tree2 = Node()
    tree2.fit(X_train, Y_train)
    tree3 = Node()
    tree3.fit(X_train, Y_train)

    # Number of all labels which we data set have
    label_number = len(set(Y_train))

'''
Start Visualizing the Tree and make Pruning 
'''
print('##############################################################')
print('################ Tree without Pruning Started ################')
print('##############################################################')
tree_without_pruning(tree1, test_data, label_number, argv)


print('##############################################################')
print('########### Minimum error pruning method Started #############')
print('##############################################################')
min_err_pruning(tree1, label_number, test_data, argv)


print('##############################################################')
print('########### Pessimistic error pruning method Started #########')
print('##############################################################')
pessim_err_pruning(tree2, test_data, argv)


print('##############################################################')
print('########### Complexity error pruning method Started ##########')
print('##############################################################')
err_complexity_pruning(tree3, Y_test, X_train, X_test, test_data, argv)
