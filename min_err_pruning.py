import numpy as np
import pandas as pd
import queue
from classifier_py_file import Node
from visualization import tree_visualizer
from graphviz import Graph


'''
Data to make examples with
'''
data = np.array(pd.read_csv('dataset_61_iris.csv'))
attr = ["sepallength", "sepalwidth", "petallength", "petalwidth"]

X_train = np.array(data[:, 0:4])
X_train, indices = np.unique(np.round(X_train.astype(np.double)), axis=0, return_index=True)
X_train = np.vstack((np.array([1, 1, 1, 1]), X_train))
Y_train = data[indices, 4]

'''
Training model
'''
tree = Node()
tree.fit(X_train, Y_train)

# Number of all labels which we data set have
label_number = len(set(Y_train))

'''
Add root children to the queue
'''
tree_queue = queue.Queue()
for child in tree.children:
    if child.leaf is False:
        tree_queue.put(child)

'''
Start to pruning with compare each node and its leafs miss-classification error(BFS)
'''
while tree_queue.empty() is False:
    tree_node = tree_queue.get()
    classification_err_without_pruning = 0
    leafs_queue = queue.Queue()
    node_queue = queue.Queue()
    node_queue.put(tree_node)
    while node_queue.empty() is False:
        node = node_queue.get()
        if node.leaf is False:
            for child in node.children:
                node_queue.put(child)
        else:
            leafs_queue.put(node)
    '''
    Calculate the miss-classification error with pruning
    '''
    classification_err_pruning = (sum(tree_node.result.values()) - max(tree_node.result.values()) + label_number - 1) / (sum(tree_node.result.values()) + label_number)
    '''
    Calculate the sum of leafs miss-classification errors without pruning
    '''
    while leafs_queue.empty() is False:
        leaf = leafs_queue.get()
        classification_err_without_pruning += (sum(leaf.result.values()) / sum(tree_node.result.values())) * (sum(leaf.result.values()) - max(leaf.result.values()) + label_number - 1) / (sum(leaf.result.values()) + label_number)
    '''
    Check if class-err with pruning smaller than without pruning then prune sub tree of node
    '''
    print('classification_err_pruning: ', classification_err_pruning)
    print('classification_err_without_pruning: ', classification_err_without_pruning)
    if classification_err_pruning <= classification_err_without_pruning:
        tree_node.children = []
        tree_node.pruned = True
        tree_node.leaf = True
    else:
        for child in tree_node.children:
            if child.leaf is False:
                print(child.leaf)
                tree_queue.put(child)


# Visualization the pruned tree
tree_visualizer(tree)

