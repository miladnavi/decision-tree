import numpy as np
import pandas as pd
import queue
from classifier_py_file import Node
from visualization import tree_visualizer
from predict import predict_result


def err_rate(tree, label_number):
    nodes_counter = 1
    leaf_counter = 0
    error_rate_min = 0
    node_queue = queue.Queue()
    leaf_queue = queue.Queue()
    for child in tree.children:
        nodes_counter += 1
        if child.leaf is True or child.pruned is True:
            leaf_counter += 1
            leaf_queue.put(child)
        else:
            node_queue.put(child)
    while node_queue.empty() is False:
        node = node_queue.get()
        for child in node.children:
            nodes_counter += 1
            if child.leaf is True or child.pruned is True:
                leaf_counter += 1
                leaf_queue.put(child)
            else:
                node_queue.put(child)
    while leaf_queue.empty() is False:
        leaf = leaf_queue.get()
        error_rate_min += (sum(leaf.result.values()) / sum(tree.result.values())) * (
                    sum(leaf.result.values()) - max(leaf.result.values()) + label_number - 1) / (
                                                          sum(leaf.result.values()) + label_number)
    print('Error rate minimum: ', error_rate_min)
    print('Leaf number: ', leaf_counter)
    print('Node number: ', nodes_counter)


def tree_without_pruning(tree, test_data, label_number, argv):
    # Calculating error rate
    err_rate(tree, label_number)

    # Predict test data and print result
    predict_result(tree, test_data)

    # Visualization tree without pruning
    tree_visualizer(tree, 'tree-without-pruning', argv)

    return tree
