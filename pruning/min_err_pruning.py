import numpy as np
import pandas as pd
import queue
import time
from classifier_py_file import Node
from visualization import tree_visualizer
from predict import predict_result

def err_rate(tree, label_number):
    nodes_counter = 1
    leaf_counter = 0
    error_rate = 0
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
        error_rate += (sum(leaf.result.values()) / sum(tree.result.values())) * (
                    sum(leaf.result.values()) - max(leaf.result.values()) + label_number - 1) / (
                                                          sum(leaf.result.values()) + label_number)
    print('Error rate: ', error_rate)
    print('Leaf number: ', leaf_counter)
    print('Node number: ', nodes_counter)


def min_err_pruning(tree, label_number, test_data, argv):
    start = time.time()
    # Add root children to the queue
    tree_queue = queue.Queue()
    for child in tree.children:
        if child.leaf is False:
            tree_queue.put(child)

    # Start to pruning with compare each node and its leafs miss-classification error(BFS)
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

        # Calculate the miss-classification error with pruning
        classification_err_pruning = (sum(tree_node.result.values()) - max(
            tree_node.result.values()) + label_number - 1) / (sum(tree_node.result.values()) + label_number)

        # Calculate the sum of leafs miss-classification errors without pruning
        while leafs_queue.empty() is False:
            leaf = leafs_queue.get()
            classification_err_without_pruning += (sum(leaf.result.values()) / sum(tree_node.result.values())) * (
                    sum(leaf.result.values()) - max(leaf.result.values()) + label_number - 1) / (
                                                          sum(leaf.result.values()) + label_number)

        # Check if class-err with pruning smaller than without pruning then prune sub tree of node
        if classification_err_pruning <= classification_err_without_pruning:
            tree_node.children = []
            tree_node.pruned = True
            tree_node.leaf = True
        else:
            for child in tree_node.children:
                if child.leaf is False:
                    tree_queue.put(child)
    end = time.time()
    print('Time complexity: ', end - start)

    # Calculating error rate
    err_rate(tree, label_number)
    # Predict test data and print result
    predict_result(tree, test_data)
    # Visualization the pruned tree
    tree_visualizer(tree, 'tree-min-err-pruned', argv)
    return tree
