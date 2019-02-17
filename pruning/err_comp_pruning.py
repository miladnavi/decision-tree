import numpy as np
import pandas as pd
import queue
import time
from copy import deepcopy
from math import sqrt
from dict_argopt import argmin, argmax
from classifier_py_file import Node
from visualization import tree_visualizer
from predict import predict_result


def err_rate(tree, X_train):
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
        error_rate += ((sum(leaf.result.values()) - max(leaf.result.values())) / len(X_train))
    print('Error rate: ', error_rate)
    print('Leaf number: ', leaf_counter)
    print('Node number: ', nodes_counter)


def err_complexity_pruning(tree, Y_test, X_train, X_test, test_data, argv):
    start = time.time()
    # Pruning trees until there is no subtree to prune
    pruned_trees = [tree]
    eligible_tree = queue.Queue()

    # Checking whether there is any node left to prune
    node_count = 0
    for child in tree.children:
        if child.leaf is False:
            node_count += 1
    if node_count != 0: eligible_tree.put(tree)
    while eligible_tree.empty() is False:
        selected_tree = eligible_tree.get()

        # Add root children to the queue
        # path dictionary saves indices of the path to each node of the tree_queue
        path = {}
        tree_queue = queue.Queue()
        for child in selected_tree.children:
            if child.leaf is False:
                tree_queue.put(child)
                path[child] = [selected_tree.children.index(child)]

        # Calculating alpha for each node
        alpha = {}
        while tree_queue.empty() is False:
            tree_node = tree_queue.get()
            classification_err_without_pruning = 0.0
            leafs_queue = queue.Queue()
            node_queue = queue.Queue()
            node_queue.put(tree_node)
            leafs_count = 0.0
            while node_queue.empty() is False:
                node = node_queue.get()
                if node.leaf is False:
                    for child in node.children:
                        node_queue.put(child)
                else:
                    leafs_queue.put(node)
                    leafs_count += 1
            # Calculate the miss-classification error with pruning
            classification_err_pruning = (sum(tree_node.result.values()) - max(tree_node.result.values())) / len(
                X_train)

            # Calculate the sum of leafs miss-classification errors without pruning
            while leafs_queue.empty() is False:
                leaf = leafs_queue.get()
                classification_err_without_pruning += (sum(leaf.result.values()) - max(leaf.result.values())) / len(
                    X_train)

            alpha[tree_node] = (classification_err_pruning - classification_err_without_pruning) / (leafs_count - 1)

            for child in tree_node.children:
                if child.leaf is False:
                    tree_queue.put(child)
                    path[child] = path[tree_node] + [tree_node.children.index(child)]

        # Pruning the tree with minimum alpha and saving a copy of it to the pruned_trees list
        next_tree = deepcopy(selected_tree)
        node_to_prune = next_tree
        for i in path[argmin(alpha)]:
            node_to_prune = node_to_prune.children[i]
        node_to_prune.children = []
        node_to_prune.pruned = True
        node_to_prune.leaf = True
        pruned_trees.append(next_tree)

        # Checking whether there is any node left to prune
        node_count = 0
        for child in next_tree.children:
            if child.leaf is False:
                node_count += 1
        if node_count != 0: eligible_tree.put(next_tree)
    print('Pruning finished.')

    # Calculating standard_error for pruned trees
    standard_error = {}
    for tree in pruned_trees:
        miss_classification_count = 0.0
        for i in range(len(X_test)):
            if tree.predict(X_test[i]) != Y_test[i]:
                miss_classification_count += 1
        miss_classification_rate = miss_classification_count * 100 / len(X_test)
        standard_error[tree] = sqrt((miss_classification_rate * (100 - miss_classification_rate)) / len(X_test))
    best_pruned_tree = argmin(standard_error)
    leafs_queue = queue.Queue()
    node_queue = queue.Queue()
    node_queue.put(best_pruned_tree)
    best_pruned_tree_leafs_count = 0.0
    while node_queue.empty() is False:
        node = node_queue.get()
        if node.leaf is False:
            for child in node.children:
                node_queue.put(child)
        else:
            leafs_queue.put(node)
            best_pruned_tree_leafs_count += 1

    for tree in pruned_trees:
        if standard_error[best_pruned_tree] - 1 <= standard_error[tree] <= standard_error[best_pruned_tree] + 1:
            leafs_queue = queue.Queue()
            node_queue = queue.Queue()
            node_queue.put(tree)
            leafs_count = 0.0
            while node_queue.empty() is False:
                node = node_queue.get()
                if node.leaf is False:
                    for child in node.children:
                        node_queue.put(child)
                else:
                    leafs_queue.put(node)
                    leafs_count += 1
            if leafs_count < best_pruned_tree_leafs_count: best_pruned_tree = tree
    print('Best pruned tree found, with standard error: ' + str(standard_error[best_pruned_tree]) + ' and ' + str(
        best_pruned_tree_leafs_count - 1) + ' leafs.')
    end = time.time()
    print('Time Complexity: ', end - start)

    # Calculating error rate
    err_rate(best_pruned_tree, X_train)
    # Predict test data and print result
    predict_result(best_pruned_tree, test_data)

    tree_visualizer(best_pruned_tree, 'tree-err-comp-pruned', argv)
