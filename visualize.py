import numpy as np
import pandas as pd
import queue
from classifier_py_file import Node
from graphviz import Graph

'''
Data to make examples with
'''
data = np.array(pd.read_csv('dataset_61_iris.csv'))
attr = ["sepallength", "sepalwidth", "petallength", "petalwidth"]

X_train = np.array(data[:, 0:4])
print(len(X_train))
X_train, indices = np.unique(np.round(X_train.astype(np.double)), axis=0, return_index=True)
X_train = np.vstack((np.array([1, 1, 1, 1]), X_train))
Y_train = data[indices, 4]

print(len(X_train))
tree = Node()
tree.fit(X_train, Y_train)

g = Graph(format='png')
q = queue.Queue()
g.node(str(id(tree)),
       'Attr: ' + str(attr[tree.attr]) + '\nSplit: ' + str(tree.split_criterion) + '\n Value: ' + str(tree.result), style='filled', color='green')

for idx, val in enumerate(tree.children):
    q.put({
        "root": id(tree),
        "node": val,
    })

while q.empty() == False:
    node = q.get()
    if node["node"].leaf is False:
        g.node(str(id(node["node"])),
               'Attr: ' + str(attr[node["node"].attr]) + '\nSplit: ' + str(
                   node["node"].split_criterion) + '\n Value: ' + str(node["node"].result), style='filled', color='green')
        g.edge(str(node["root"]), str(id(node["node"])))
    else:
        g.node(str(id(node["node"])), 'Class: ' + str(node["node"].result), style='filled', color='orange')
        g.edge(str(node["root"]), str(id(node["node"])))

    if len(node["node"].children) != 0:
        for idx, val in enumerate(node["node"].children):
            q.put({
                "root": id(node["node"]),
                "node": val,
            })
g.render('tree-visualization', view=True)



"""
from graphviz import Graph
g = Graph(format='png')
g.node(str(id(tree)), 'Attr: ' + str(attr[tree.attr]) + '\nSplit: ' + str(tree.split_criterion) + '\n Value: ' + str(tree.result))
g.node(str(id(tree.children[0])),  'Split: ' + str(tree.children[0].split_criterion) + '\n Value: ' + str(tree.children[0].result))
g.node('D', 'Split: ' + str(tree.children[1].split_criterion) + '\n Value: ' + str(tree.children[1].result))
g.node('E', 'Split: ' + str(tree.children[2].split_criterion) + '\n GiniIndex: ' + str(tree.children[2].result))
g.node('F', 'Split: ' + str(tree.children[3].split_criterion) + '\n GiniIndex: ' + str(tree.children[3].result))
g.node('G', 'Split: ' + str(tree.children[4].split_criterion) + '\n GiniIndex: ' + str(tree.children[4].result))
g.node('H', 'Split: ' + str(tree.children[5].split_criterion) + '\n GiniIndex: ' + str(tree.children[5].result))
g.node('I', 'Split: ' + str(tree.children[6].split_criterion) + '\n GiniIndex: ' + str(tree.children[6].result))
g.edge(index1, index2)
g.render('path', view=True)
"""
