import queue
from graphviz import Graph

attr = ["sepallength", "sepalwidth", "petallength", "petalwidth"]

'''
Visualize tree 
'''
def tree_visualizer (tree):
    g = Graph(format='png')
    q = queue.Queue()
    g.node(str(id(tree)),
           'Attr: ' + str(attr[tree.attr]) + '\nSplit: ' + str(tree.split_criterion) + '\n Value: ' + str(tree.result),
           style='filled', color='green')

    for idx, val in enumerate(tree.children):
        q.put({
            "root": id(tree),
            "node": val,
        })

    while q.empty() is False:
        node = q.get()
        if node["node"].leaf is False:
            g.node(str(id(node["node"])),
                   'Attr: ' + str(attr[node["node"].attr]) + '\nSplit: ' + str(
                       node["node"].split_criterion) + '\n Value: ' + str(node["node"].result), style='filled',
                   color='green')
            g.edge(str(node["root"]), str(id(node["node"])))
        elif node["node"].pruned is True:
            g.node(str(id(node["node"])), 'Class: ' + str(node["node"].result), style='filled', color='red')
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
    g.render('tree-min-err-pruned', view=True)
