# Decision Tree

A decision tree is a supervised machine learning tool used in classification problems to predict the class of an instance. It is a tree like structure where internal nodes of the decision tree test an attribute of the instance and each subtree indicates the outcome of the attribute split. Leaf nodes indicate the class of the instance based on the model of the decision tree.<br />
In this repository you can find the implementation of decision tree and 3 methods `Pruning`, `Random Forest` and `Adaboost` to avoid overfitting and generalisation.
In this Project our goal is not only to create a classifier but also we are trying to optimize and accelerate the prediction progress for our classifier (`Decision Tree`). <br/>

#### Pruning

There are some approaches to tackle the generalisation problem and prevent overfitting. One of these methods which we used in this project called the pruning method. As the method's name explains itself, this method try to determine useless sub-trees (branches) and remove them and converts the nodes to the leaves. At the end it leads up to reduce the `misclassification error` and the `tree complexity`. We implemented three different pruning methods in this project: `Error-Complexity Pruning` , `Minimum-Error Pruning` and `Pessimistic-Error Pruning`

```
Note: Pruning reduces the miss-classfication error on test-dataset and not on training-dateset.
```




## Getting Start

### Prerequisites

### Running
