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
First clone or download the project and locate it wherever you want on your machine.

### Prerequisites
You need python > 3.0 (recommended 3.6.7):
* [Python](https://www.python.org/download/releases/3.0/)

Navigate with your terminal to the project dir and install the project's dependencies:

```
pip install -r requirements.txt
```

### Running

#### Pruning

Navigate with your terminal to the project dir and then in to the sub-dir `pruning` and run following cmd:
```
python main.py 1
```

There are 4 argument variables possible (1, 2, 3, 4) which trained the model with different data-set:<br />

1. [Iris](https://github.com/miladnavi/decision-tree/blob/master/dataset_61_iris.csv)
2. [Pump Status](https://github.com/miladnavi/decision-tree/blob/master/dataset_pump-status.csv)
3. [Breast Cancer](https://github.com/miladnavi/decision-tree/blob/master/dataset_13_breast-cancer.csv)
4. [LED Display](https://github.com/miladnavi/decision-tree/blob/master/dataset_LED-display-domain-7digit.csv)

#### Forest


### Add new data-set

#### Pruning
Navigate to the project dir and then in to the sub-dir `pruning`  in `main.py` file and this after last elif at the top and increase the number of elif:

```
elif argv is 5:
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
```

import you data:

```
data = np.array(pd.read_csv('../dataset_LED-display-domain-7digit.csv'))
```

Determinate the number of the training data and number of attribute (in this example is respectively: 460, 7):
```
X_train = np.array(data[:460, 0:7])
```

Determinate the kind of attributes (1 : `nominal` and 2 : `numeric`):
```
X_train = np.vstack((np.array([2, 2, 2, 2, 2, 2, 2]), X_train))
```

Determinate the number of the test data and number of attribute:
```
X_test = np.array(data[461:501, 0:7])
```

Determinate the test date for calculating Standard error `Error Complexity Pruning`  (in this example is respectively: 461:501, 0:7):
```
test_data = np.array(data[461:, :])
```
Now go to file `visualization.py` in `pruning` dir and proportional your elif number in your `main.py` add data's attribute's name in new array and new eleif:
```
attr4 = ["V1", "V2", "V3", "V4", "V5", "V6", "V7"]

elif argv is 4:
        attr = attr4
```
