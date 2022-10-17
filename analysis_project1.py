import time
import pandas as pd
import random
import numpy as np
from anytree.exporter import DotExporter
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from project1_decisiontree import DecisionTree


def eclipse_data(filename: list) -> tuple:
    """
    The funcion loads the eclipse data and splits it into a training set (past) and a test set (future).
    # Arguments
        :param filename: list. Accepts a list of filenames
        :return: tuple. The features and labels of the tain and test set, as well as, the dataframe of the data.
    """
    # Collect the data sets
    data_sets = []
    for idx, name in enumerate(filename):
        with open(name, 'r') as f:
            data = pd.read_csv(f, delimiter=';')
            data_sets.append(data)

    # Prepare the training set
    train_x = np.asarray((data_sets[0].iloc[:, 2:44]).drop(['post'], axis=1))
    train_y = np.asarray(data_sets[0].iloc[:, 3])
    train_y[train_y[:] != 0] = 1 # binarize the classes

    # Prepare the test set
    test_x = np.asarray((data_sets[1].iloc[:, 2:44]).drop(['post'], axis=1))
    test_y = np.asarray(data_sets[1].iloc[:, 3])
    test_y[test_y[:] != 0] = 1

    feature_names = (data_sets[0].iloc[:, 2:44]).drop(['post'], axis=1).columns
    return train_x, train_y, test_x, test_y, feature_names


def mc_nemar(y: list, pred1: list, pred2: list):
    """
    The McNemar test computes the cross table of the two predictions (i.e. a table containing the number of correctly
    predictions in both, number of correct prediction in 1 but no in 2, etc.). The cases correct1/non-correct2 and
    non-correct1/correct2 (anti-diagonal of the table) are the elements of the statistic for the test.
        # Arguments
            :param y: list. A list of ground truth labels
            :param pred1: list. A list of predictions of the first model
            :param pred2: list. A list of predictions of the second model
    """

    # Initialize boolean vector (prediction is the same as ground truth)
    bool1, bool2 = y == pred1, y == pred2
    # Computing the contingency tables for the McNemar test
    print(pd.crosstab(bool1, bool2))


def main():
    # Set seeds for consistency
    np.random.seed(1234)
    random.seed(1234)

    # Collect the training and test set
    filenames = ['eclipse-metrics-packages-2.0.csv', 'eclipse-metrics-packages-3.0.csv']
    train_x, train_y, test_x, test_y, names = eclipse_data(filenames)

    # Get a Tree object
    dt = DecisionTree(names)

    """ (1) without random forest, no bagging (all features) """
    tic = time.time()
    tr = dt.tree_grow(train_x, train_y, nmin=15, minleaf=5, nfeat=41)
    print(f'\nWithout random forest/Without bootstrap: \n In {round(time.time() - tic, 3)}s')

    # predict tree1 on the test set
    y_pred = dt.tree_pred(test_x, tr)
    print(f' accuracy: {round(accuracy_score(test_y, y_pred), 3)},'
          f' precision: {round(precision_score(test_y, y_pred), 3)},'
          f' recall: {round(recall_score(test_y, y_pred), 3)}')
    print(confusion_matrix(test_y, y_pred))

    # plotting
    DotExporter(tr.node).to_picture('NoBagNoBoot.png')

    """ (2) with random forest, no bagging (all features) """
    tic = time.time()
    tr2 = dt.tree_grow_b(train_x, train_y, nmin=15, minleaf=5, nfeat=41, m=100)
    print(f'\n Without random forest/with bagging: \nIn {round(time.time() - tic, 3)}')

    # predict tree2 on the test set
    y_pred2 = dt.tree_pred_b(test_x, tr2)
    print(f' accuracy: {round(accuracy_score(test_y, y_pred2), 3)},'
          f' precision: {round(precision_score(test_y, y_pred2), 3)},'
          f' recall: {round(recall_score(test_y, y_pred2), 3)}')

    print(confusion_matrix(test_y, y_pred2))

    """ (3) with random forest, with bagging (all features) """
    tic = time.time()
    tr3 = dt.tree_grow_b(train_x, train_y, nmin=15, minleaf=5, nfeat=6, m=100)
    print(f'\nWith random forest/with bagging: \nIn {round(time.time() - tic, 3)}')

    # predict tree2 on the test set
    y_pred3 = dt.tree_pred_b(test_x, tr3)
    print(f' accuracy: {round(accuracy_score(test_y, y_pred3), 3)},'
          f' precision: {round(precision_score(test_y, y_pred3), 3)},'
          f' recall: {round(recall_score(test_y, y_pred3), 3)}')
    print(confusion_matrix(test_y, y_pred3))

    """ McNemar test on the models """
    print("\nContingency table for model 1 and 2: ")
    mc_nemar(test_y, y_pred, y_pred2)
    print("\nContingency table for model 1 and 3: ")
    mc_nemar(test_y, y_pred, y_pred3)
    print("\nContingency table for model 2 and 3: ")
    mc_nemar(test_y, y_pred2, y_pred3)


main()