"""
Dal Col Giada 0093122
Keller Dirk 4282264
"""
import random
import numpy as np
from anytree import Node, RenderTree

class DecisionTree:
    def __init__(self, features: list):
        """
        Initializes the Decision Tree classifier.
            # Arguments
                :param features: list. The names of the features that are considered
        """
        self.feature_names = features
        self.tree_iter = 1

    def tree_grow(self, x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int) -> RenderTree:
        """
        A tree expansion function that builds a single decision tree depending on the nmin, minleaf, nfeat
        hyper-parameters. The  decision tree selects the feature that allows the best separation of the values and
        splits the data set at the level of the features value. The best separation is measured by the split's impurity,
        with gini-index as qualitative measure for a split's impurity. nmin and minleaf are used to stop growing the
        tree early, to prevent overfitting and/or to save computation. The feature with the larges impurity reduction
        will be expanded, the other candidate features will be discarded. The function will only perform a split, if it
        meets the nmin and minleaf constrain. If there is no split that meets the nmin constraint, the node becomes a
        leaf node. The function returns a tree object that is represents the best data-fitted hypothesis of the
        hypothesis space. It can be further used for prediction.
            # Arguments
                :param x: ndarray. A data matrix (2-dimensional array) containing the feature values.
                :param y: ndarray. A vector (1-dimensional array) of class labels. The class label must be binary, with
                    values coded as 0 and 1.
                :param nmin: integer. The number of observations that a node must contain at least, for it to be allowed
                    to be split.
                :param minleaf: integer. The minimum number of observations required for a leaf node; hence a split that
                    node with fewer than minleaf observations is not acceptable.
                :param nfeat: integer. The parameter nfeat denotes the number of features that should be considered for 
                    each split; the number of features are drawn at random
                :return: RenderTree. Tree object containing all linked nodes and their attributes to construct the full 
                    tree.
        """

        # Initialize a node list to store all current nodes, starting with the root node.
        node_list = [Node(f'root {self.tree_iter - 1}', features=x, label=y, attribute_index=None, best_split=None)]
        root = node_list[0]
        self.tree_iter = 1

        while len(node_list) != 0:
            # Extract the first node and remove it from the node list
            current_node = node_list[0]
            node_list.remove(current_node)

            # Check if the current node is a leaf node; expand, otherwise node is just deleted.
            if self.impurity(current_node.label) > 0:
                # Check if nmin constraint is satisfied; expand, otherwise node is just deleted.
                if current_node.features.shape[0] >= nmin:
                    # For each tree expansion initialize: the best feature index, split and impurity reduction score.
                    best_imp_reduc = 0
                    best_split, best_feat_idx = None, None
                    candidate_children = [None, None]

                    # Random forest random feature selection
                    rng_feat_idx = random.sample(range(current_node.features.shape[1]), k=nfeat)
                    sub_feat = current_node.features[:, rng_feat_idx].T

                    # Iterate through all selected features
                        # First, check the impurity reduction of the parent node, then perform the splits, argmax over
                        # the previous and current impurity reduction and save the value; select the feature with the
                        # best impurity reduction and its candidate children
                    for idx, feat in enumerate(sub_feat):
                        # Check that the number of unique to-be-splitted feature values is at least larger than 1.
                        if len(np.unique(feat)) > 1:
                            split_value = self.best_split(feat, current_node.label)
                            temp_Lchild, temp_Rchild = self.split(current_node, rng_feat_idx[idx], split_value, minleaf) # rng_feat_idx[idx] is the attribute we are splitting on
 
                            # Impurity reduction is only computed if the split is acceptable
                            if temp_Lchild is not None and temp_Rchild is not None:
                                new_imp_reduc = self.impurity_reduction(current_node, temp_Lchild, temp_Rchild)

                                # Incremental increase of the impurity reduction (no history);
                                if best_imp_reduc < new_imp_reduc:
                                    best_imp_reduc = new_imp_reduc
                                    best_feat_idx, best_split = rng_feat_idx[idx], split_value
                                    candidate_children[0], candidate_children[1] = temp_Lchild, temp_Rchild

                    # Append only valid 'candidate children' to the node list nodelist (and the tree)
                    if candidate_children[0] is not None and candidate_children[1] is not None:
                        candidate_children[0].parent, candidate_children[1].parent = current_node, current_node
                        current_node.best_split, current_node.feat_index = best_split, best_feat_idx
                        current_node.name = current_node.name + '\n' + self.feature_names[
                            best_feat_idx] + ': \n L > ' + str(round(best_split, 3)) + ' > R' +'\n' + '0:' + str(len(current_node.label[current_node.label==0])) + ' | 1:' + str(len(current_node.label[current_node.label==1]))
                        node_list.extend([candidate_children[0], candidate_children[1]])

                        self.tree_iter += 1
        return RenderTree(root)

    def tree_pred(self, x: np.ndarray, tr: RenderTree) -> list:
        """
        A prediction function that uses a tree object to predict classes based on data. The function requires a tree
        build on binary classification. The function returns a list of predicted labels.
            # Arguments
                :param x: ndarray. A data matrix (2-dimensional array) containing the feature values.
                :param tr: RenderTree. A tree object containing linked nodes.
                :return: list. List of predictions for all instances in x.
        """

        y_pred = []
        # For each instance in the data set:
        # split the data according to the best split of the selected feature in the trained tree object until a
        # leaf node is reached. Then take the majority vote of a leaf node to decide on the predicted class label.
        for idx, row in enumerate(x):
            current_node = tr.node
            while not current_node.is_leaf:
                split_attribute = current_node.feat_index
                if row[split_attribute] >= current_node.best_split:
                    current_node = current_node.children[0]
                else:
                    current_node = current_node.children[1]

            # Take the majority vote by sum and append it to the list
            if sum(current_node.label) > 0.5 * len(current_node.label):
                y_pred.append(1)
            elif sum(current_node.label) == 0.5 * len(current_node.label):
                y_pred.append(round(random.random(), 0))
            else:
                y_pred.append(0)
        return y_pred

    def tree_grow_b(self, x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int, m: int) -> list:
        """
        A wrapper function for the tree_grow() function that implements bagging. The function bootstraps a new sample
        from the data set (across all features) with the same sample size. Each time, a simple tree is grown and appended
        to a list of trees. The function returns a list with m trees that can be used for prediction.
            # Arguments
                :param m: integer. The number of bootstrap samples drawn from the data set; also the number of trees
                    returned.
                :return: list. A list of single trees.
                See tree_grow() for the other parameters
        """

        tree_list = []
        for idx in range(m):
            n = x.shape[0]
            v = range(n)
            v_new = np.random.choice(v, n, replace=True)
            X_new, Y_new = x[v_new, :], y[v_new]
            tree_list.append(self.tree_grow(X_new, Y_new, nmin, minleaf, nfeat))
        return tree_list

    def tree_pred_b(self, x: np.ndarray, tree_list: list) -> list:
        """
        A wrapper function for the tree_pred() function that implements bagging. The prediction function that uses a
        tree object to predict classes based on data. The function requires a list of trees build on binary classification. The
        function returns a list of predicted labels.
            # Arguments
                :param tree_list: list. Contains a list of tree objects
                :return: list. List of predictions for all instances in x.
                See tree_pred() for the other parameters
        """

        # For each tree: Predict the vector of labels for each instance in the data set.
        y_pred_b, majority_y_pred_b = [], []
        for idx, tree in enumerate(tree_list):
            y_pred_b.append(self.tree_pred(x, tree))

        # Across all predictions for the ith instance determine the majority vote across all tree predictions.
        y = np.array(y_pred_b)
        for i in range(y.shape[1]):
            if sum(y[:, i]) > 0.5 * len(y[:, i]):
                majority_y_pred_b.append(1)
            elif sum(y[:, i]) == 0.5 * len(y[:, i]):
                majority_y_pred_b.append(round(random.random(), 0))
            else:
                majority_y_pred_b.append(0)
        return majority_y_pred_b

    def impurity(self, v):
        """ Computes the impurity of a node: proportion class 1 elements/node total * proportion class 2 elements/node total """
        return (len(v[v == 0]) / len(v)) * (len(v[v == 1]) / len(v))

    def best_split(self, x, y):
        """  Computes the best split value according to the numeric attribute x. It considers binary split of the form x[i]>=c 
        where "c" is the average of two consecutive values of x in the sorted order. The choice of the best split is done accordingly
        to the best impurity reduction using gini-index.
            # Arguments:
                :param x: ndarray. A vector (1-dimensional array) containing the feature values.
                :param y: ndarray. A vector (1-dimensional array) of class labels. The class label must be binary, with
                    values coded as 0 and 1.
                :return: float. Value of the best split for the considered attribute."""

        x_sorted = np.sort(np.unique(x))
        split_points = (x_sorted[0:-1] + x_sorted[1:]) / 2
        imp_y = self.impurity(y)
        reduc_imp = []
        for i in range(len(split_points)):
            left_child = y[x >= split_points[i]]
            imp_left = self.impurity(left_child)
            prop_left = len(left_child) / len(x)
            right_child = y[x < split_points[i]]
            imp_right = self.impurity(right_child)
            prop_right = len(right_child) / len(x)
            reduc_imp.append(imp_y - prop_left * imp_left - prop_right * imp_right)

        index = reduc_imp.index(max(reduc_imp))
        return split_points[index]

    def impurity_reduction(self, parent: Node, left_child: Node, right_child: Node) -> float:
        """ Computes the impurity reduction of the split that generates Nodes "left_child" and "right_child" from the Node "parent" , based on gini-index. 
             # Arguments:
                :param parent: Node. The parent node.
                :param left_child: Node. The left child.
                :param right_child: Node. The right child.
                :return: float. Value of the impurity reduction of the considered split.
             """
        return self.impurity(parent.label) - self.impurity(left_child.label) * len(left_child.features) / len(
            parent.features) - self.impurity(right_child.label) * len(right_child.features) / len(parent.features)

    def split(self, parent: Node, feat_idx: int, pos, minleaf) -> tuple:  # add the contraint minleaf
        """ Computes the split from the Node "parent" satisfying the constraint of minleaf (see "tree_grow"). The split is done according to the value
        "pos" for the attribute "feat_idx": the left child contains values larger or equal than "pos".
            # Arguments:
                   :param parent: Node. The parent node on which apply the split.
                   :param feat_idx: int. Index of the attribute considered for the split.
                   :param pos: float. Split value.
                   :param minleaf: int. See "tree_grow".
                   :return: tuple. Tuple of two elements of class Node, containinf left adn right children. If no split is performed returns (None,None)."""
        if len(parent.features[parent.features[:, feat_idx] >= pos]) < minleaf or len(
                parent.features[parent.features[:, feat_idx] < pos]) < minleaf:
            return (None, None)
        else:
            Lchild = Node(f'L {self.tree_iter}', features=parent.features[parent.features[:, feat_idx] >= pos],
                          label=parent.label[parent.features[:, feat_idx] >= pos])
            Rchild = Node(f'R {self.tree_iter}', features=parent.features[parent.features[:, feat_idx] < pos],
                          label=parent.label[parent.features[:, feat_idx] < pos])
        return Lchild, Rchild



