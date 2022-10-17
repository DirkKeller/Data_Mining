To run the DecisionTree import DecisionTree from project1_decisiontree.py to instantiate DecisionTree object.

For example,  dt = DecisionTree(names) where "names" are the feature names considered in the analysis.

To get the feature names from the Eclipse dataset use the following line: feature_names = (data_sets.iloc[:, 2:44]).drop(['post'], axis=1).columns

To run tree_grow or any other function do: dt.tree_grow(train_x, train_y, nmin, minleaf, nfeat)