"""Dal Col Giada 0093122
Keller Dirk """

import numpy as np
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
import pandas as pd
import random
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import time
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
#import statsmodels
#from statsmodels.stats.contingency_tables import mcnemar
np.random.seed(1234)
random.seed(1234)

def tree_grow(X: np.ndarray, Y : np.ndarray, nmin: int, minleaf: int, nfeat: int): # -> RenderTree:
    global k; k=1
    nodelist =[]
    root = Node(f'root {k-1}', features=X, label=Y, attribute_index=None, best_split=None)
    nodelist.append(root)

    while(len(nodelist) != 0):

        current_node = nodelist[0]
        nodelist.remove(current_node)

        best_impurity_reduction=0
        candidate_child_nodes=[None, None] # it saves the child nodes that gives the current best impurity reduction; it is updated after the firs loop
        best_split_value=None
        if impurity(current_node.label) > 0:

            c=random.sample(range(current_node.features.shape[1]),k=nfeat)

            selected_col = current_node.features[:,c].T


            #check the constraint nmin
            if current_node.features.shape[0]>=nmin :
                best_attribute_index=None
                for idx, attribute in enumerate(selected_col):

                    if len(np.unique(attribute))>1:
                        split_value = bestsplit(attribute, current_node.label)

                        temp_Lchild, temp_Rchild = split(current_node, c[idx], split_value, minleaf)

                        if (temp_Lchild,temp_Rchild)!=(None, None): # we can compute impurity reduction only the split is acceptable
                            new_impurity_reduction = impurity_reduction(current_node, temp_Lchild,temp_Rchild)


                            if best_impurity_reduction < new_impurity_reduction: #at the first for loop it results always true
                                best_impurity_reduction=new_impurity_reduction
                                best_attribute_index=c[idx]
                                best_split_value=split_value
                                candidate_child_nodes[0]=temp_Lchild
                                candidate_child_nodes[1]=temp_Rchild


                    # impurity reduction of the parent node - proportion elemts to the left * the impurity of the left child node - proportion elemts to the right * the impurity of the right child node -
                        # make all the splits -> save the impurity reduction in a list
                        # argmax over the impurity reduction list; only save the current impurity reduction if its better than the previous

                # when we exit the for loop we have checked all the nfeat attributes required and in the list "candidate_child" we have saved
                # children that need to be analyzed next: we append them in nodelist

                if candidate_child_nodes[0]!=None and candidate_child_nodes[1] != None:
                    candidate_child_nodes[0].parent=current_node
                    candidate_child_nodes[1].parent=current_node
                    current_node.best_split = best_split_value
                    current_node.attribute_index = best_attribute_index
                    current_node.name=current_node.name + '\n' + feature_names[best_attribute_index] + ': \n L > ' + str(round(best_split_value,3)) + ' > R'
                    # elif current_node.name[0] == 'R': current_node.name=current_node.name + ': \n' + feature_names[best_attribute_index] + ': <=' + str(round(best_split_value,3))
                    # else: current_node.name=current_node.name + ': \n' + feature_names[best_attribute_index] + ': ' + str(round(best_split_value,3))


                    nodelist.append(candidate_child_nodes[0])
                    nodelist.append(candidate_child_nodes[1])

                    k+=1
                   # print("k, shape of left and right children: \n",k,   candidate_child_nodes[0].features.shape, candidate_child_nodes[1].features.shape)

        #print("length of the nodelist", len(nodelist))
    return RenderTree(root)

def tree_pred(x, tr):
    y=[]
    for idx, row in enumerate(x):
        current_node=tr.node
        while(not current_node.is_leaf):
            split_attribute= current_node.attribute_index
            if row[split_attribute]>=current_node.best_split: current_node=current_node.children[0]
            else: current_node=current_node.children[1]
        if sum(current_node.label)>0.5*len(current_node.label): y.append(1)
        elif sum(current_node.label)==0.5*len(current_node.label): y.append(round(random.random(),0))
        else: y.append(0)
    return y

def tree_grow_b(X: np.ndarray, Y : np.ndarray, nmin: int, minleaf: int, nfeat: int, m: int): # -> RenderTree:
    T=[]
    for idx in range(m):
        n=X.shape[0]
        v=range(n)
        v_new = np.random.choice(v, n, replace=True)
        X_new=X[v_new, :]
        Y_new=Y[v_new]
        T.append(tree_grow(X_new,Y_new,nmin,minleaf,nfeat))
    return T

def tree_pred_b(T, x):
    Y=[]
    z=[]
    for idx, tree in enumerate(T):
        Y.append(tree_pred(x,tree))
    y=np.array(Y) # ith column corresponds to ith tree prediction
    for i in range(y.shape[1]):
        if sum(y[:,i])>0.5*len(y[:,i]): z.append(1)
        elif sum(y[:,i])==0.5*len(y[:,i]): z.append(round(random.random(), 0))
        else: z.append(0)
    return z

# def nodeattrfunc(node):
#     return '%s %s %s' % (str(impurity(node.label)), str(len(node.label[node.label==0])),str(len(node.label[node.label==1])))

# def edgeattrfunc(node):
#     sorted=np.sort(node.features[node.attribute_index])
#     return ' %s' % (sorted[node.pos_best_split])


def impurity(v):
    n0=len(v[v==0])
    n1=len(v[v==1])
    return n0/len(v) *n1/len(v)


def bestsplit(x,y):
    x_sorted=np.sort(np.unique(x))
    splitpoints=(x_sorted[0:-1]+x_sorted[1:])/2
    imp_y=impurity(y)
    reduc_imp=[]
    for i in range(len(splitpoints)):
        # if y[i] != y[i_prev]:
        left_child= y[x>=splitpoints[i]]
        imp_left=impurity(left_child)
        prop_left=len(left_child)/len(x)
        right_child=y[x<splitpoints[i]]
        imp_right=impurity(right_child)
        prop_right=len(right_child)/len(x)
        reduc_imp.append(imp_y-prop_left*imp_left-prop_right*imp_right)

    index=reduc_imp.index(max(reduc_imp))
    return splitpoints[index]


def impurity_reduction(parent: Node, Lchild: Node, Rchild: Node) -> int:
# we don't have to pass to "impurity" the attribute beacuse it's not used, we only need to know how are the elements divided
# in the two classes (how many in 0 and how many in 1)
# Node.features is a matrix: if we want to know the number of elements in that node we have to compute the number of rows of that matrix
    return impurity(parent.label) - impurity(Lchild.label) * len(Lchild.features) / len(parent.features) - impurity(Rchild.label) * len(Rchild.features) / len(parent.features)


def split(parent: Node, type, pos: int, minleaf) -> tuple: #add the contraint minleaf
    if len(parent.features[parent.features[:,type] >= pos])<minleaf or len(parent.features[parent.features[:,type] < pos])<minleaf: return (None,None)
    else: # w ehav to find a way to pass some other names; IDEA: save the depth of the three as another attribute of the object Node?
        # or better save the attribute we have split for (useful for the prediction later)

        Lchild = Node(f'L {k}', features=parent.features[parent.features[:,type] >= pos], label=parent.label[parent.features[:,type] >= pos])
        Rchild = Node(f'R {k}', features=parent.features[parent.features[:,type] < pos], label=parent.label[parent.features[:, type] < pos])
        return Lchild, Rchild














data_sets = []
for idx, name in enumerate(['eclipse-metrics-packages-2.0.csv', 'eclipse-metrics-packages-3.0.csv', 'pimaindians.txt']):
    with open(name, 'r') as f:
        data = pd.read_csv(f, delimiter=';')
        data_sets.append(data)

train_X=data_sets[0].iloc[:,2:44]
trainY= np.asarray(train_X.iloc[:,1])
trainY[trainY[:] != 0] = 1
trainX=train_X.drop(['post'], axis=1)

global feature_names; feature_names = trainX.columns

trainX = np.asarray(trainX)

test_X=data_sets[1].iloc[:,2:44]
testY= np.asarray(test_X.iloc[:,1])
testY[testY[:] != 0] = 1

testX=np.asarray(test_X.drop(['post'], axis=1))



""" (1) without bootstrapping, no bagging (all features) """
# tic=time.time()
# tr=tree_grow(trainX, trainY, nmin=15, minleaf=5, nfeat=41)
# print('No bagging/No bootstrap: ', time.time()-tic)

# # predict tree1 on the test set
# Y_pred=tree_pred(testX, tr)
# print('accuracy: ', accuracy_score(testY, Y_pred), ', precision: ', precision_score(testY, Y_pred), ', recall: ', recall_score(testY, Y_pred ))
# print(confusion_matrix(testY, Y_pred))

# # plotting
# DotExporter(tr.node).to_picture('NoBagNoBoot.png')


# """ (2) with bootstrapping, no bagging (all features) """
# tic=time.time()
# tr2=tree_grow_b(trainX, trainY, nmin=15, minleaf=5, nfeat=41, m=100)
# print('No bagging/With bootstrap: ',time.time()-tic)

# # predict tree2 on the test set
# Y_pred2=tree_pred_b( tr2, testX)
# print('accuracy: ', accuracy_score(testY, Y_pred2), ', precision: ', precision_score(testY, Y_pred2), ', recall: ', recall_score(testY, Y_pred2))
# print(confusion_matrix(testY, Y_pred2))

# """ (3) with bootstrapping, with bagging (all features) """
# tic=time.time()
# tr3=tree_grow_b(trainX, trainY, nmin=15, minleaf=5, nfeat=6, m=100)
# print('With bagging/With bootstrap: ',time.time()-tic)

# # predict tree2 on the test set
# Y_pred3=tree_pred_b(tr3, testX)
# print( 'accuracy: ', accuracy_score(testY, Y_pred3),', precision: ', precision_score(testY, Y_pred3), ', recall: ', recall_score(testY, Y_pred3))
# print(confusion_matrix(testY, Y_pred3))



# This part is for computing the contingency tables  for the McNemar test
tr=tree_grow(trainX, trainY, nmin=15, minleaf=5, nfeat=41)
Y_pred=tree_pred(testX, tr)
bool1=testY==Y_pred # boolean vector with True if the prediction is correct or false

tr2=tree_grow_b(trainX, trainY, nmin=15, minleaf=5, nfeat=41, m=100)
Y_pred2=tree_pred_b( tr2, testX)
bool2=testY==Y_pred2

tr3=tree_grow_b(trainX, trainY, nmin=15, minleaf=5, nfeat=6, m=100)
Y_pred3=tree_pred_b(tr3, testX)
bool3=testY==Y_pred3


# in the mcNemar test I need to compute the cross table of the two predictions, i.e. a table containing the number of
# correctly predictions in both, number of correct prediction in 1 but no in 2, etc
# what we actually need are the cases correct1/non-correct2 and non-correct1/correct2 (anti diagonal of the table) because
# these are the elements of the statistic for the test

# This will build the contingency table for the comparison of model 1 and 2, we use these values in an online calcualtor for mcnemar
print("Contingency table for model 1 and 2: \n",pd.crosstab(bool1,bool2))
print("Contingency table for model 1 and 3: \n",pd.crosstab(bool1,bool3))
print("Contingency table for model 2 and 3: \n",pd.crosstab(bool2,bool3))






# without baggin, works!!
# X=np.asarray(data_sets[1].iloc[:,:-1])
# Y=np.asarray(data_sets[1].iloc[:,-1])
# t1 = tree_grow(X,Y,nmin=20, minleaf=5,nfeat=8)
# y_pred=tree_pred(X, t1)
# print(confusion_matrix(Y, y_pred))


# clf = DecisionTreeClassifier()
# clf = clf.fit(X, Y)
# print(confusion_matrix(Y, clf.predict(X)))


#DotExporter(t1.node).to_picture('tree.png')
# for pre, fill, node in t1:
#     class_zero=str(len(node.label[node.label==0]))
#     class_one=str(len(node.label[node.label==1]))
#     print("%s%s %s %s %s" % (pre, node.name, class_zero, class_one, node.attribute_index))



# with bagging works but it doesn't seem to improve increasing the m
# X=np.asarray(data_sets[2].iloc[:,:-1])
# Y=np.asarray(data_sets[2].iloc[:,-1])
# t = tree_grow_b(X,Y,nmin=20, minleaf=5, nfeat=8, m=10)

# y_pred=tree_pred_b(t,X)
# print(confusion_matrix(Y, y_pred))

# clf = DecisionTreeClassifier(random_state=0)
# clf = clf.fit(X, Y)
# print(confusion_matrix(Y, clf.predict(X)))
# plot_tree(clf)
# plt.show()
