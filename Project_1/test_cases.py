from numpy import genfromtxt

import numpy as np

import pandas as pd

import time



# Basic test on credit data. Prediction should be perfect.



credit_data = genfromtxt('C:/Data Mining/credit.txt', delimiter=',', skip_header=True)
credit_x = credit_data[:,0:5]
credit_y = credit_data[:,5]
credit_tree = tree_grow(credit_x,credit_y,2,1,5)
credit_pred = tree_pred(credit_x, credit_tree)
pd.crosstab(credit_y,credit_pred)

# Single tree on pima data
pima_data = genfromtxt('C:/Data Mining/pima.txt', delimiter=',')
pima_x = pima_data[:,0:8]
pima_y = pima_data[:,8]
pima_tree = tree_grow(pima_x,pima_y,20,5,8)
pima_pred = tree_pred(pima_x,pima_tree)
pd.crosstab(pima_y,pima_pred)

# Function for testing single tree
def single_test(x,y,nmin,minleaf,nfeat,n):
  acc = np.zeros(n)
  for i in range(0, n):
    tr = tree_grow(x,y,nmin,minleaf,nfeat)
    pred = tree_pred(x,tr)
    acc[i] = sum(pred == y)/len(y)
  return [np.mean(acc), np.std(acc)]

# Function for testing bagging/random forest
def rf_test(x,y,nmin,minleaf,nfeat,m,n):
  acc = np.zeros(n)

  for i in range(0,n):
    tr_list = tree_grow_b(x,y,nmin,minleaf,nfeat,m)
    pred = tree_pred_b(x,tr_list)
    acc[i] = sum(pred == y)/len(y)
  return [np.mean(acc), np.std(acc)]

# Compute average and standard deviation of accuracy for single tree

single_test(pima_x,pima_y,20,5,2,25)
single_test(pima_x,pima_y,20,5,8,25)

# Compute average and standard deviation of accuracy for bagging/random forest
rf_test(pima_x,pima_y,20,5,2,25,25)
rf_test(pima_x,pima_y,20,5,8,25,25)

