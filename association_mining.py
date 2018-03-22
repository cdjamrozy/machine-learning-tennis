#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 13:22:04 2017

@author: cdjamrozy
"""


import numpy as np
import xlrd
from scipy import stats
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, subplot, clim
from scipy.linalg import svd
import sklearn.linear_model as lm
from sklearn.mixture import GaussianMixture
from sklearn import model_selection
from toolbox_02450 import clusterplot, clusterval
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from similarity import binarize
from writeapriorifile import WriteAprioriFile

# Load xls sheet with data
doc = xlrd.open_workbook('GrandSlamMatchStats.xls').sheet_by_index(0)

# Extract attribute names (1st row, column 4 to 12)
attributeNames = doc.row_values(1, 2, 27)

# Extract class names to python list,
# then encode with integers (dict)
classLabels = doc.col_values(27, 2, 345)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(5)))

# Extract vector y, convert to NumPy matrix and transpose
y = np.mat([classDict[value] for value in classLabels]).T

# Preallocate memory, then extract excel data to matrix X
X = np.mat(np.empty((343, 25)))
for i, col_id in enumerate(range(2, 27)):
    X[:, i] = np.mat(doc.col_values(col_id, 2, 345)).T

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

#Binarize the data
X = binarize(X)
X1 = np.mat(np.empty((N, M*2)))
for i in range(0, M):
    X1[:, 2*i] = X[:, i]
    X1[:, 2*i+1] = (X[:, i]==0)
    
X = X1

#Create apiori file
WriteAprioriFile(X)

#Run Apriori algorithm (from run_apriori.py)