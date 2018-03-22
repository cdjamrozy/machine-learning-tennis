#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 21:14:07 2017

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
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

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

# Perform hierarchical/agglomerative clustering on data matrix
Method = 'complete'
Metric = 'euclidean'

Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Y = X - np.ones((N,1))*X.mean(0)
U,S,V = svd(Y,full_matrices=False)
V = V.T
rho = (S*S) / (S*S).sum()

Maxclust = 9
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
figure(1, figsize=(14,9))
clusterplot((Y*V)[:, :2], cls.reshape(cls.shape[0],1), y=y)

# Display dendrogram
max_display_levels=50
figure(2,figsize=(10,4))
dendrogram(Z, truncate_mode='level', p=max_display_levels)

show()