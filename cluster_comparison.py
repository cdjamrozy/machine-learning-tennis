#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 21:45:51 2017

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

cov_type = 'diag'     # you can try out 'diag' as well
reps = 10               # number of fits with different initalizations, best result will be kept


#Set K to value earlier found through cross validation
K = 9

#Fit GMM to data
gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps).fit(X)
cls = gmm.predict(X)

#Compute error with respect to actual classes
rand_gmm, Jaccard_gmm, NMI_gmm = clusterval(np.asarray(y).ravel(), cls)

# Perform hierarchical/agglomerative clustering on data matrix
Method = 'complete'
Metric = 'euclidean'

Z = linkage(X, method=Method, metric=Metric)
cls = fcluster(Z, criterion='maxclust', t=K)

rand_h, Jaccard_h, NMI_h = clusterval(np.asarray(y).ravel(), cls)

print("GMM:")
print("rand: {0}".format(rand_gmm))
print("Jaccard: {0}".format(Jaccard_gmm))
print("NMI: {0}".format(NMI_gmm))

print()
print("Hierarchical:")
print("rand: {0}".format(rand_h))
print("Jaccard: {0}".format(Jaccard_h))
print("NMI: {0}".format(NMI_h))

