#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 20:24:32 2017

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

# Range of K's to try
KRange = range(1,11)
T = len(KRange)

covar_type = 'diag'     # you can try out 'diag' as well
reps = 10               # number of fits with different initalizations, best result will be kept

# Allocate variables
BIC = np.zeros((T,))
AIC = np.zeros((T,))
CVE = np.zeros((T,))

# K-fold crossvalidation
CV = model_selection.KFold(n_splits=10,shuffle=True)

for t,K in enumerate(KRange):
        print('Fitting model for K={0}'.format(K))

        # Fit Gaussian mixture model
        gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X)

        # Get BIC and AIC
        BIC[t,] = gmm.bic(X)
        AIC[t,] = gmm.aic(X)

        # For each crossvalidation fold
        for train_index, test_index in CV.split(X):

            # extract training and test set for current CV fold
            X_train = X[train_index]
            X_test = X[test_index]

            # Fit Gaussian mixture model to X_train
            gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)

            # compute negative log likelihood of X_test
            CVE[t] += -gmm.score_samples(X_test).sum()
            
figure(1); 
plot(KRange, BIC,'-*b')
plot(KRange, AIC,'-xr')
plot(KRange, 2*CVE,'-ok')
legend(['BIC', 'AIC', 'Crossvalidation'])
xlabel('K')
show()


#Project onto primary principal components
Y = X - np.ones((N,1))*X.mean(0)
U,S,V = svd(Y,full_matrices=False)
V = V.T
rho = (S*S) / (S*S).sum()

X = (Y * V)[:, :2]
M = 2

K = np.argmin(CVE)+1

#Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X)
cls = gmm.predict(X)

# extract cluster labels
cds = gmm.means_        
# extract cluster centroids (means of gaussians)
covs = gmm.covariances_
# extract cluster shapes (covariances of gaussians)
if covar_type == 'diag':    
    new_covs = np.zeros([K,M,M])    

count = 0    
for elem in covs:        
    temp_m = np.zeros([M,M])        
    for i in range(len(elem)):            
        temp_m[i][i] = elem[i]        
    
    new_covs[count] = temp_m        
    count += 1
        
covs = new_covs
# Plot results:
figure(figsize=(14,9))
clusterplot(X, clusterid=cls, centroids=cds, y=y, covars=covs)
show()