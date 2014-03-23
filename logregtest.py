# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 18:34:54 2014

@author: Ken
"""
import numpy as np
from sklearn import linear_model


def soft_thresh(x, thresh):
    if thresh < x:
        return x - thresh
    elif x < -thresh:
        return x + thresh
    else:
        return 0


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def elasticnet_obj(beta, XtX, XtY, YtY, mylambda=0, myalpha=0):
    sq_err = YtY - 2 * np.dot(beta.T, XtY) + np.dot(np.dot(beta.T, XtX), beta)
    l1_penalty = mylambda * myalpha * sum(np.abs(beta))
    l2_penalty = mylambda * (1 - myalpha) * np.dot(beta.T, beta)
    return 0.5 * sq_err + l1_penalty + l2_penalty


def elasticnet_coordesc(XtX, XtY, YtY, mylambda=0, myalpha=0, tol=1e-8):
    last_score = np.inf
    beta_vec = np.zeros((XtX.shape[0], 1))
    index = 0
    while True:
        new_score = elasticnet_obj(beta_vec, XtX, XtY, YtY, mylambda, myalpha)
        if last_score - new_score < tol:
            break
        else:
            last_score = new_score
        for i, beta_i in enumerate(beta_vec):
            not_i = [j for j in xrange(len(beta_vec)) if j != i]
            numer = XtY[i, 0] - np.dot(XtX[i, not_i].T, beta_vec[not_i, 0])
            denom = XtX[i, i] + mylambda * (1 - myalpha)
            thresh = mylambda * myalpha / denom
            beta_vec[i, 0] = soft_thresh(numer / denom, thresh)
        #print index, last_score
        index += 1
    return beta_vec


N = 100000
K = 50
sig = np.random.randn(N, 1)
X = np.random.randn(N, K) + sig
y = np.random.randn(N, 1) + sig
XtX = np.dot(X.T, X)
XtY = np.dot(X.T, y)
YtY = np.dot(y.T, y)
sumY = np.sum(y)
n = y.shape[0]

linear = linear_model.LinearRegression(fit_intercept=False)
%time linear.fit(X, y)
print linear.coef_
print linear.score(X, y)
print elasticnet_obj(linear.coef_.T, XtX, XtY, YtY)

%time beta = np.linalg.solve(XtX, XtY)
print beta.T
print elasticnet_obj(beta, XtX, XtY, YtY)

%time beta_vec = elasticnet_coordesc(XtX, XtY, YtY)
print beta_vec.T
print elasticnet_obj(beta_vec, XtX, XtY, YtY)

expVar = n * np.dot(beta.T, XtY) - sumY ** 2
totVar = n * YtY - sumY ** 2
print expVar / totVar

''' Classification '''

N = 10000
K = 20
y = np.random.randint(0, 2, N).reshape(N, 1)
X = np.random.randn(N, K) + (y - 0.5) / K

logit = linear_model.LogisticRegression(fit_intercept=False)
print N
%time logit.fit(X, y[:, 0])
print 'SciKit Learn'
print logit.coef_
P = sigmoid(np.dot(X, logit.coef_.T))
W = np.sqrt(P * (1 - P))
Z = np.dot(X, beta) + (y - P) / (P * (1 - P))
WX = W * X
WZ = W * Z
XtWX = np.dot(WX.T, WX)
XtWZ = np.dot(WX.T, WZ)
ZtWZ = np.dot(WZ.T, WZ)
ols_score = elasticnet_obj(logit.coef_.T, XtWX, XtWZ, ZtWZ)
logl = np.dot(np.dot(logit.coef_, X.T), y) - np.sum(log(1 - P))
print ols_score, logl

W = np.ones((N, 1))
beta = np.zeros((K, 1))
for ii in range(10):
    Xb = np.dot(X, beta)
    P = sigmoid(Xb)
    W = np.sqrt(P * (1 - P))
    Z = Xb + (y - P) / (P * (1 - P))
    ## can optimize WX a little more

    WX = W * X
    WZ = W * Z
    ## Aggregate these across days for MapReduce
    XtWX = np.dot(WX.T, WX)
    XtWZ = np.dot(WX.T, WZ)
    ZtWZ = np.dot(WZ.T, WZ)

    ## Iterative inside Reduce function for convergence
    ## Additive because of OLS properties
    #beta = np.linalg.solve(XtWX, XtWZ)
    beta = elasticnet_coordesc(XtWX, XtWZ, ZtWZ, mylambda=0, myalpha=0, tol=1e-8)
smart
    ols_score = elasticnet_obj(beta, XtWX, XtWZ, ZtWZ, mylambda=0, myalpha=0)  # Approx score
    logl = np.dot(np.dot(beta.T, X.T), y) - np.sum(log(1 - P))  # Exact score
    print ii, ols_score, logl
print beta.T






































