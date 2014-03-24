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


def elasticnet_coordesc(XtX, XtY, YtY, mylambda=0, myalpha=0,
                        tol=1e-8, num_iter=np.inf):
    last_score = np.inf
    beta_vec = np.zeros((XtX.shape[0], 1))
    index = 0
    while True:
        new_score = elasticnet_obj(beta_vec, XtX, XtY, YtY, mylambda, myalpha)
        if last_score - new_score < tol or index > num_iter:
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


def kfold_split(X, y, K=5):
    X_list = []
    y_list = []
    n = y.shape[0]
    for k in range(K):
        start_idx = int(k * n / K)
        end_idx = int((k + 1) * n / K)
        X_list.append(X[start_idx:end_idx, :])
        y_list.append(y[start_idx:end_idx, :])
    return X_list, y_list


N = 1000
K = 20
sig = np.random.randn(N, 1)
w = [(K - k - 1) / (0.5 * K * (K-1)) for k in range(K)]
X = 2 * np.random.randn(N, K) + w * sig
y = np.random.randn(N, 1) + sig

linear = linear_model.LinearRegression(fit_intercept=False)
linear.fit(X, y)
print linear.coef_
print linear.score(X, y)

'''using solver'''
XtX = np.dot(X.T, X)
XtY = np.dot(X.T, y)
YtY = np.dot(y.T, y)
beta = np.linalg.solve(XtX, XtY)
print beta.T
print elasticnet_obj(beta, XtX, XtY, YtY)
'''using generic elastic net coor desc'''
XtX = np.dot(X.T, X)
XtY = np.dot(X.T, y)
YtY = np.dot(y.T, y)
beta_vec = elasticnet_coordesc(XtX, XtY, YtY)
print beta_vec.T
print elasticnet_obj(beta_vec, XtX, XtY, YtY)

mylambda = 0
myalpha = 1
num_fold = 5
X_list, y_list = kfold_split(X, y, num_fold)
XtX_list = [np.dot(X_i.T, X_i) for X_i in X_list]
XtY_list = [np.dot(X_i.T, y_i) for X_i, y_i in zip(X_list, y_list)]
YtY_list = [np.dot(y_i.T, y_i) for y_i in y_list]

XtX_all = np.dot(X.T, X)
XtY_all = np.dot(X.T, y)
YtY_all = np.dot(y.T, y)


lambda_list = [10 ** (lam - 6) for lam in range(13)]
for mylambda in lambda_list:
    cv_score = 0
    for XtX_i, XtY_i, YtY_i in zip(XtX_list, XtY_list, YtY_list):
        beta_cv = elasticnet_coordesc(XtX_all - XtX_i, XtY_all - XtY_i,
                                      YtY_all - YtY_i, mylambda, myalpha)
        #beta_vec = np.linalg.solve(XtX_all - XtX_i, XtY_all - XtY_i)
        #print beta_vec.T
        cv_i = elasticnet_obj(beta_cv, XtX_i, XtY_i, YtY_i, mylambda, myalpha)
        cv_score += cv_i
    beta_all = elasticnet_coordesc(XtX_all, XtY_all, YtY_all,
                                   mylambda, myalpha)
    in_score = elasticnet_obj(beta_all, XtX_all, XtY_all, YtY_all,
                              mylambda, myalpha)
    print mylambda, cv_score, in_score


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






































