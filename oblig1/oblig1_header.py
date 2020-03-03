import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed
from mpl_toolkits import mplot3d
import sys

import scipy.linalg as scl

import sklearn.linear_model as skl
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split, KFold

# Class for k-fold algorithm, made to look similar to sklearn for the purpose of returning
# an iterable object with shuffled folds. The folds themselves are sorted.
class k_fold:
    def __init__(self, n_splits, shuffle = True):
        self.k = n_splits
        self.shuffle = shuffle

    def get_n_splits(self, X):
        finshape = X.shape[0]/self.k
        self.totN = len(X)
        self.batchSize = int(self.totN/self.k)
        if self.shuffle == True:
            self.p = np.random.permutation(self.totN)
        elif self.shuffle == False:
            self.p = np.arange(self.totN)
        else:
            print("Error: You need to set shuffle to True or False")
            exit()
            
    def split(self):
        foldedp = np.empty(self.k)
        foldedp = self.p[: self.batchSize*self.k].reshape(self.k, int(self.batchSize))

        # Must return k different sets, where one test set is returned as test,
        # while the rest is combined into train
        return_object = []
        for i, sets in enumerate(foldedp):
            return_object.append( (np.sort(np.delete(foldedp, i, 0).reshape(-1)), np.sort(sets)) )
        return return_object

def findBestLambda(X, z, f, k, lambdas, degree, bestLambda = True):
    lamErrOLS = np.zeros(lambdas.shape[0])
    lamErrRidge = np.zeros(lambdas.shape[0])
    lamErrLasso = np.zeros(lambdas.shape[0])

    numBetas = X.shape[1]
    numLambdas = len(lambdas)
    betasOLS = np.empty((numLambdas, numBetas))
    betasRidge = np.empty((numLambdas, numBetas))
    betasLasso = np.empty((numLambdas, numBetas))

    betasSigmaOLS = np.empty((numLambdas, numBetas))
    betasSigmaRidge = np.empty((numLambdas, numBetas))
    betasSigmaLasso = np.empty((numLambdas, numBetas))

    ##################################
    # Start of K-fold algorithm      #
    ##################################

    # Generally k = 5 is a good choice
    #k = 50
    kf = k_fold(n_splits=k, shuffle=True)
    kf.get_n_splits(X)

    for nlam, _lambda in enumerate(lambdas):
        ######### KFold! #############

        errorsOLS = np.empty(k)
        zPredictsOLS = np.empty((int(z.shape[0]/k)))
        betasOLSTemp = np.empty((k, numBetas))
        betasSigmaOLSTemp = np.empty((k, numBetas))

        errorsRidge = np.empty(k)
        zPredictsRidge = np.empty((int(z.shape[0]/k)))
        betasRidgeTemp = np.empty((k, numBetas))
        betasSigmaRidgeTemp = np.empty((k, numBetas))

        errorsLasso = np.empty(k)
        zPredictsLasso = np.empty((int(z.shape[0]/k)))
        betasLassoTemp = np.empty((k, numBetas))
        betasSigmaLassoTemp = np.empty((k, numBetas))

        #zTests = np.empty((int(z.shape[0]/k)))
        i = 0
        X_rest, z_rest, f_rest = X, z, f
        for train_index, test_index in kf.split():
            X_train, X_validation = X_rest[train_index], X_rest[test_index]
            z_train, z_validation = z_rest[train_index], z_rest[test_index]
            f_train, f_validation = f_rest[train_index], f_rest[test_index]

            # OLS, Finding the best lambda
            betaOLS = linFit(X_train, z_train, model='OLS', _lambda = _lambda)
            betasOLSTemp[i] = betaOLS.reshape(-1)
            zPredictsOLS = (X_validation @ betaOLS)
            errorsOLS[i] = np.mean((z_validation - zPredictsOLS)**2)
            sigmaOLSSq = 1/(X_validation.shape[0] - 0*X_validation.shape[1]) * np.sum((z_validation - zPredictsOLS)**2)
            sigmaBetaOLSSq = sigmaOLSSq * np.diag(np.linalg.pinv(X_validation.T @ X_validation))
            betasSigmaOLSTemp[i] = np.sqrt(sigmaBetaOLSSq)


            # Ridge, Finding the best lambda
            betaRidge = linFit(X_train, z_train, model='Ridge', _lambda = _lambda)
            betasRidgeTemp[i] = betaRidge.reshape(-1)
            zPredictsRidge = (X_validation @ betaRidge)
            errorsRidge[i] = np.mean((z_validation - zPredictsRidge)**2)
            sigmaRidgeSq = 1/(X_validation.shape[0] - 0*X_validation.shape[1]) * np.sum((z_validation - zPredictsRidge)**2)
            XInvRidge = np.linalg.pinv(X_validation.T @ X_validation + _lambda * np.eye(len(betaRidge)))
            sigmaBetaRidgeSq = sigmaRidgeSq * np.diag(XInvRidge @ X_validation.T @ X_validation @ XInvRidge.T)
            betasSigmaRidgeTemp[i] = np.sqrt(sigmaBetaRidgeSq)

            # Lasso, Finding the best lambda

            lasso = skl.Lasso(alpha = _lambda, fit_intercept=True, max_iter=10**9, precompute=True).fit(X_train, z_train)
            betaLasso = lasso.coef_
            betasLassoTemp[i] = betaLasso.reshape(-1)
            zPredictsLasso = lasso.predict(X_validation)
            errorsLasso[i] = mean_squared_error(z_validation, zPredictsLasso)

            i += 1
            #print(i, nlam)

        betasOLS[nlam] = np.mean(betasOLSTemp,axis=0)
        betasRidge[nlam] = np.mean(betasRidgeTemp,axis=0)
        betasLasso[nlam] = np.mean(betasLassoTemp,axis=0)
        betasSigmaOLS[nlam] = np.mean(betasSigmaOLSTemp, axis=0)
        betasSigmaRidge[nlam] = np.mean(betasSigmaRidgeTemp, axis = 0)
        lamErrOLS[nlam] = min(errorsOLS)
        lamErrRidge[nlam] = min(errorsRidge)
        lamErrLasso[nlam] = min(errorsLasso)
    if bestLambda == False:
        # in this case, return the betas and errors
        return betasOLS, betasRidge, betasLasso, betasSigmaOLS, betasSigmaRidge, lamErrOLS, lamErrRidge, lamErrLasso
    else:
        # In this case just return the best value for Ridge and Lasso in that order
        minimumRidge = np.min(np.array([lambdas[lamErrRidge == min(lamErrRidge)]]))
        minimumLasso = np.min(np.array([lambdas[lamErrLasso == min(lamErrLasso)]]))
        return minimumRidge, minimumLasso

# The expectation number:
def E(z):
    return np.sum(z)/len(z)

# The mean squared error:
def mse( z, zTilde, axis=1, keepdims=True ):
    if len(zTilde.shape) == 1:
        return E( (z - ztilde)**2 )
    else:
        return np.mean( np.mean( ( z - zTilde)**2, axis=axis, keepdims=keepdims ) )

# The variance:
def var( z, axis=1, keepdims=True):
    if len(z.shape) == 1:
        return E( z**2 ) - E( z )**2
    else:
        return np.mean( np.var(z, axis=axis, keepdims=keepdims) )

# The bias:
def bias( z, zTilde, axis=1, keepdims=True):
    if len(zTilde.shape) == 1:
        return E( zTilde ) - z
    else:
        return np.mean( ( z - np.mean( zTilde, axis=axis, keepdims=keepdims ) )**2 )

def R2_score(z, zTilde, axis=1, keepdims=True ):
    if len(zTilde.shape) == 1:
        SS1 = np.sum( (z - zTilde)**2 )
        SS2 = np.sum( (z - E(z))**2 )
        return 1 - SS1/SS2
    else:
        SSres = np.sum( ( np.mean( z - zTilde, axis=axis, keepdims=keepdims) )**2 )
        SStot = np.sum( ( np.mean( zTilde - np.mean( np.mean( zTilde, axis=axis, keepdims=keepdims) ), axis=axis, keepdims=keepdims ) )**2 )
        return 1 - SSres / SStot

def frankeFunction(x, y):
    b1 = -((9*x - 2)**2)/4 - ((9*y - 2)**2)/4
    b2 = -((9*x + 1)**2)/49 - ((9*y + 1)**2)/10
    b3 = -((9*x - 7)**2)/4 - ((9*y - 3)**2)/4
    b4 = -((9*x - 4)**2) - ((9*y - 7)**2)
    
    a1 = 3/4 * np.exp(b1)
    a2 = 3/4 * np.exp(b2)
    a3 = 1/2 * np.exp(b3)
    a4 = -1/5 * np.exp(b4)
    return a1 + a2 + a3 + a4# + 0.25 * np.random.randn(len(x),)

def create_X(x, y, order, intercept=True):
    '''
    Creates design matrix for a polynomial up to degree of 'order'
    '''
    n_terms = int((order+2)*(order+1)/2) #number of combinations of powers in the polynomial
    if hasattr(x, "__len__"):
        X = np.zeros((len(x), n_terms))
    else:
        X = np.zeros((1, n_terms))
    n = 0 #counter for which column we are at in the design matrix
    for j in range(0, order+1 ):
        for i in range(j + 1):
            X[:,n] = x**(j - i)*y**i
            n += 1
    if intercept == True:
        return X
    else:
        return np.delete(X, 0, 1)


def linFit(X, z, model = 'OLS', _lambda = 0):
    dim = (X.T @ X).shape[0]

    if model == 'OLS':
        invXTX = SVDinv(X.T @ X)
        beta = invXTX @ X.T @ z
    elif model == 'Ridge':
        invXTX = SVDinv((X.T @ X) + (_lambda * np.eye(dim)))
        beta = invXTX @ X.T @ z
    elif model == 'Lasso':
        if _lambda == 0:
            print("In Lasso regression, Lambda needs to be larger than 0")
            exit()
        lasso = skl.Lasso(alpha = _lambda, fit_intercept=True, max_iter=10**8)
        lasso.fit(X, z)
        beta = lasso.coef_
    return beta

# SVD inversion
def SVDinv(A):
    ''' Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.
    '''
    U, s, VT = np.linalg.svd(A)
    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=s[i]
    UT = np.transpose(U); V = np.transpose(VT); invD = np.linalg.inv(D)
    return np.matmul(V,np.matmul(invD,UT))
