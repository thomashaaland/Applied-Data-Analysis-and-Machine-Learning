"""
##############################################################################################
## Can use following input arguments. If these are not used, program will run as standard   ##
## with n = 4000, minLambda=-7, maxLambda=-3, numLambdas=30, maxDegree=12, k=10,            ##
## randomSeed=2019                                                                          ##
## Usage: sys.argv = [n = number of points used in Frankefunction,                          ##
##                    minLambda = minimum Lambda used in search for optimal Lambda          ##
##                    maxLambda = maximum Lambda used in search for optimal Lambda          ##
##                    numLambdas = how dense should the lambda search be?                   ##
##                    maxDegree = What is the maximum degree in BiasVar plots,              ##
##                    k = number of folds in k-fold                                         ##
##                    randomSeed = True at default]                                         ##
##############################################################################################
"""
import numpy as np
import matplotlib.pyplot as plt
import oblig1_header as oh
import sklearn.linear_model as skl
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample
from sklearn.pipeline import make_pipeline
from matplotlib import cm
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D

n = 4000 # number of points in frankeFunction
minDegree = 0 # the minimum degree for bias variance 
maxDegree = 20 # the maximum degree for bias variancec
minLambda = -7 # minimum lambda to test for 
maxLambda = -3 # maximum lambda to test for
numLambdas = 30 # lambda resolution
randomSeed = True # Whether to have the progam run with a predetermined randomseed
k = 50

# Save and create a direcoty for all files in this file, will create a new folder one step
# down from where the file is located
path = os.path.dirname(os.path.abspath(__file__)) + "/output_oblig1_frankFunc_N"+str(n)+"/"
if not os.path.isdir(path):
    os.mkdir(path)

# Interprets additional arguments sent via command line
print(__doc__)
if len(sys.argv) > 1:
    if len(sys.argv) != 8:
        print("Error: You need to input all 6 variables above or none")
        sys.exit()
    for arg in [sys.argv[1], sys.argv[4], sys.argv[5], sys.argv[6]]:
        if type(eval(arg)) != int:
            print("Error: You need to input integers for n, numLambdas, maxDegree and k.")
            sys.exit()
    for arg in [sys.argv[2], sys.argv[3]]:
        if type(eval(arg)) != int and type(eval(arg)) != float:
            print("Error: You need to input integer or float for minLambda and maxLambda.")
            sys.exit()
    
    n = eval(sys.argv[1])
    minLambda = eval(sys.argv[2])
    maxLambda = eval(sys.argv[3])
    numLambdas = eval(sys.argv[4]) 
    maxDegree = eval(sys.argv[5])
    k = eval(sys.argv[6])
    randomSeed = eval(sys.argv[7])

if randomSeed == True:
    np.random.seed(2019) # Random seed to guarantee reproducibility
elif type(randomSeed) == int:
    np.random.seed(randomSeed)
else:
    print("Error: Randomseed needs to be type bool or int")
    sys.exit()

def center(a):
    return a - np.mean(a)

###############################################################
# Set up the grid, x and y are coordinates for frankeFunction #
###############################################################
x = np.random.rand(n).reshape(-1,1)
y = np.random.rand(n).reshape(-1,1)

f = oh.frankeFunction(x,y)
noise = np.random.normal(0, 0.1, f.shape) # Amount of noise to give frankeFunction
z =  f + noise

###############
# Test plot z #
###############

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.scatter3D(x, y, z)
plt.tight_layout()
plt.show()


# Set up all variables to be used later
degrees = np.arange(minDegree,maxDegree)
lambdas = np.logspace(minLambda,maxLambda,numLambdas)
##### Start the loop over many degrees of polynomial to produce Bias Variance Tradeoff ########

variancesOLS = np.zeros((lambdas.shape[0], degrees.shape[0]))
biasesOLS = np.zeros((lambdas.shape[0], degrees.shape[0]))
biasefsOLS = np.zeros((lambdas.shape[0], degrees.shape[0]))
errorsOLS = np.zeros((lambdas.shape[0], degrees.shape[0]))
errorsTrainOLS = np.zeros((lambdas.shape[0], degrees.shape[0]))
r2sOLS = np.zeros((lambdas.shape[0], degrees.shape[0]))

variancesRidge = np.zeros((lambdas.shape[0], degrees.shape[0]))
biasesRidge = np.zeros((lambdas.shape[0], degrees.shape[0]))
biasefsRidge = np.zeros((lambdas.shape[0], degrees.shape[0]))
errorsRidge = np.zeros((lambdas.shape[0], degrees.shape[0]))
r2sRidge = np.zeros((lambdas.shape[0], degrees.shape[0]))

variancesLasso = np.zeros((lambdas.shape[0], degrees.shape[0]))
biasesLasso = np.zeros((lambdas.shape[0], degrees.shape[0]))
biasefsLasso = np.zeros((lambdas.shape[0], degrees.shape[0]))
errorsLasso = np.zeros((lambdas.shape[0], degrees.shape[0]))
r2sLasso = np.zeros((lambdas.shape[0], degrees.shape[0]))

for nlam, _lambda in enumerate(np.array([0])):
    for degree in degrees:
        ######### KFold! #############
                         
        X = oh.create_X(x.ravel(), y.ravel(), degree, intercept=True)
        
        #k = 50
        kf = oh.k_fold(n_splits=k-1, shuffle=True)
                         
        zPredictsOLS = np.empty((int(z.shape[0]/k), k))
        #zTrainOLS = [] #np.empty( (int(z.shape[0]), k) )
        zTrainOLS = np.empty( (int(z.shape[0]), k) )
        zPredictsRidge = np.empty((int(z.shape[0]/k), k))
        zPredictsLasso = np.empty((int(z.shape[0]/k), k))
        zTests = np.empty((int(z.shape[0]/k), k))
        i = 0
    
        X_rest, X_test, x_rest, x_test, y_rest, y_test, z_rest, z_test, f_rest, f_test = train_test_split(X, x, y, z, f, test_size = int(z.shape[0]/k), shuffle=True)
        lambdaRidge, lambdaLasso = oh.findBestLambda(X_rest, z_rest, f_rest, k, lambdas, degree)
        zTests[:, i] = z_test.reshape(zTests.shape[0])
        kf.get_n_splits(X_rest)
        for train_index, test_index in kf.split():
            X_train, X_validation = X_rest[train_index], X_rest[test_index]
            x_train, x_validation = x_rest[train_index], x_rest[test_index]
            y_train, y_validation = y_rest[train_index], y_rest[test_index]
            z_train, z_validation = z_rest[train_index], z_rest[test_index]
            f_train, f_validation = f_rest[train_index], f_rest[test_index]

            # OLS, Finding the best lambda
            betaOLS = oh.linFit(X_train, z_train, model='OLS', _lambda = _lambda)
            zPredictsOLS[:,i] = (X_test @ betaOLS).reshape(-1)

            zTrainOLS[:,i] = (X @ betaOLS).reshape(-1)
            
            # Ridge, Finding the best lambda
            betaRidge = oh.linFit(X_train, z_train, model='Ridge', _lambda = lambdaRidge)
            zPredictsRidge[:,i] = (X_test @ betaRidge).reshape(-1)

            # Lasso, Finding the best lambda
            
            lasso = skl.Lasso(alpha = lambdaLasso, fit_intercept=True, max_iter=10**8, precompute=True)
            if X_train.shape[1] == 0:
                zPredictsLasso[:,i] = np.zeros(len(X_test))
            else:
                lasso.fit(X_train, z_train)
                zPredictsLasso[:,i] = lasso.predict(X_test).reshape(-1)
            
                      
            i += 1
            print(i, degree)

        ##########################################################################################################
        ## Finds the variance, bias, error and r2 score. Since the initial dimensions are [Ndatapoints, kfolds] ##
        ## we need to average over kfolds first before dealing with the Ndatapoints dimension                   ##
        ##########################################################################################################

        varianceOLS = oh.var(zPredictsOLS)
        biasOLS = oh.bias(z_test, zPredictsOLS)
        biasfOLS = oh.bias(f_test, zPredictsOLS)
        errorOLS = oh.mse(z_test, zPredictsOLS)
        errorTrainOLS = oh.mse(z,zTrainOLS)
        r2OLS = oh.R2_score(z_test, zPredictsOLS)
        
        variancesOLS[nlam][degree] = varianceOLS
        biasesOLS[nlam][degree] = biasOLS
        biasefsOLS[nlam][degree] = biasfOLS
        errorsOLS[nlam][degree] = errorOLS
        errorsTrainOLS[nlam][degree] = errorTrainOLS
        r2sOLS[nlam][degree] = r2OLS

        varianceRidge = oh.var(zPredictsRidge)
        biasRidge = oh.bias(z_test, zPredictsRidge)
        biasfRidge = oh.bias(f_test, zPredictsRidge)
        errorRidge = oh.mse(z_test, zPredictsRidge)
        r2Ridge = oh.R2_score(z_test, zPredictsRidge)
        
        variancesRidge[nlam][degree] = varianceRidge
        biasesRidge[nlam][degree] = biasRidge
        biasefsRidge[nlam][degree] = biasfRidge
        errorsRidge[nlam][degree] = errorRidge
        r2sRidge[nlam][degree] = r2Ridge
        
        varianceLasso = oh.var(zPredictsLasso)
        biasLasso = oh.bias(z_test, zPredictsLasso)
        biasfLasso = oh.bias(f_test, zPredictsLasso)
        errorLasso = oh.mse(z_test, zPredictsLasso)
        r2Lasso = oh.R2_score(z_test, zPredictsLasso)
        
        variancesLasso[nlam][degree] = varianceLasso
        biasesLasso[nlam][degree] = biasLasso
        biasefsLasso[nlam][degree] = biasfLasso
        errorsLasso[nlam][degree] = errorLasso
        r2sLasso[nlam][degree] = r2Lasso

#############################################################
## Plot the bias variane and r2 scores                     ##
#############################################################

fig = plt.figure()
plt.plot(np.mean(errorsOLS, axis=0), label="MSE on test set")
plt.plot(np.mean(errorsTrainOLS, axis=0), label="MSE on train set")
plt.title("MSE using train set and test set.")
plt.xlabel("Degree")
plt.ylabel("Error")
plt.legend()
plt.tight_layout()
plt.savefig(path + "trainTestTestOLS_N"+str(n)+".png")
plt.show()

fig = plt.figure()
plt.plot(np.mean(errorsOLS, axis=0), label="MSE")
plt.plot(np.mean(variancesOLS, axis=0), label="Variance")
plt.plot(np.mean(biasesOLS, axis=0), label="Bias")
plt.title("Bias Variance tradeoff for OLS")
plt.xlabel("Degree")
plt.ylabel("Error")
plt.legend()
plt.tight_layout()
plt.savefig(path + "biasVarianceOLS_N"+str(n)+".png")
plt.show()

fig = plt.figure()
plt.plot(np.mean(errorsRidge, axis=0), label="MSE")
plt.plot(np.mean(variancesRidge, axis=0), label="Variance")
plt.plot(np.mean(biasesRidge, axis=0), label="Bias")
plt.title("Bias Variance tradeoff for Ridge")#, lambda={:.7f}".format(lambdaRidge))
plt.xlabel("Degree")
plt.ylabel("Error")
plt.legend()
plt.tight_layout()
plt.savefig(path + "biasVarianceRidge_N"+str(n)+".png")
plt.show()

fig = plt.figure()
plt.plot(np.mean(errorsLasso, axis=0), label="MSE")
plt.plot(np.mean(variancesLasso, axis=0), label="Variance")
plt.plot(np.mean(biasesLasso, axis=0), label="Bias")
plt.title("Bias Variance tradeoff for Lasso")#, lambda={:.7f}".format(lambdaLasso))
plt.xlabel("Degree")
plt.ylabel("Error")
plt.legend()
plt.tight_layout()
plt.savefig(path + "biasVarianceLasso_N"+str(n)+".png")
plt.show()

fig = plt.figure()
plt.plot(np.arange(maxDegree-1)+1, np.mean(errorsOLS, axis=0)[1:], label="MSE")
plt.plot(np.arange(maxDegree-1)+1, np.mean(r2sOLS, axis=0)[1:], label="R2 score")
plt.title("R2 vs MSE for OLS")
plt.xlabel("Degree")
plt.ylabel("Error")
plt.legend()
plt.tight_layout()
plt.savefig(path + "mseR2OLS_N"+str(n)+".png")
plt.show()

fig = plt.figure()
plt.plot(np.arange(maxDegree-1)+1, np.mean(errorsRidge, axis=0)[1:], label="MSE")
plt.plot(np.arange(maxDegree-1)+1, np.mean(r2sRidge, axis=0)[1:], label="R2 score")
plt.title("R2 vs MSE for Ridge")
plt.xlabel("Degree")
plt.ylabel("Error")
plt.legend()
plt.tight_layout()
plt.savefig(path + "mseR2Ridge_N"+str(n)+".png")
plt.show()

fig = plt.figure()
plt.plot(np.arange(maxDegree-1)+1, np.mean(errorsLasso, axis=0)[1:], label="MSE")
plt.plot(np.arange(maxDegree-1)+1, np.mean(r2sLasso, axis=0)[1:], label="R2 score")
plt.title("R2 vs MSE for Lasso")
plt.xlabel("Degree")
plt.ylabel("Error")
plt.legend()
plt.tight_layout()
plt.savefig(path + "mseR2Lasso_N"+str(n)+".png")
plt.show()

