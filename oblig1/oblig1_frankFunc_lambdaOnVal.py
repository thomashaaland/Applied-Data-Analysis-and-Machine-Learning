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

degree = 5 # This is used for finding the mimimum lambda. 5 is a good amount,
           # since anything greater slows the program down significantly due to Lasso
X = oh.create_X(x.ravel(), y.ravel(), degree, intercept=True) # Design matrix to be used when finding mimum lambda

betasOLS, betasRidge, betasLasso, betasSigmaOLS, betasSigmaRidge, lamErrOLS, lamErrRidge, lamErrLasso = oh.findBestLambda(X, z, f, k, lambdas, degree, bestLambda=False)


############################
# Write to file the errors #
############################
write2OLS = open(path + "betasOLS_N"+str(n)+".txt", "w+")
# FORMAT: lambda -> beta0 -> beta1 -> beta...
write2OLS.write("lambdas\tbetas\t\t\tsigmaBeta\t\tconfidence interval\n")
for i in range(betasOLS.shape[0]):
    lineOLS = str(lambdas[i])+"\t"
    for j in range(betasOLS.shape[1]):
        lineOLS += str(betasOLS[i][j]) + "\t" + str(betasSigmaOLS[i][j]) + "\t" + str(betasOLS[i][j] - 2*betasSigmaOLS[i][j]) + "\t" + str(betasOLS[i][j] + 2*betasSigmaOLS[i][j]) + "\n\t"
    lineOLS += "\n"

write2OLS.write(lineOLS)
write2OLS.close()

write2Ridge = open(path + "betasRidge_N"+str(n)+".txt", "w+")
# FORMAT: lambda -> beta0 -> beta1 -> beta...
write2Ridge.write("lambdas\tbetas\tsigmaBeta\tconfidence interval")
for i in range(betasRidge.shape[0]):
    lineRidge = str(lambdas[i])+"\t"
    for j in range(betasRidge.shape[1]):
        lineRidge += str(betasRidge[i][j]) + "\t" + str(betasSigmaRidge[i][j]) + "\t" + str(betasRidge[i][j] - 2*betasSigmaRidge[i][j]) + "\t" + str(betasRidge[i][j] + 2*betasSigmaRidge[i][j]) + "\n\t"
    lineRidge += "\n"

write2Ridge.write(lineRidge)
write2Ridge.close()

write2Lasso = open(path + "betasLasso_N"+str(n)+".txt", "w+")
# FORMAT: lambda -> beta0 -> beta1 -> beta...
write2Lasso.write("lambdas\tbetas")
for i in range(betasLasso.shape[0]):
    lineLasso = str(lambdas[i])+"\t"
    for j in range(betasLasso.shape[1]):
        lineLasso += str(betasLasso[i][j]) + "\n\t"
    lineLasso += "\n"

write2Lasso.write(lineRidge)
write2Lasso.close()

##########################################
## Plot beta with errorbar              ##
##########################################

plt.errorbar(np.arange(betasOLS.shape[1]), betasOLS[0], yerr = 2*betasSigmaOLS[0], fmt = '.')
plt.title("Beta OLS")
plt.xlabel(r"$\beta_i$")
plt.ylabel("Beta error")
plt.tight_layout()
plt.savefig(path + "betasOLS_N"+str(n)+".png")
plt.show()

for i in range(betasRidge.shape[1]):
    error = 2*betasSigmaRidge.T[i]
    plt.plot(lambdas, betasRidge.T[i])
    plt.fill_between(lambdas, betasRidge.T[i]-error, betasRidge.T[i]+error, alpha = 0.2)
plt.title("Ridge varying with lambda")
plt.xlabel("Lambdas")
plt.ylabel("Beta error")
plt.xscale("log")
plt.tight_layout()
plt.savefig(path + "betasRidge_N"+str(n)+".png")
plt.show()

plt.plot(lambdas, betasLasso)
plt.title("Lasso varying with lambda")
plt.xlabel("Lambdas")
plt.ylabel("Beta error")
plt.tight_layout()
plt.savefig(path + "betasLasso_N"+str(n)+".png")
#plt.xscale("log")
plt.show()

    
plt.loglog(lambdas, lamErrOLS,label="OLS")
plt.loglog(lambdas, lamErrRidge,label="Ridge")
plt.loglog(lambdas, lamErrLasso,label="Lasso")
plt.xlabel("Lambda")
plt.legend()
plt.tight_layout()
plt.savefig(path + "bestLambdaCompare_N"+str(n)+".png")
plt.show()

lambdaRidge = np.array([lambdas[lamErrRidge == min(lamErrRidge)]])
lambdaLasso = np.array([lambdas[lamErrLasso == min(lamErrLasso)]])

