import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed
from mpl_toolkits import mplot3d

import sklearn.linear_model as skl
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

import oblig1_header as oh


# Make and plot the Frankefunction as a preview
x = np.arange(0,1,0.05)
y = x
x, y = np.meshgrid(x,y)

z_preview = oh.frankeFunction(y,x) + 0.1 * np.random.randn(20,20)

fig = plt.figure()
ax = fig.add_subplot(3,2,1,projection='3d')
ax.title.set_text("FrankeFunction")
surf = ax.plot_surface(x,y,z_preview, cmap=cm.coolwarm,
                       linewidth = 0, antialiased=False)

ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#fig.colorbar(surf, shrink=0.5, aspect=5)


n = 1000
x_ = np.random.rand(n)
y_ = np.random.rand(n)

z = oh.frankeFunction(x_,y_) + 0.1*np.random.randn(n)

# Set up the design matrix

MSE = []
R2_score = []

for grad in range(1,6):
    X = oh.create_X(x_, y_, grad)
    invXTX = np.linalg.inv(X.T @ X) # Need this anyway
    #beta = invXTX @ X.T @ z
    beta = oh.linFit(X, z)
    ztilde = X @ beta

    MSE.append(oh.mse(z, ztilde))
    R2_score.append(oh.R2_score(z, ztilde))
    #sigma = np.sqrt(np.var(ztilde))
    #print("Sigma numpy: ", sigma)
    sigma = np.sqrt(1/(X.shape[0] - X.shape[1] - 1) * np.sum((z - ztilde)**2))
    #print("Sigma self: ", sigma)
    
    betaSigma = np.zeros(len(beta))
    relative = betaSigma
    betaConf = np.zeros((len(beta),2))
    for i in range(len(beta)):
        #betaSigma[i] = sigma * np.sqrt(np.sqrt(invXTX[i][i]))
        betaSigma[i] = sigma * np.sqrt(invXTX[i][i])
        
        # Confidence interval
        #betaConf[i][0] = beta[i] - 1.96*betaSigma[i]/np.sqrt(len(beta))
        #betaConf[i][1] = beta[i] + 1.96*betaSigma[i]/np.sqrt(len(beta))


        betaConf[i][0] = beta[i] - 2*betaSigma[i]
        betaConf[i][1] = beta[i] + 2*betaSigma[i]

        relative[i] = (2*betaSigma[i])/beta[i]
	#confidenceInterval = []
        #confidenceInterval_start = np.sqrt(np.mean(betaSigma[i])) - 2*sigma
        #confidenceInterval_end = np.sqrt(np.mean(betaSigma[i])) + 2*sigma
        #confidenceInterval.append([confidenceInterval_start, confidenceInterval_end])
    
    print("Grad: {}".format(grad))
    print("Mean Square Error: {:.6f}".format(MSE[grad-1]))
    print("R2 score: {:.6f}".format(R2_score[grad-1]))
    print("BetaSigma: ")
    print(betaSigma)
    print("ConfidenceInterval: ")
    print(betaConf)
    print("Beta: ")
    print(beta)
    print("Relative confidense: ")
    print(relative)
    
    
    #plt.subplot(int(str(32) + str((grad+1))))
    #ax = plt.axes(projection='3d')
    #ax.scatter3D(x_,y_,z)
    ax = fig.add_subplot(3,2,grad+1, projection='3d')
    ax.scatter3D(y_,x_,ztilde, marker = '.')
    ax.title.set_text("Linearfit of degree " + str(grad))
    #plt.legend(["Linearfit of degree " + str(grad)])
    #plt.legend(["Frankefunction", "LinearFit of degree " + str(grad)])
fig.suptitle("OLS fit to FrankeFunction")
plt.tight_layout()
plt.show()


plt.plot(MSE)
plt.plot(R2_score)
plt.show()


