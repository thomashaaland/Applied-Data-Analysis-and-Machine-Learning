import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from IPython.display import display
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#Help function for plotting
def plot_data(x, y, line, xlabel='', ylabel='', name=''):
    plt.plot(x,y,line,label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

# Self maded data to regress
x = np.random.rand(100)
y = 5*x*x+0.1*np.random.randn(100)

#display data
plot_data(x,y,'s',name="Initial data")

# Make the design matrix:
X = np.zeros((len(x), 3))
X[:,0] = 1
X[:,1] = x
X[:,2] = x**2

# Finding beta
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
ytilde = X @ beta

x_ = np.linspace(0,1,101)
# Make the design matrix for round2:
X_ = np.zeros((len(x_), 3))
X_[:,0] = 1
X_[:,1] = x_
X_[:,2] = x_**2

y_tilde = X_ @ beta
plot_data(x_, y_tilde, '-', name='LinalgFit')
plt.legend()
plt.show()

print("Mean squared error: {0:.3f}".format(mean_squared_error(y, ytilde)))
print("Variance score: {0:.3f}".format(r2_score(y, ytilde)))

# Introduce Ridge regression:
# Finding Ridge beta
lamb = 0.01
betaRidge = np.linalg.inv(X.T.dot(X) + lamb*np.eye(3)).dot(X.T).dot(y)
ytildeRidge = X @ (betaRidge)

plot_data(x,y,'s',name="Initial data")
plot_data(x, ytildeRidge,'o', name="Ridge")
plt.legend()
plt.show()
