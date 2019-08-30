import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl

# Help function for plotting
def plot_data(x, y, line='-', xlabel='', ylabel='', name=''):
    plt.plot(x,y,line,label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

# self made data to regress
x = np.random.rand(100,1)
y = 5*x*x+0.1*np.random.randn(100,1)

# display data
plot_data(x,y,'*',name="Initial data")


# Make the design matrix:
X = np.zeros((len(x), 3))
X[:,0] = 1
X[:,1] = x.reshape(-1)
X[:,2] = x.reshape(-1)**2

# Finding beta
beta = (np.linalg.inv(X.T@X) @ X.T) @ y
ytilde = X @ beta

plot_data(x, ytilde, 's', name="No lambda")
#plt.legend()
#plt.show()

#############
## Ridge   ##
#############

lambdas = np.linspace(0.1,4,3)
for lamb in lambdas:
    betaRidge = np.linalg.inv(X.T.dot(X) + lamb*np.eye(3)).dot(X.T).dot(y)
    ytildeRidge = X @ betaRidge
    plot_data(x, ytildeRidge, 'o', name=("Ridge w/lambda "+(str(lamb))))

plt.legend()
plt.show()

# Using sklearn ridge for comparison
_lambda = 0.1
clf_ridge = skl.Ridge(alpha=_lambda).fit(X, y)
yridge = clf_ridge.predict(X)
plot_data(x,y,'s')
plot_data(x,yridge,'*')
plt.show()
