import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from IPython.display import display
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# Regress this data
x = np.random.rand(100)
y = 5*x*x+0.1*np.random.randn(100)

plt.plot(x,y, 's')
#plt.show()

# Set up the design matrix X
X = np.zeros((len(x), 3))
X[:,0] = 1
X[:,1] = x
X[:,2] = x**2



#Finding beta
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
print('beta0 = {}, beta1 = {}, beta2 = {}'.format(beta[0], beta[1], beta[2]))

# create y_tilde by doing matrix multiplication between X and beta
ytilde = X@beta

plt.plot(x,ytilde, 'o')

# beta contains coefficients, so can use it the same way as when we made y
# excluding the noise
x_ = np.linspace(0,1,101)
y_ = beta[2]*x_*x_ + beta[1]*x_ + beta[0]

plt.plot(x_,y_)

# Using scikitlearn for linear regression

clf = skl.LinearRegression().fit(X, y)
ytilde2 = clf.predict(X)

plt.plot(x, ytilde2, 'o')
plt.show()

# using sklearn to find Mean Square Error (MSE)
print("Mean squared error: {0:.6f}".format(mean_squared_error(y, ytilde2)))
print("Variance score: {0:.6f}".format(r2_score(y, ytilde2)))

"""
The MSE is around 0.01, which is what is expected when the data is
distributed normally around some function with sigma 0.1. 

MSE = sigma^2
"""

"""
r_squared is about 0.9957 which means nearly all the variance is explained 
by the model.
"""

