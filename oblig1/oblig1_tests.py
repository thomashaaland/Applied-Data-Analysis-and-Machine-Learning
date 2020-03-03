import oblig1_header as oh
import numpy as np

# Test Expectation number:
print("Testing expectation number:")
z = np.array([1,2,3,4,5])
assert oh.E(z) == 3
print("The function E(z) works as advertised")


# test Mean Squared Error
print("Testing Mean Squared Error:")
print("Testing a one dimensional array")
z = np.ones(4)
assert oh.mse(z,z) == 0
assert oh.mse(z,z+1) == 1
print("Testing a matrix")
z2 = z.reshape(2,2)
assert oh.mse(z,z) == 0
assert oh.mse(z,z+1) == 1
print("The function mse(z,ztilde) works as advertised")

# Test the variance
print("Testing the variance")
z = np.ones(4)
assert oh.var(z,z) == 0
assert oh.var(z,z+1) == 0
print("Testing a matrix")
z2 = z.reshape(2,2)
assert oh.var(z,z) == 0
assert oh.var(z,z+1) == 0
print("The function var(z,ztilde) works as advertised")

# Test the bias
print("Testing the bias")
z = np.ones(4)
assert oh.bias(z,z) + oh.var(z,z) == oh.mse(z,z)
assert oh.bias(z,z+1) + oh.var(z,z+1) == oh.mse(z,z+1)
print("Testing a matrix")
z2 = z.reshape(2,2)
assert oh.bias(z,z) + oh.var(z,z) == oh.mse(z,z)
assert oh.bias(z,z+1) + oh.var(z,z+1) == oh.mse(z,z+1)
print("The function bias(z,ztilde) works as advertised")

# Test R2 score
print("Testing the R2 score")
z = np.arange(4)+1
assert oh.R2_score(z,z) == 1
print("Testing a matrix")
z2 = z.reshape(2,2)
assert oh.R2_score(z,z) == 1
print("The function R2_score(z,ztilde) works as advertised")

print("Testing linear Regression functions")
x = np.random.randn(11)
y = np.random.randn(11)
X = oh.create_X(x, y, 2)
np.set_printoptions(precision=2)
print(X)
z = x + y
zTildeOLS = X @ oh.linFit(X, z, model='OLS')
assert oh.mse(z, zTildeOLS) < 10**(-28)
print("OLS works as advertised")
zTildeRidge = X @ oh.linFit(X, z, model='Ridge', _lambda=0.1)
assert oh.mse(z, zTildeOLS) < 10**(-28)
print("Ridge works as advertised")
zTildeLasso = X @ oh.linFit(X, z, model='Lasso', _lambda=0.1)
assert oh.mse(z, zTildeOLS) < 10**(-28)
print("Ridge works as advertised")
