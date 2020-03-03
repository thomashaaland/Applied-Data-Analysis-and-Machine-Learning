import numpy as np

# Note: for this dataset y.shape = (3000, 1), and X.shape = (3000, 23).
# beta thus needs to be beta.shape = (23, 1); (X @ beta).shape = (3000, 1)

class LogisticRegression():
    """
    When initialising, choose which solver to use.
    Choices:
     nrm = Newton Raphson's method
     steepest descent
     sgd = stochastic gradient descent
    """
    def __init__(self, epochs = 5, solver="nrm", learning_rate = 0.01, M = 10):
        self.epochs = epochs
        if solver == "nrm":
            self.solver = self.rnm
        elif solver == "steepest descent":
            self.solver = self.steepest_descent
            self.learning_rate = learning_rate
        elif solver == "sgd":
            self.solver = self.sgd
            self.M = M

    def softMax(self, X, beta):
        #exponent = np.exp( X @ beta)
        #exponent[-1] = 1
        #summ = np.sum(exponent, axis=1).reshape(-1,1)
        #print("SUMM: {}".format(summ.shape))
        #z = exponent / summ
        #print("Z: {}".format(z.shape))
        
        z = 1/(1+np.exp(- X @ beta))
        return z

    def dC(self, X, p, y):
        dC = -X.T @ (y - p)
        return dC

    # ddC shape: X.T.shape = (23, 3000) -> ddC.shape = (23, 23)
    def ddC(self, X, p):
        p = p.ravel(-1)
        w = (p * (1 - p))
        result = np.zeros((X.shape[1], X.shape[1]))
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                first_Mat = X.T[i][:].reshape(1,-1)
                second_Mat = (w * X.T[:][j]).reshape(-1,1)
                result[i][j] = first_Mat @ second_Mat
        result =  np.linalg.pinv(result)
        return result

    # returns a new beta from an old beta
    def rnm(self, i): # Raphsod newton method
        p = self.softMax(self.X, self.beta)
        dC_numeric = self.dC(self.X, p, self.y)
        ddC_numeric = self.ddC(self.X, p)
        increment = ddC_numeric @ dC_numeric
        self.beta = self.beta - increment

    def sgd(self, i):
        # make selection of points:
        # number of minibatches: M
        M = self.M
        # number of datapoints in set
        n = self.y.shape[0]
        # number of points per set
        K = int(n/M)
        for i in range(M):
            indices = np.random.choice(n, K, replace=False)
            p = self.softMax(self.X[indices], self.beta)
            dC_numeric = self.dC(self.X[indices], p, self.y[indices])
            ddC_numeric = self.ddC(self.X[indices], p)
            increment = ddC_numeric @ dC_numeric
            self.beta = self.beta - increment
        
    def steepest_descent(self, i):
        p = self.softMax(self.X, self.beta)
        dC_numeric = self.dC(self.X, p, self.y)
        self.beta = self.beta - self.learning_rate * dC_numeric
        
    def regression(self, beta, dC, ddC):
        return beta - ddC @ dC

    def initBeta(self, x, y):
        beta = np.random.randn(x,y)
        return beta

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.beta = self.initBeta(X.shape[1], y.shape[1])
        for epoch in range(self.epochs):
            self.solver(epoch)

    def predict(self, X):
        return self.softMax(X, self.beta)
