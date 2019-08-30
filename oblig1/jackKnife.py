from numpy import *
from numpy.random import randint, randn
from time import time
import matplotlib.pyplot as plt

def jackknife(data, stat):
    n = len(data);t = zeros(n); inds = arange(n); t0 = time()
    for i in range(n):
        t[i] = stat(delete(data,i) )

    print("Runtime: %g sec" % (time()-t0)); print("Jackknife Statistics :")
    print("original           bias      std. error")
    print("%8g %14g %15g %17g" % (stat(data),(n-1)*mean(t)/n, (n*var(t))**.5, (var(data))**.5))
    print("len data: {}, len jackKnife: {}".format(len(data), len(t)))
    return t

#returns mean of data samples
def stat(data):
    return mean(data)

mu, sigma = 100, 15
datapoints = 10000
x = mu + sigma*random.randn(datapoints)
# jackknife returns the data sample
t = jackknife(x, stat)
