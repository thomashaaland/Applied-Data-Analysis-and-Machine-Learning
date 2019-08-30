import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.random.rand(100,1)
y = 5*x+0.01*np.random.randn(100,1)
linreg = LinearRegression()
linreg.fit(x,y)
ypredict = linreg.predict(x)

plt.plot(x, np.abs(ypredict-y)/abs(y), "ro")
plt.axis([0,1.0,0.0,0.5])
plt.xlabel(r'$x$')
plt.ylabel(r'$\epsilon_{mathrm{relative}}$')
plt.title(r'Relative error')
plt.show()
