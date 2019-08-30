import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
eye = np.eye(4)
print(eye)
sparse_mtx = sparse.csr_matrix(eye)
print(sparse_mtx)
x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x,y,marker='x')
plt.show()
