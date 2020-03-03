import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
import homeBrewNN as hBNN
from sklearn.model_selection import train_test_split
import oblig1_header as oh
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import sklearn.neural_network
import sklearn.model_selection
import sklearn.metrics


# load terrain
path = os.path.dirname(os.path.abspath(__file__))
print(path)
terrain1 = imread(path + "/SRTM_data_Norway_1.tif")
# Show terrain
plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

print(terrain1.shape)

################################
## Preprocessing              ##
################################



terrain = terrain1[200:401, 200:401]
terrain = terrain[::1,::1]
print('ts',terrain.shape)
terrain = terrain - np.mean(terrain)
terrain = terrain/np.max(terrain)
print("Min max, terrain: ", np.max(terrain), np.min(terrain))


x = np.linspace(-1,1,terrain.shape[0])
y = np.linspace(-1,1,terrain.shape[1])
x_, y_ = np.meshgrid(x, y)
x = x_.ravel()
y = y_.ravel()
X = np.c_[x, y]
z = terrain.ravel()

plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')



X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.5)

reg = sklearn.neural_network.MLPRegressor(
    hidden_layer_sizes=(100, 80, 60, 40, 20),
    learning_rate="adaptive",
    learning_rate_init=0.01,
    max_iter=1000,
    tol=1e-9,
    verbose=True,
)
reg = reg.fit(X_train, y_train)

# See some statistics
pred = reg.predict(X_test)

print(f"MSE = {sklearn.metrics.mean_squared_error(y_test, pred)}")
print(f"R2 = {reg.score(X_test, y_test)}")
pred = reg.predict(X)
z = pred.reshape(x_.shape)
plt.figure()
plt.title("Terrain over Norway prediction")
plt.imshow(z, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
