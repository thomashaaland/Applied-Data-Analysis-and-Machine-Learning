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
terrain = terrain - np.mean(terrain)
terrain = terrain/np.max(terrain)
print("Min max, terrain: ", np.max(terrain), np.min(terrain))

pX = np.random.permutation(terrain.shape[0])
pX_train = np.sort(pX[:int(terrain.shape[0]/2)])
pX_test = np.sort(pX[int(terrain.shape[0]/2):int(terrain.shape[0]*3/4)])
pX_validate = np.sort(pX[int(terrain.shape[0]*3/4):])

pY = np.random.permutation(terrain.shape[1])
pY_train = np.sort(pY[:int(terrain.shape[1]/2)])
pY_test = np.sort(pY[int(terrain.shape[1]/2):int(terrain.shape[1]*3/4)])
pY_validate = np.sort(pY[int(terrain.shape[1]*3/4):])

terrain_train = ((terrain[pX_train]).T[pY_train]).T
terrain_test = ((terrain[pX_test]).T[pY_test]).T
terrain_validate = ((terrain[pX_validate]).T[pY_validate]).T
trainDims = terrain_train.shape
testDims = terrain_test.shape
validateDims = terrain_validate.shape

#p_xTrain = ((np.arange(terrain.shape[0]**2)).reshape(terrain.shape[0], terrain.shape[1])[p_train].T)[p_train]
#p_xTest = ((np.arange(terrain.shape[0]**2)).reshape(terrain.shape[0], terrain.shape[1])[p_test].T)[p_test]
#p_xValidate = ((np.arange(terrain.shape[0]**2)).reshape(terrain.shape[0], terrain.shape[1])[p_validate].T)[p_validate]


plt.imshow(terrain_train)
plt.title("Training sample")
plt.show()
plt.imshow(terrain_test)
plt.title("Test sample")
plt.show()
plt.imshow(terrain_validate)
plt.title("Validation sample")
plt.show()

x = np.linspace(0,1,terrain.shape[0])
y = np.linspace(0,1,terrain.shape[1])
x, y = np.meshgrid(x, y)
x = x.ravel().reshape(-1,1)
y = y.ravel().reshape(-1,1)
xy = oh.create_X(x.ravel(), y.ravel(), 7, intercept=True)
print("X: ", xy.shape)

polyGons = 11
# Sklearn PCA to reduce down to 2 dims!
pca = PCA(n_components = 0.95)

x_train = np.linspace(0,1,terrain.shape[0])[pX_train]
y_train = np.linspace(0,1,terrain.shape[1])[pY_train]
x_train, y_train = np.meshgrid(x_train, y_train)
x_train = x_train.ravel().reshape(-1,1)
y_train = y_train.ravel().reshape(-1,1)
xy_train = oh.create_X(x_train.ravel(), y_train.ravel(), polyGons, intercept=False)
xyTransform_train = pca.fit(xy_train)
xy_train = xyTransform_train.transform(xy_train)

x_test = np.linspace(0,1,terrain.shape[0])[pX_test]
y_test = np.linspace(0,1,terrain.shape[1])[pY_test]
x_test, y_test = np.meshgrid(x_test, y_test)
x_test = x_test.ravel().reshape(-1,1)
y_test = y_test.ravel().reshape(-1,1)
xy_test = oh.create_X(x_test.ravel(), y_test.ravel(), polyGons, intercept=False)
xyTransform_test = pca.fit(xy_test)
xy_test = xyTransform_test.transform(xy_test)

x_validate = np.linspace(0,1,terrain.shape[0])[pX_validate]
y_validate = np.linspace(0,1,terrain.shape[1])[pY_validate]
x_validate, y_validate = np.meshgrid(x_validate, y_validate)
x_validate = x_validate.ravel().reshape(-1,1)
y_validate = y_validate.ravel().reshape(-1,1)
xy_validate = oh.create_X(x_validate.ravel(), y_validate.ravel(), polyGons, intercept=False)
xyTransform_validate = pca.fit(xy_validate)
xy_validate = xyTransform_validate.transform(xy_validate)

terrainOriginalShape = terrain.shape
terArray = np.array(terrain).ravel()

plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terArray.reshape(terrainOriginalShape), cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

################################
## Initiate NN                ##
################################

inputY = terArray.reshape(-1,1)

x_train = xy_train
y_train = np.array(terrain_train).ravel().reshape(-1,1)

x_test = xy_test
y_test = np.array(terrain_test).ravel().reshape(-1,1)

x_validate = xy_validate
y_validate = np.array(terrain_validate).ravel().reshape(-1,1)

print(x_train.shape, y_train.shape)

################################################
## GRID SEARCH:                               ##
## Learning rate                              ##
## Lambda                                     ##
################################################
nGrid = 11
learningRates = np.linspace(1./nGrid, 0.4, nGrid)
lambdas = np.linspace(0, 1, nGrid)
costTrainMatrix = np.zeros((nGrid, nGrid))
costValMatrix = np.zeros((nGrid, nGrid))


for i, learningRate in enumerate(learningRates):
    for j, _lambda in enumerate(lambdas):
        print("Learningrate: {}, Lambda: {}".format(learningRate, _lambda))
        myModel = hBNN.NNhomebrew([hBNN.layers(features = x_train, layerType="featureLayer"), # sending in x-train at this layer. X = (nInputs, nFeatures) = (101, 1)
                                   hBNN.layers(outputNodes=64, activation="ReLu"),
                                   hBNN.layers(outputNodes=64, activation="ReLu"),
                                   hBNN.layers(outputNodes=1, activation="linearFit")])
        myModel.compile(loss = "MSE", optimizer = "gradientDescent", regularization="L2", _lambda=_lambda, weightNormalizing = True)
        myModel.fit(y_train, validation_target = y_validate, validation_data = x_validate, learningRate = learningRate, epochs=300, backPropagation = "normal")
        costTrainMatrix[i,j] = myModel.scores()[0]
        costValMatrix[i,j] = myModel.scores()[1]

plt.imshow(costTrainMatrix)
plt.title("Cost for Training")
plt.show()
plt.imshow(costValMatrix)
plt.title("Cost for Validation")
plt.show()
learnRateAndLambda = np.where(costTrainMatrix == np.nanmin(costTrainMatrix))
bestLearnRate = learningRates[learnRateAndLambda[0]]
bestLambda = lambdas[learnRateAndLambda[1]]
print(learnRateAndLambda)
print("Best learnRate: ", bestLearnRate)
print("Best lambda: ", bestLambda)

myModel = hBNN.NNhomebrew([hBNN.layers(features = x_train, layerType="featureLayer"), # sending in x-train at this layer. X = (nInputs, nFeatures) = (101, 1)
                           hBNN.layers(outputNodes=64, activation="ReLu"),
                           hBNN.layers(outputNodes=64, activation="ReLu"),
                           hBNN.layers(outputNodes=1, activation="linearFit")])
myModel.compile(loss = "MSE", optimizer = "gradientDescent", regularization="L2", _lambda=bestLambda, weightNormalizing = True)
myModel.fit(y_train, validation_target = y_validate, validation_data = x_validate, learningRate = bestLearnRate, epochs=300, backPropagation = "normal")
historyTrain, historyVal = myModel.history()
prediction = myModel.predict(x_test) # predict takes in features on which the model perform a prediction

print("Test MSE: ", 0.5/np.x_test.shape[0]*np.sum((prediction - y_test)**2))

print(np.max(prediction), np.min(prediction))
plt.imshow(prediction.reshape(testDims), cmap='gray')
plt.show()

plt.plot(historyTrain, label="Train")
plt.plot(historyVal, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("MSE as function of epoch")
plt.legend()
plt.show()
