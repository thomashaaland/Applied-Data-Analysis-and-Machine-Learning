import oblig2_header
import homeBrewNN as hBNN
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad as egrad
from autograd import jacobian

# ReLu
def ReLu(z):
    z[z < 0] = 0
    return z

# The sigmoid, simple and no fuzz
def sigmoid(z):
    return 1/(1+np.exp(-z))

def softMax(z):
    new_z = []
    for zpart in z:
        new_z.append([1, zpart[1]])
    new_z = np.array(new_z)
    expo = np.exp(new_z)
    a = expo
    #print(a)
    b = (np.sum(expo, axis=1, keepdims=True))
    return a/b

f = softMax

def crossEntropy(z, f):
    return -np.sum(t*np.log(f(z)))

crossEntropyGrad = egrad(crossEntropy, 0)

x = np.linspace(-8,8,2*26).reshape(26,2)
p = softMax(x) # The real probability distribution
t = (np.random.rand(len(p)) < p.T[:][0]).astype(int).reshape(-1,1)

tOneHot = t.copy()
tOneHot[t == 0] = 1
tOneHot[t == 1] = 0
tOneHot = np.append(tOneHot, t, axis=1)

print(crossEntropy(x, f))
grad = crossEntropyGrad(x, f)

crossEntropyGradAnalytical = (f(x) - t)

plt.plot(x[:,0], f(x))
plt.plot(x, sigmoid(x))
plt.show()
    
def main():
    
    # Create the data:
    # first make the features: x
    x = np.linspace(-4,4,26*10*4).reshape(4*13*10,2)
    p = sigmoid(1 * x) # The real probability distribution
    data = (np.random.rand(len(p)) < p.T[:][0]).astype(int).reshape(-1,1)

    # One hot data:
    print(data[data == 0])
    #print("Data: ", data.shape)
    dataOneHot = data.copy()
    dataOneHot[data == 0] = 1
    dataOneHot[data == 1] = 0
    dataOneHot = np.append(dataOneHot, data, axis=1)

    #print("data one hot:", dataOneHot)
    
    # the model takes in a list of the desired layers with specified number of outputlayers,
    # type of layer (only have dense so far)
    myModel = hBNN.NNhomebrew([hBNN.layers(features = x, layerType="featureLayer"), # sending in x-train at this layer. X = (nInputs, nFeatures) = (101, 1)
                               hBNN.layers(outputNodes=50, activation="ReLu"),
                               hBNN.layers(outputNodes=50, activation="ReLu"),
                               hBNN.layers(outputNodes=30, activation="ReLu"),
                               hBNN.layers(outputNodes=2, activation="softMax")])
    myModel.compile(loss = "crossEntropy", optimizer = "gradientDescent", regularization="L2", _lambda=0.0000001, weightNormalizing = True)
    myModel.fit(dataOneHot, validation_target=dataOneHot, validation_data=x, learningRate = 0.1, epochs=500, backPropagation = "normal")
    # try to make a fit to the data; sending in y-train and y-validation at this point. Backward propagation can start at this point

    # Make test data
    x_test = np.linspace(-4,4,26*10).reshape(13*10,2)
    data_test = sigmoid(x_test)
    y_test = (np.random.rand(len(data_test)) < data_test.T[:][0]).astype(int).reshape(-1,1)
    
    prediction = myModel.predict_probabilities(x_test) # predict takes in features on which the model perform a prediction
    #print(prediction)
    plt.plot(x_test, prediction, '.')
    plt.plot(x, dataOneHot, '.')
    plt.show()

    #hBNN.cumulative_plot(y_test, prediction[:,1])
    
### Execute the program
main()
