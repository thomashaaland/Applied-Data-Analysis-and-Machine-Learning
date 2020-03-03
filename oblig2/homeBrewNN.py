import autograd.numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, auc, roc_curve
from autograd import elementwise_grad as egrad
from autograd import grad

def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    if n_categories <2:
        n_categories = 2
    else:
        pass
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[np.arange(n_inputs), integer_vector] = 1
    return onehot_vector
class layers():
    def __init__(self, outputNodes=None, features=None, activation="None", layerType="Dense"):
        self.activation = activation
        self.layerType = layerType
        if self.layerType == "featureLayer":
            self.featureLayer(features)
        elif self.layerType == "Dense":
            self.Dense(outputNodes)
        elif self.layerType == "output":
            self.outputLayer(outputNodes)
        self.hasInitiated = False
        
    def Dense(self, outputNodes):
        self.outputNodes = outputNodes

    def featureLayer(self, features):
        if len(features.shape) < 2:
            self.a = features.reshape(-1,1)
        else:
            self.a = features

    def outputLayer(self, outputNodes):
        self.outputNodes = outputNodes

    def getFeatures(self):
        return self.a

    def constantsInit(self, X, weightNormalizing = False): #inputNodes):
        if self.hasInitiated == False:
            print("Initiate variables")
            nInput, nFeatures = X.shape
            ##################################
            ## Using Kaiming Initialization ##
            ##################################
            if weightNormalizing == True:
                self.W = np.random.randn(nFeatures, self.outputNodes) * np.sqrt(2./nInput)#*np.sqrt(1/5) # W = (nFeatures, nOutnodes)
            else:
                self.W = np.random.randn(nFeatures, self.outputNodes)#*np.sqrt(2/nInput) # W = (nFeatures, nOutnodes)
            self.b = np.zeros((1, self.outputNodes)) + 0.1 # b = (nInput, nOutnodes)
            self.hasInitiated = True

## Neural network class
class NNhomebrew():
    # Initialise the network
    def __init__(self, layerList, network_type='classification'):
        self.LOGO()
        self.layerList = layerList # compile a list of all the layers
        self.network_type = network_type
        self.costTrainHistory = [] # keep track of cost history
        self.costValHistory = []
        self.predictTrainHistory = [] # keep track of predicts
        self.predictValHistory = []
        self.a = layerList[0].getFeatures() # Initiate the featurelayer

    def accuracy(self, z, f, t):
        if self.network_type == 'classification':
            predict = f(z)
            predict = to_categorical_numpy(np.argmax(predict, axis=1))
            return np.sum(predict==t)/t.shape[1]/t.shape[0]

        elif self.network_type == 'regression':
            predict = f(z)
            return np.mean((predict-t)**2)

    # Returns a set of scores: Currently for train and val: Cost, accuracy 
    def scores(self):
        return self.costTrainHistory[-1], self.costValHistory[-1]#, self.accuracyTrainHistory[-1], self.accuracyValHistory[-1]
    
    def history(self):
        return self.costTrainHistory, self.costValHistory
    
    def learning_schedule(self, t):
        t0 = self.eta
        return t0*np.exp(-t)


    ####################################
    ## Backpropagation using          ##
    ## gradient descent               ##
    ####################################
    def backpropagation_norm(self, layerList, epoch):
        # Start with the top layer and continue down to the bottom layer
        # For the output Layer:
        N = len(self.layerList)
        for i in range(0, N-1):
            currentLayerCount = N-i-1
            previousLayerCount = N-i-2
            nextLayerCount = N-i
            cLayer = self.layerList[currentLayerCount] # current layer
            pLayer = self.layerList[previousLayerCount] # previous layer
            cA = cLayer.a
            pA = pLayer.a
            
            # If in outputLayer:
            if currentLayerCount == N-1:
                self.activationMethod(cLayer)
                dCdz = egrad(self.costMethod, 0)
                cLayer.delta = dCdz(cLayer.z, self.f, cLayer.W, self._lambda, self.t) 

                self.layerList[currentLayerCount].W = cLayer.W - self.eta * np.dot(pA.T, cLayer.delta) 
                self.layerList[currentLayerCount].b = cLayer.b - self.eta * np.mean(cLayer.delta, axis=0, keepdims=True)

                
            # For all other layers:
            else:
                self.activationMethod(cLayer)
                nLayer = layerList[nextLayerCount] # Next layer
                cLayer.delta = np.dot(nLayer.delta, nLayer.W.T) * self.df(cLayer.z)
                self.layerList[currentLayerCount].W = cLayer.W - self.eta * np.dot(pA.T, cLayer.delta)
                self.layerList[currentLayerCount].b = cLayer.b - self.eta * np.mean(cLayer.delta, axis=0, keepdims=True).reshape(1,-1)
    
        
    ####################################
    ## Backpropagation using          ##
    ## stochastic gradient descent    ##
    ####################################


    
    def learning_schedule(self, t):
        if self.eta != 0:
            t0 = self.eta
        else:
            t0 = self.eta + 0.0000001
        t1 = 0.1
        return t0/(t+t1)
    
    def backpropagation_stoc(self,layerList, epoch):

    # Start with the top layer and continue down to the bottom layer
    # For the output Layer:
        N = len(layerList)
        indices = self.chosen_datapoints #indices of the datapoints in the current minibatch
        for i in range(0, N-1):
            currentLayerCount = N-i-1
            previousLayerCount = N-i-2
            nextLayerCount = N-i
            cLayer = self.layerList[currentLayerCount] # current layer
            pLayer = self.layerList[previousLayerCount] # previous layer
            cA = cLayer.a
            pA = pLayer.a
            nData = cLayer.z.shape[0]
            # If in outputLayer:
            if currentLayerCount == N-1:
                self.activationMethod(cLayer)
                C = self.costMethod
                dCdz = egrad(C, 0)
                temp_z = cLayer.z[indices]
                temp_pA = pA[indices]
                temp_t = self.t[indices]
                
                eta = self.learning_schedule(epoch*nData+i)
                eta = self.eta
                self.layerList[currentLayerCount].delta = dCdz(temp_z, self.f, cLayer.W, self._lambda, temp_t)
                self.layerList[currentLayerCount].W = cLayer.W - eta * np.dot(temp_pA.T, cLayer.delta)
                self.layerList[currentLayerCount].b = cLayer.b - eta * np.mean(cLayer.delta, axis=0, keepdims=True)    
                    
            # For all other layers:
            else:
                self.activationMethod(cLayer)
                nLayer = self.layerList[nextLayerCount] # Next layer
                temp_z = cLayer.z[indices]
                temp_pA = pA[indices]
                eta = self.learning_schedule(epoch*nData+i)
                eta = self.eta
                self.layerList[currentLayerCount].delta = np.dot(nLayer.delta, nLayer.W.T) * self.df(temp_z)
                self.layerList[currentLayerCount].W = cLayer.W - eta * np.dot(temp_pA.T, self.layerList[currentLayerCount].delta)
                self.layerList[currentLayerCount].b = cLayer.b - eta * np.mean(self.layerList[currentLayerCount].delta, axis=0, keepdims=True)
            
    def fit(self, train_t, validation_target = None, validation_data = None, learningRate = 0.01, 
            epochs = 10,batchsize = 100,  backPropagation = "normal"):
        # The targets come in here. This is the training data
        self.t = train_t
        self.val_t = validation_target
        self.val_x = validation_data
        self.eta = learningRate
        self.epochs = range(epochs)
        if backPropagation == "normal":
            self.backpropagation = self.backpropagation_norm
            self.feedForward()
            for epoch in self.epochs:

                self.backpropagation(self.layerList, epoch)
                self.feedForward()
                cost_train = self.costMethod(self.layerList[-1].z, self.f, self.layerList[-1].W, self._lambda, self.t)
                self.costTrainHistory.append(cost_train)

                #if self.val_t != None:
                validation_z = self.feedForwardOut(self.val_x, returnZ = True)
                cost_validation = self.costMethod(validation_z, self.f, self.layerList[-1].W, self._lambda, self.val_t)
                self.costValHistory.append(cost_validation)
            
                if epoch % 1 == 0:
                    print("Epoch: {}, cost: {}, TrainAccuracy: {}".format(
                        epoch, cost_train, self.accuracy(self.layerList[-1].z, self.f, self.t)))
        
        
        if backPropagation == "stochastic":
            self.backpropagation = self.backpropagation_stoc
            n_inputs = train_t.shape[0]
            n_batches = int(n_inputs/batchsize)
            self.feedForward()
            for epoch in self.epochs:
                data_indices = np.arange(n_inputs)
                for i in range(n_batches):
                    self.chosen_datapoints = np.random.choice(
                        data_indices, size=batchsize, replace=False)
                    
                    self.backpropagation(self.layerList, epoch)
                    self.feedForward()
                cost_train = self.costMethod(self.layerList[-1].z, self.f, self.layerList[-1].W, self._lambda, self.t)
                self.costTrainHistory.append(cost_train)

                #if self.val_t != None:
                validation_z = self.feedForwardOut(self.val_x, returnZ = True)
                cost_validation = self.costMethod(validation_z, self.f, self.layerList[-1].W, self._lambda, self.val_t)
                self.costValHistory.append(cost_validation)
                if epoch % 1 == 0:
                    print("Epoch: {}, cost: {}, TrainAccuracy: {}".format(
                        epoch, cost_train, self.accuracy(self.layerList[-1].z, self.f, self.t)))

    
    #################################
    ## Predict is used to send in  ##
    ## in a validation set to      ##
    ## make predictions on the     ##
    ## dataset.                    ##
    #################################
                
    def predict(self, x):
    
        if self.network_type == 'classification':
            probabilities = self.feedForwardOut(x)
            onehot = to_categorical_numpy(np.argmax(probabilities, axis=1))
            return onehot

        else:
            print('reg')
            predict = self.feedForwardOut(x)
            return predict

    def predict_probabilities(self, x):   
        probabilities = self.feedForwardOut(x)    
        probabilities/np.sum(probabilities, axis=1, keepdims=True)
        return probabilities
    
    def feedForward(self):
        for layer in self.layerList:
            if layer.layerType == "featureLayer":
                a = layer.a
            if layer.layerType == "Dense":
                layer.constantsInit(a, self.weightNormalizing)
                z = np.dot(a, layer.W) + layer.b
                self.activationMethod(layer.activation)
                a = self.f(z)
                layer.z = z
                layer.a = a

    def feedForwardOut(self, x, returnZ = False):
        for layer in self.layerList:
            if layer.layerType == "featureLayer":
                a = x
            elif layer.layerType == "Dense":
                z = np.dot(a, layer.W)  + layer.b
                self.activationMethod(layer.activation)
                a = self.f(z) # y is the 'prediction' for this layer
        if returnZ == False:
            return a # y is the 'prediction' from this layer. Should have outputshape (outputNodes, len(featureData))
        else:
            return z
        
    # Use this to set the activation function, should be tied into cost
    # and loss functions? Or maybe we should be able to choose freely
    def activationMethod(self, activation):
        if activation == "sigmoid":
            self.f = self.sigmoid
            self.df = self.dsigmoid
        if activation == "softMax":
            self.f = self.softMax
            self.df = self.dsoftMax
        if activation == "ReLu":
            self.f = self.ReLu
            self.df = self.dReLu
        if activation == "tanh":
            self.f = self.tanh
            self.df = self.dtanh
        if activation == "linearFit":
            self.f = self.linearFit
            self.df = self.dlinearFit
        if activation == "leakyRelu":
            self.f = self.leakyRelu
            self.df = self.dleakyRelu

    ###############################################
    ## ACTIVATION FUNCTIONS                      ##
    ###############################################
    # Here follows methods for all different activation functions:

    # C = e^(logC) = e^D => D = logC
    # The sigmoid, simple and no fuzz
    def sigmoid(self, z):
        #z = np.dot(pA, W) + b
        f = 1/(1+np.exp(-z))
        return f

    dsigmoid = egrad(sigmoid,1)
    
    # Softmax with arbitrary number of classes.
    # input needs to be: X.shape = (nData, nDataType),
    #                    beta.shape = (nDataType, nClasses)
    def softMax(self, z):
        #z = np.dot(pA, W) + b
        b = np.exp(z - np.max(z))
        a = np.divide(b, (np.sum(b, axis=1, keepdims=True)))
        return a

    dsoftMax = egrad(softMax,1)
    
    # Tanh:
    def tanh(self, z):
        #z = np.dot(pA, W) + b
        return np.tanh(z)
    
    dtanh = egrad(tanh,1)

    # ReLu
    def ReLu(self, z):
        #z = np.dot(pA, W) + b
        return np.maximum(z, 0)

    def dReLu(self, z):
        z[a <=0] = 0
        z[a > 0] = 1
        return z

    #leaky ReLu
    c = 0.001
    def leakyRelu(self, x):
        return np.where(x > 0, x, x * c)
    
    def dleakyRelu(self, x):
        return np.where(x>0, 1, c)

    
    # Linear fit
    def linearFit(self, z):
        #z = np.dot(pA, W) + b
        return z

    dlinearFit = egrad(linearFit,1)

    ###############################################
    ## COST FUNCTIONS                            ##
    ###############################################
    # The different cost functions:

    def MSE(self, z, f, W, _lambda, t):
        #z = np.dot(pA, W) + b
        return 0.5 * np.sum((f(z) - t)**2)/(z.shape[0]) # + self.reg(W, _lambda)
    def dMSE(self, z, f, W, _lambda, t):
        return np.sum(f(z)-t)
    '''
    def crossEntropy(self,z, f, W, _lambda, t):
        pred = f(z)
        label = t
        
        yl=np.multiply(pred,label)
        yl=yl[yl!=0]
        yl=-np.log(yl)
        yl=np.mean(yl)
        return yl
    '''
    def crossEntropy(self, z, f, W, _lambda, t):
        #z = np.dot(pA, W) + b
        return -np.sum(np.sum(t * np.log(f(z)), axis=1))/(z.shape[0])# + self.reg(W, _lambda)
    
    ##############################################
    ## Regularization methods                   ##
    ##############################################

    def noReg(self, W, _lambda):
        return 0
    dnoReg = lambda W, _lambda: 0
    
    def regL2(self, W, _lambda):
        return _lambda * np.sum(W**2) / (2 * W.shape[0] * W.shape[1])
    dregL2 = egrad(regL2)
                                         
    def regL1(self, W, _lambda):
        return _lambda * np.sum(W) / (2 * W.shape[0] * W.shape[1])
    dregL1 = egrad(regL1)

    ##############################################
    ## Compile function:                        ##
    ## Must be called to compile the network    ##
    ## and assign a cost function, a type of    ##
    ## descent and the metric to determine the  ##
    ## effectiveness of the network             ##
    ##############################################
    
    def compile(self, optimizer="gradientDescent", loss="crossEntropy", metrics=["accuracy"], regularization="no reg", _lambda = 0.01, weightNormalizing = False):
        self.loss = loss
        self._lambda = _lambda
        self.weightNormalizing = weightNormalizing
        #############################
        # Determine loss            #
        #############################
        if loss == "crossEntropy":
            self.costMethod = self.crossEntropy
        elif loss == "MSE":
            self.costMethod = self.MSE
        #############################
        # Determine regularization  #
        #############################
        if regularization == "no reg":
            self.reg = self.noReg
            self.dreg = self.dnoReg
        elif regularization == "L2":
            self.reg = self.regL2
            self.dreg = self.dregL2
        elif regularization == "L1":
            self.reg = self.regL1
            self.dreg = self.dregL1
            
            if epoch % 100 == 0:
                print("Epoch: {}, cost: {}".format(
                    epoch, cost_train))#, self.accuracy(self.layerList[-1].z, self.f, self.t)))
       
    def LOGO(self):
        print(
            """
            #########################################################
            ##                                                     ##
            ##  HH  HH  OO  M   M EEEEE BBBB  RRR   EEEEE W     W  ##
            ##  HH  HH O  O MM MM E     B   B R  R  E     W     W  ##
            ##  HHHHHH O  O M M M EEE   BBBB  RRR   EEE    W W W   ##
            ##  HH  HH O  O M   M E     B   B R  R  E      W W W   ##
            ##  HH  HH  OO  M   M EEEEE BBBB  R   R EEEEE   W W    ##
            ##                                                     ##
            #########################################################
            """
        )


def cumulative_plot(y_test,y_test_pred, title = 'cumulative curve'):
    #cumulative gains/lift chart/area ratio
    print(y_test_pred)
    sort = np.argsort(-y_test_pred,axis = 0)
    #print('sort', sort)
    y_test_pred[np.squeeze(sort)]
    curve_model_1 = np.cumsum(y_test[np.squeeze(sort)])
    print('curve_1', len(curve_model_1))
    curve_perfect_model = np.cumsum(-np.sort(-y_test, axis = 0))
    #print(curve_perfect_model)
    curve_no_model = np.linspace(curve_perfect_model[-1]/len(y_test),curve_perfect_model[-1],num=len(y_test))
    #print('y_testtttt', y_test)
    #print(curve_no_model)
    print('cuuur',curve_perfect_model[-1])
    print('len_ytest',len(y_test))
    print('curve_1', len(curve_model_1))
    area_model = auc(np.arange(len(y_test)), curve_model_1)
    #print('area_model', area_model)
    area_perfect_model = auc(np.arange(len(y_test)), curve_perfect_model)
    #print('area_perfect_model', area_perfect_model)
    area_no_model = auc(np.arange(len(y_test)), curve_no_model)
    #print('area_no_model', area_no_model)
    cumulative_area_ratio = (area_model-area_no_model)/(area_perfect_model-area_no_model)
    plt.figure(3)
    plt.plot(np.arange(len(y_test))[:,None], curve_perfect_model[:,None])
    plt.plot(np.arange(len(y_test))[:,None], curve_model_1[:,None])
    plt.plot(np.arange(len(y_test))[:,None], curve_no_model[:,None])
    plt.legend(['Perfect model','Model', 'Baseline'])
    plt.xlabel('Number of predictions')
    plt.ylabel('Cumulative number of defaults')
    plt.title('Cumulative curve')
    plt.show()
    print(cumulative_area_ratio)
    return cumulative_area_ratio

