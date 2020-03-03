from __future__ import absolute_import, division, print_function, unicode_literals
import glob
import mahotas as mt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import project3_header as p3h
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors.kde import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVC

import tensorflow as tf
from keras.utils import to_categorical


################################
# Unpickle results from RFE.py #

path = "./results/features/"
if os.path.isdir(path):
    # import the files
    try:
        XTrain = pd.read_pickle("./results/features/FeaturesTrain.pkl").values
        
    except:
        print("The file ./results/features/FeaturesTrain.pkl was not found. Have you tried to run RFE.py in current directory?")
        exit()
    try:
        XTest = pd.read_pickle("./results/features/FeaturesTest.pkl").values
    except:
        print("The file ./results/features/FeaturesTest.pkl was not found. Have you tried to run RFE.py in current directory?")
        exit()
    try:
        yTrain = pd.read_pickle("./results/features/FeaturesLabelsTrain.pkl").values.ravel()
    except:
        print("The file ./results/features/FeaturesLabelsTrain.pkl was not found. Have you tried to run RFE.py in current directory?")
        exit()
    try:
        yTest = pd.read_pickle("./results/features/FeaturesLabelsTest.pkl").values.ravel()
    except:
        print("The file ./results/features/FeaturesLabelsTest.pkl was not found. Have you tried to run RFE.py in current directory?")
        exit()

# Correlation Matrix
corrX = np.corrcoef(np.concatenate((XTrain, yTrain.reshape(-1,1)), axis=1).T)
ax = sns.heatmap(corrX)
plt.show()

pdX = pd.DataFrame(data = np.concatenate((XTrain, yTrain.reshape(-1,1)), axis=1),
columns = list(range(72)))
pdX.rename(columns={"71":"Target"}, inplace=True)
corrX = pdX.corr()
                   
ax = sns.heatmap(corrX, xticklabels=10, yticklabels=10)
plt.title("Correlation Matrix for extracted features")
plt.tight_layout()
plt.savefig("./results/finalFeatureCorrMat.png")
plt.show()

############
# Training:
yTrain = yTrain.ravel()
yTest = yTest.ravel()

XTrain, XVal, yTrain, yVal = train_test_split(XTrain, yTrain, test_size=0.2)

print("Starting training")
print("XTrain shape: ", XTrain.shape)
print("yTrain shape: ", yTrain.shape)
print("XVal shape: ", XVal.shape)
print("yVal shape: ", yVal.shape)
print("XTest shape: ", XTest.shape)
print("yTest shape: ", yTest.shape)

#######################
## KERAS             ##
#######################

epoch = 16
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation="relu", input_shape=(XTrain.shape[1],)),
    tf.keras.layers.Dense(200, activation="relu"),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
    ])

model.compile(
    optimizer="adam",
    learning_rate=10**(-240),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
    )

history = model.fit(
    XTrain,
    to_categorical(yTrain),
    validation_data=(XVal, to_categorical(yVal)),
    epochs=epoch,
    #batch_size=32,
)

model.evaluate(
    XTest,
    to_categorical(yTest),
    )



predictions = model.predict(XTest)
print(predictions)
print(np.argmax(predictions, axis=1))
# Confusion matrix
print("Confusion matrix: ")
confusionMatrix = confusion_matrix(yTest, np.argmax(predictions, axis=1)) # Y: True label; X: Predicted label 
print(confusionMatrix)
print("Sensitivity: ", confusionMatrix[0,0]/(confusionMatrix[0,1]+confusionMatrix[0,0])) # True Pos / (True Pos + False Pos)
print("Specificity: ", confusionMatrix[1,1]/(confusionMatrix[1,0]+confusionMatrix[1,1])) # True Neg / (True Neg + False Neg)

# ROC vs AUC i numpy?

# Classification report
print("Classification report: ")
print(classification_report(yTest, np.argmax(predictions, axis=1)))

print(history.history.keys())
epochs = range(1, epoch + 1)
plt.plot(epochs, history.history['accuracy'], label="Train")
plt.plot(epochs, history.history['val_accuracy'], label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.tight_layout()
plt.savefig("./results/NNaccuracy.png")
plt.show()

plt.plot(epochs, history.history['loss'], label="Train")
plt.plot(epochs, history.history['val_loss'], label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss")
plt.tight_layout()
plt.savefig("./results/NNloss.png")
plt.show()

p3h.cumulative_plot(to_categorical(yTest)[:,0], predictions[:,0], "./results/cumulativePlotNormalNN.png", title="NN Normal")
p3h.cumulative_plot(to_categorical(yTest)[:,1], predictions[:,1], "./results/cumulativePlotBacteriaNN.png", title="NN Bacteria")
p3h.cumulative_plot(to_categorical(yTest)[:,2], predictions[:,2], "./results/cumulativePlotVirusNN.png", title="NN Virus")
