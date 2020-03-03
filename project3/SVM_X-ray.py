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
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import tensorflow as tf
#import keras
#from keras.layers import Dense
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


############
# Training:
yTrain = yTrain.ravel()
yTest = yTest.ravel()

print("Starting training")
print("XTrain shape: ", XTrain.shape)
print("yTrain shape: ", yTrain.shape)
print("XTest shape: ", XTest.shape)
print("yTest shape: ", yTest.shape)

# Cross validation
param_grid = [
    {'C': np.logspace(-4, 2, 21), # search grid: -4, 2, 21
     'kernel': ['linear']},
    {'C': np.logspace(0, 3, 21), #[1, 10, 100, 1000],
     'gamma': np.logspace(-5,0, 21), # search grid: -5, 0, 51
     'kernel': ['rbf']},
    {'C': np.logspace(-4, 2, 21), # search grid: -4, 2, 51
     'gamma': np.logspace(-4, 0, 21), # search grid: -4, 0, 51
     'degree': [3],
     'coef0': np.logspace(-1, 1, 21), # search grid: -1, 1, 31
     'kernel': ['poly']},
    {'C': np.logspace(-3, 1, 21), # search grid: -3, 1, 41
     'coef0': np.logspace(-1, 1, 21), # search grid: -1, 1, 31
     'gamma': np.logspace(-3, -1, 21), # search grid: -3, -1, 31
     'kernel': ['sigmoid'],
    }
]

clf = GridSearchCV(SVC(probability=False), param_grid, cv=3, verbose=5, n_jobs=4, iid=False, return_train_score=True, refit=True)
clf.fit(XTrain, yTrain)
cv_results_PD = pd.DataFrame.from_dict(clf.cv_results_)
cv_results_PD.to_pickle("./results/CV_results.pkl")
print(cv_results_PD)

# Look at the best model
best_params = clf.best_params_
print("Best params", best_params)

kernel = best_params['kernel']
C = best_params['C']
if kernel == 'linear':
    clf_best = SVC(probability=True, C=C, kernel=kernel)
elif kernel == 'rbf':
    gamma = best_params['gamma']
    clf_best = SVC(probability=True, C=C, kernel=kernel, gamma=gamma)
elif kernel == 'poly':
    gamma = best_params['gamma']
    coef0 = best_params['coef0']
    degree = best_params['degree']
    clf_best = SVC(probability=True, C=C, kernel=kernel, gamma=gamma, coef0=coef0, degree=degree)
elif kernel == 'sigmoid':
    gamma = best_params['gamma']
    coef0 = best_params['coef0']
    clf_best = SVC(probability=True, C=C, kernel=kernel, gamma=gamma, coef0=coef0)

# Predictors and errors:
#clf_best = SVC(probability=True), best_params)
clf_best.fit(XTrain, yTrain)
predict = clf_best.predict(XTest)
predict_prob = clf_best.predict_proba(XTest)
    
# R2 score:
R2_score = r2_score(yTest, predict)
print("R2 score: ", R2_score)
    
# Accuracy score
score = clf_best.score(XTest, yTest)
print("Score: ", score)
    
# Confusion matrix
print("Confusion matrix: ")
confusionMatrix = confusion_matrix(yTest, predict) # Y: True label; X: Predicted label 
print(confusionMatrix)
print("Sensitivity: ", confusionMatrix[0,0]/(confusionMatrix[0,1]+confusionMatrix[0,0])) # True Pos / (True Pos + False Pos)
print("Specificity: ", confusionMatrix[1,1]/(confusionMatrix[1,0]+confusionMatrix[1,1])) # True Neg / (True Neg + False Neg)

# ROC vs AUC i numpy?

# Classification report
print("Classification report: ")
print(classification_report(yTest, predict))

p3h.cumulative_plot(to_categorical(yTest)[:,0], predict_prob[:,0], "./results/cumulativePlotNormal.png", title="Normal")
p3h.cumulative_plot(to_categorical(yTest)[:,1], predict_prob[:,1], "./results/cumulativePlotBacteria.png", title="Bacteria")
p3h.cumulative_plot(to_categorical(yTest)[:,2], predict_prob[:,2], "./results/cumulativePlotVirus.png", title="Virus")

