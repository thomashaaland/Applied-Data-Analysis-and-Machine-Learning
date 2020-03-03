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
        X = pd.read_pickle("./results/features/Features.pkl").values
    except:
        print("The file ./results/features/Features.pkl was not found. Have you tried to run RFE.py in current directory?")
        exit()
    try:
        y = pd.read_pickle("./results/features/FeatureLabels.pkl").values.ravel()
    except:
        print("The file ./results/features/FeaturesLabels.pkl was not found. Have you tried to run RFE.py in current directory?")
        exit()

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2)

############
# Training:
yTrain = yTrain.ravel()
yTest = yTest.ravel()

# Rescale:
qt = QuantileTransformer()
qt.fit(XTrain)
XTrain = qt.transform(XTrain)
XTest = qt.transform(XTest)

print("XTrain max: ", np.max(XTrain))

"""
# LDA:
lda = LDA()
lda.fit(XTrain, yTrain)
XTrain = lda.transform(XTrain)
XTest = lda.transform(XTest)
"""

print("Starting training")
print("XTrain shape: ", XTrain.shape)
print("yTrain shape: ", yTrain.shape)
print("XTest shape: ", XTest.shape)
print("yTest shape: ", yTest.shape)

# Cross validation
param_grid = [
    {'C': np.logspace(-4, -1, 21), # search grid: -4, 2, 21
     'kernel': ['linear']},
]

clf = GridSearchCV(SVC(probability=False), param_grid, cv=3, verbose=10, n_jobs=4, iid=False, return_train_score=True, refit=True)
clf.fit(XTrain, yTrain)
cv_results_PD = pd.DataFrame.from_dict(clf.cv_results_)
cv_results_PD.to_pickle("./results/CV_Image_results.pkl")
print(cv_results_PD)

# Look at train and test curves
# Linear:

cv_results_PD_linear = cv_results_PD.loc[cv_results_PD["param_kernel"] == "linear"]
CHist = cv_results_PD_linear["param_C"]
trainHist = cv_results_PD_linear["mean_train_score"]
trainHistError = cv_results_PD_linear["std_train_score"]
testHist = cv_results_PD_linear["mean_test_score"]
testHistError = cv_results_PD_linear["std_train_score"]
plt.semilogx(CHist, trainHist, label="Train")
#plt.fill_between(CHist, trainHist-trainHistError, trainHist+trainHistError, alpha=0.5)
plt.semilogx(CHist, testHist, label="Test")
#plt.fill_between(CHist, testHist-testHistError, testHist+testHistError, alpha=0.5)
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Linear SVM")
plt.savefig("./results/SVMLinearCV_Image.png")
plt.show()


# Look at the best model
best_params = clf.best_params_
print("Best params", best_params)

kernel = best_params['kernel']
C = best_params['C']
clf_best = SVC(probability=True, C=C, kernel=kernel)



"""

#Best params from previously
kernel = 'rbf'
C = 31.62
gamma = 0.0178
clf_best = SVC(probability=True, C=C, kernel=kernel, gamma=gamma)
"""
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

p3h.cumulative_plot(to_categorical(yTest)[:,0], predict_prob[:,0], "./results/cumulativePlotNormal_Image.png", title="Normal")
p3h.cumulative_plot(to_categorical(yTest)[:,1], predict_prob[:,1], "./results/cumulativePlotBacteria_Image.png", title="Bacteria")
p3h.cumulative_plot(to_categorical(yTest)[:,2], predict_prob[:,2], "./results/cumulativePlotVirus_Image.png", title="Virus")

