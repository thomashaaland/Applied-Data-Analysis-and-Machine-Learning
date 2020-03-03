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



path = "./results/features/"
if os.path.isdir(path):
    # import the files
    try:
        X = pd.read_pickle("./results/features/Features.pkl").values
    except:
        print("The file ./results/features/Features.pkl was not found. Have you tried to run featureExtract.py in current directory?")
        exit()
    try:
        y = pd.read_pickle("./results/features/FeatureLabels.pkl").values.ravel()
    except:
        print("The file ./results/features/FeatureLabels.pkl was not found. Have you tried to run featureExtract.py in current directory?")
        exit()
        
print("The shape of X and y: ", X.shape, y.shape)
# Now shuffle the datasets
print("Splitting and shuffling dataset")

# Produce 
XTrain, XTest, yTrain, yTest = train_test_split(X, y.ravel(), test_size=0.25, shuffle = True)

#rescale:
qt = QuantileTransformer()
print("Fitting scaler")
qt.fit(XTrain)
print("Applying fit")
XTrain = qt.transform(XTrain)
XTest = qt.transform(XTest)
print("Scaling Complete")

correlatedFeatures = set()
correlationMatrix = pd.DataFrame(data=XTrain).corr()
plt.imshow(correlationMatrix)
plt.show()

# Recursive feature elimination
print("Recursive feature elimination")
svc = SVC(C=1, kernel="linear")
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(3), scoring="accuracy", verbose=4)
rfecv.fit(XTrain, yTrain)
XTrain = rfecv.transform(XTrain)
XTest = rfecv.transform(XTest)
print("Optimal number of features : %d" % rfecv.n_features_)
print("The shape of XTrain and yTrain after feature selection :", XTrain.shape, yTrain.shape)
XTraindf = pd.DataFrame(data=XTrain)
XTestdf= pd.DataFrame(data=XTest)
yTraindf = pd.DataFrame(data=yTrain)
yTestdf = pd.DataFrame(data=yTest)
XTraindf.to_pickle("./results/features/FeaturesTrain.pkl")
XTestdf.to_pickle("./results/features/FeaturesTest.pkl")
yTraindf.to_pickle("./results/features/FeaturesLabelsTrain.pkl")
yTestdf.to_pickle("./results/features/FeaturesLabelsTest.pkl")

#Plot number of features VS Cross Validation score
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross Validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.tight_layout()
plt.savefig("./results/features/rfecv_curve.png")
plt.show()
