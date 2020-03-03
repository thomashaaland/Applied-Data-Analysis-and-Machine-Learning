#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:05:23 2019

@author: houjie
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import logreg as ph

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



#####################################
## Read in the appropriate dataset ##
#####################################
## Selecting creditcard example    ##
#####################################


cwd = os.getcwd()
filename = cwd + "/default of credit card clients.xls"
nanDict = {}
df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)

df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

# Features and targets
X = df.loc[:, df.columns != "defaultPaymentNextMonth"].values
y = df.loc[:, df.columns == "defaultPaymentNextMonth"].values

# Looking at the data
# Look at the first 5 datapoints
print(df[:5])

for i, col in enumerate(df.columns):
    print(i, col)


headers = df.columns
#print(headers[5:11])

# Sort out the different types of data, looking for
# Amount given in credit (0)
# social info (1-4) [gender, education, marital status, age]
# History of past payment (5-10): how many months have client delayed
# Amount owed (11-16)
# Amount of previous payment: (17-22)

onehotencoder = OneHotEncoder(categories="auto")

X = ColumnTransformer(
    [("", onehotencoder, [1, 2, 3]),],
    remainder="passthrough"
).fit_transform(X)

# Normalize X:
X = X.astype(float)[:]/np.max(X[:])


#print(X.shape)

def accuracy(p, y):
    return (np.sum(np.round(np.round(p).ravel())==y.ravel()))/len(p)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

##########################
## My own class method  ##
##########################

mySolver = ph.LogisticRegression()
mySolver.fit(X, y)

mySolver2 = ph.LogisticRegression(epochs = 10, solver = "steepest descent", learning_rate = 0.005)
mySolver2.fit(X,y)

mySolver_sgd = ph.LogisticRegression(epochs=5, solver = "sgd")
mySolver_sgd.fit(X,y)

########################
## Scikitlearn method ##
########################

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='lbfgs').fit(X,y.ravel())

############################
## Outputting the results ##
############################

print("SKLearn predict: ", accuracy(clf.predict(X), y))
print("mySolver using Newton Raphson predict: ", accuracy(mySolver.predict(X), y))
print("mySolver using gradient descent predict: ", accuracy(mySolver2.predict(X), y))
print("mySolver stochastic gradient descent predict: ", accuracy(mySolver_sgd.predict(X), y))
print("Naive predict with zeros: ", accuracy(np.zeros(len(y)), y))
print("Naive predict with ones: ", accuracy(np.ones(len(y)), y))
print("Naive predict with random 0 or 1: ", accuracy(np.random.rand(len(y)), y))
