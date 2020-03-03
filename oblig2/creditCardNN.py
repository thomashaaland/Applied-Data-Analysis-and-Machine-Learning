import homeBrewNN as hb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, auc, roc_curve
import numpy as np
import random
import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from scipy.optimize import fmin_tnc
from sklearn.decomposition import PCA
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from keras.utils import to_categorical
from keras.optimizers import adam
import os

# Trying to set the seed
random.seed(0)

# Reading file into data frame
nanDict = {}

cwd = os.getcwd()
filename = cwd + "/default of credit card clients.xls"

df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
df_pay = pd.read_excel('default of credit card clients.xls', header=1, skiprows=0, usecols = 'G:L, Y', index_col=0, na_values=nanDict)

print(df_pay)

df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)
df_pay.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

# Remove instances with zeros only for past bill statements or paid amounts:
df = df.drop(df[(df.BILL_AMT1 == 0) &
                (df.BILL_AMT2 == 0) &
                (df.BILL_AMT3 == 0) &
                (df.BILL_AMT4 == 0) &
                (df.BILL_AMT5 == 0) &
                (df.BILL_AMT6 == 0)].index)
df = df.drop(df[(df.PAY_AMT1 == 0) &
                (df.PAY_AMT2 == 0) &
                (df.PAY_AMT3 == 0) &
                (df.PAY_AMT4 == 0) &
                (df.PAY_AMT5 == 0) &
                (df.PAY_AMT6 == 0)].index)
df = df.drop(df[(df.EDUCATION == 0) |
                (df.EDUCATION == 5) |
                (df.EDUCATION == 6)].index)
df = df.drop(df[(df.MARRIAGE == 0)].index)

# Features and targets
X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values
X_pay = df_pay.loc[:, df_pay.columns != 'defaultPaymentNextMonth'].values
y_pay = df_pay.loc[:, df_pay.columns == 'defaultPaymentNextMonth'].values
target = to_categorical(y)
onehotencoder = OneHotEncoder(categories="auto")

df.head()
correlation_matrix = df.corr().round(2)
plt.figure(0)
sns.set(font_scale=0.5)
sns.heatmap (data= correlation_matrix, linewidths=0.1, annot = True,  annot_kws={"size":6})
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()
#hard code Sex and Marrital status, should also take maybe education
X = ColumnTransformer(
    [("", onehotencoder, [1,2,3,5,6,7,8,9,10]),],
    remainder="passthrough"
).fit_transform(X).A
print('X',X)
X_pay = ColumnTransformer(
    [("", onehotencoder, [0,1,2,3,4]),],
    remainder="passthrough"
).fit_transform(X_pay).A
print('X_pay', X_pay)
y = ColumnTransformer(
    [("", onehotencoder, [0]),],
    remainder="passthrough"
).fit_transform(y)
print('y_shape',y.shape)
print(y[:5])
col_to_norm = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
              'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

df[col_to_norm]=df[col_to_norm].apply(lambda x: (x-np.mean(x))/np.std(x))
#Train-test split for pay
trainingShare = 0.5 
seed  = 2
XTrain_pay, XTest_pay, yTrain_pay, yTest_pay =train_test_split(X_pay, y_pay, train_size=trainingShare, \
                                              test_size = 1-trainingShare,
                                             random_state=seed)



#Train-test split
trainingShare = 0.5 
seed  = 1
XTrain, XTest, yTrain, yTest=train_test_split(X, y, train_size=trainingShare, \
                                              test_size = 1-trainingShare,
                                             random_state=seed)
print('xtrain', XTrain.shape)

X = StandardScaler(with_mean = False).fit_transform(X)
sc = StandardScaler(with_mean = False)
XTrain = sc.fit_transform(XTrain)
XTest = sc.transform(XTest)
XTrain_pay = sc.fit_transform(XTrain_pay)
XTest_pay = sc.transform(XTest_pay)
print(sc.fit_transform(yTest))

lambdas=np.logspace(-5,7,13)
parameters = [{'C': 1./lambdas, "solver":["lbfgs"]}]#*len(parameters)}]
scoring = ['accuracy', 'roc_auc']
logReg = LogisticRegression()
logReg_pay = LogisticRegression()
logReg_pay.fit(XTrain_pay, yTrain_pay)
y_pred_pay = logReg_pay.predict(XTest_pay)
print('y_pred_pay', y_pred_pay.shape)
print('Accuracy of scikit learn logistic regression classifier on test set only for pay: {:.4f}'.\
      format(logReg_pay.score(XTest_pay, yTest_pay)))

alpha = 0.1 #learning reate
_lambda = 160
lambda_tests = np.logspace(1, 3, num = 8)
alpha_tests = np.logspace(-1.5,0, num = 6)
max_iter = 1000
k = 10 #for k-folds cv

sns.set(style="white", context="notebook", font_scale=1.5, 
            rc={"axes.grid": True, "legend.frameon": False,
"lines.markeredgewidth": 1.4, "lines.markersize": 10})
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 4.5})

model = hb.NNhomebrew([hb.layers(features = XTrain, layerType = "featureLayer"),
                       hb.layers(outputNodes = 32, activation = "ReLu"),
                       hb.layers(outputNodes = 2, activation = "softMax"),])
model.compile(loss = "crossEntropy", optimizer = 'gradientDecent', regularization = 'L2', _lambda = 0, weightNormalizing = True)
model.fit(yTrain, validation_target = yTest, validation_data = XTest, learningRate = 1, epochs = 1000, backPropagation = 'normal')
prediction = model.predict(XTest)

predictOutcomes = prediction.copy()
predictOutcomes[prediction >= 0.5] = 1
predictOutcomes[prediction < 0.5] = 0


print("Accuracy score on test set and NN: ", accuracy_score(yTest, predictOutcomes))

hb.cumulative_plot(yTest[:,1],prediction[:,1],title = 'Cumulative Gains Curve')
plt.figure(1)
plt.plot(yTrain[:,1], '.')
plt.plot(prediction[:,1], '.')
plt.show()

## store the models for later use
#DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

# grid search
#for i, eta in enumerate(eta_vals):
#    for j, lmbd in enumerate(lmbd_vals):
#        dnn = NeuralNetwork(XTrain, yTrain, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
#                            n_hidden_neurons=20, n_categories=2)
#        dnn.train()
#        
#        DNN_numpy[i][j] = dnn
#        
#        test_predict = dnn.predict(XTest)
#        
#        print("Learning rate  = ", eta)
#        print("Lambda = ", lmbd)
#        print("Accuracy score on test set: ", accuracy_score(yTest, test_predict))
#        print()

'''
n_cols = X.shape[1]
early_stopping_monitor = EarlyStopping(patience=2)
model = Sequential()
model.add(Dense(32, activation='relu', input_shape = (n_cols,)))
model.add(Dense(16, activation='relu'))
#model.add(Dense(20, activation='relu'))
model.add(Dense(2, activation='softmax'))
opt = adam(lr=0.01, decay=0.01)
model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
model.fit(X, target, epochs=20, validation_split=0.3, callbacks = [early_stopping_monitor])

'''
