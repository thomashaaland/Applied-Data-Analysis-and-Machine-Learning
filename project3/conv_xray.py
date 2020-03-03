import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import pickle as pk 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau 
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import project3_header as p3h
from project3_header import featureExtraction
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score, auc, roc_curve, classification_report
import os
path =os.getcwd() + "/results/features/"

#parameters for featureExtraction
imSize = (150, 150)
numFiles ='all' #int or 'all' nummber of files from the different sets test/train/val and bacterial pneumonia/viral pneumonia/normal.
load_images = True #for loading images every time numFiles is changed

if load_images:
    Img, y = featureExtraction(methods=['image'], numFiles=numFiles)
else:
    if os.path.isdir(path):
        # import the files
        try:
            Img = pk.load(open(path + "Images.pkl", 'rb'))
            y = pd.read_pickle(path + "FeatureLabels.pkl").values.ravel()
            print(type(y))
        except:
            Img, y = featureExtraction(methods=['image'], numFiles=numFiles)
    else:
        Img, y = featureExtraction(methods=['image'], numFiles=numFiles)

    if Img.shape[1:3] != imSize:
        print('Image shape not as expected, reloading images.')
        Img, y = featureExtraction(methods=['image'], numFiles=numFiles)

y = np.asarray(y)
n_categories = to_categorical(y).shape[1]

ImgTrain, ImgTest, yTrain, yTest = train_test_split(Img, y, test_size=0.2)

filepath="./Oblig3"

lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-8)

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

def create_model(learning_rate = 0.000055, epochs = 50, batch_size = 64):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', padding="same", input_shape= (150,150,1)))
    model.add(Conv2D(16, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
    model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.35))
    model.add(Dense(n_categories , activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=RMSprop(lr=learning_rate),
                      metrics=['accuracy'])
    return model


##################

#score = create_model(0.000055).evaluate(ImgTest, yTest, verbose=0)

model = KerasClassifier(build_fn = create_model)

param_grid = {  
            'epochs' :[100],
            'batch_size' :[60],
            'learning_rate' :np.logspace(-5.5,-4.5,3)
               }

###############################
#gridsearch for best parameters
###############################
grid_search = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs = 1, cv = 5, iid= False, 
                           return_train_score = True, refit = True)

grid_result = grid_search.fit(ImgTrain, yTrain)

cv_results_PD = pd.DataFrame.from_dict(grid_search.cv_results_)
cv_results_PD.to_pickle("./results/CNN_CV_results.pkl")
print(cv_results_PD)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
best_parameters = grid_result.best_params_

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
print(best_parameters.keys())

#retrain model with best parameters(did not get gridsearcv model saving to work)
model = KerasClassifier(build_fn=create_model, 
                        learning_rate=best_parameters['learning_rate'], 
                        epochs=best_parameters['epochs'],
                        batch_size=best_parameters['batch_size'])


'''
##############################
#for only training one model #
##############################
model = KerasClassifier(build_fn=create_model, 
                        epochs=25,
                        learning_rate=0.000055)

'''

history = model.fit(ImgTrain, yTrain, validation_data=(ImgTest, yTest), 
                    callbacks=[lr_reduce, checkpoint])
print(history)
predict = model.predict(ImgTest)
predict_prob = model.predict_proba(ImgTest)


# R2 score:
R2_score = r2_score(yTest, predict)
print("R2 score: ", R2_score)
    
# Accuracy score
score = model.score(ImgTest, yTest)
print("Score: ", score)

print("Confusion matrix: ")
confusionMatrix = confusion_matrix(yTest, predict) # Y: True label; X: Predicted label
print(confusionMatrix)

print("Classification report: ")
print(classification_report(yTest, predict))

p3h.cumulative_plot(yTest, predict_prob[:,1], "cnn_cumulativePlot_vb.png")
print(np.sum(yTest), 'sum yTest')
print('Test accuracy:', score)
plt.figure(0)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("./results/features/validation_accuracy_cnn_vb.png")
plt.show()
# summarize history for loss
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("./results/features/validation_loss_cnn_vb.png")
plt.show()

