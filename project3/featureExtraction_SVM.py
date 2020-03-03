import glob
import matplotlib.pyplot as plt
import seaborn as sns
import project3_header as p3h
import numpy as np
import mahotas as mt
import pandas as pd
import os
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.svm import SVC
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# pca_inspection toggled on or off
# PCA inspect makes plots of the two first principal components
# eigen_Xray_images displays image of the first four eigen vectors
pca_inspect = False
eigen_Xray_images = False
image_inspect = True

# For finding the best bandwidth or PCA n_components
find_best_bandwidth = False
find_best_n_components = False

# Toggle this on to find the best combination of features
find_best_combo = False
##############################
## Collect files            ##
##############################

# Paths to the six different repositories
print("Setting up paths")
pathTrainNORMAL = "./dataset/chest_xray/train/NORMAL"
pathTrainPNEUMONIA = "./dataset/chest_xray/train/PNEUMONIA"
pathValNORMAL ="./dataset/chest_xray/val/NORMAL"
pathValPNEUMONIA ="./dataset/chest_xray/val/PNEUMONIA"
pathTestNORMAL ="./dataset/chest_xray/test/NORMAL"
pathTestPNEUMONIA ="./dataset/chest_xray/test/PNEUMONIA"

# Collect all the filenames
print("Collecting filenames")
filenamesTrainNORMAL = glob.glob(pathTrainNORMAL + "/*.jpeg")
filenamesTrainPNEUMONIA = glob.glob(pathTrainPNEUMONIA + "/*.jpeg")

filenamesValNORMAL = glob.glob(pathValNORMAL + "/*.jpeg")
filenamesValPNEUMONIA = glob.glob(pathValPNEUMONIA + "/*.jpeg")

filenamesTestNORMAL = glob.glob(pathTestNORMAL + "/*.jpeg")
filenamesTestPNEUMONIA = glob.glob(pathTestPNEUMONIA + "/*.jpeg")

# Split pneumonia between bacterial and viral
print("Splitting between bacterial and viral")
filenamesTrainBact = list(filter(lambda bact: "bacteria" in bact, filenamesTrainPNEUMONIA))
filenamesTrainVir = list(filter(lambda vir: "virus" in vir, filenamesTrainPNEUMONIA))

filenamesValBact = list(filter(lambda bact: "bacteria" in bact, filenamesValPNEUMONIA))
filenamesValVir = list(filter(lambda vir: "virus" in vir, filenamesValPNEUMONIA))

filenamesTestBact = list(filter(lambda bact: "bacteria" in bact, filenamesTestPNEUMONIA))
filenamesTestVir = list(filter(lambda vir: "virus" in vir, filenamesTestPNEUMONIA))


imSize = (150, 150) # 300 * 250
numFiles = "all"
n_components = 4
bandwidth = 0.5 #0.16595869
methods = ["pca", "kde", "haralick", "image"]
#methods = ["pca", "kde", "haralick"]
#methods = ["pca", "kde"]
#methods = ["image"]
#methods = ["kde"]
print("Loading train images")
ImgTrain0, XTrain0, yTrain0 = p3h.accrueFiles(filenamesTrainNORMAL,
                                     numFiles=numFiles,
                                     key=0,
                                     n_components=n_components,
                                     imSize=imSize,
                                     methods=methods,
                                     bandwidth=bandwidth)
ImgTrain1, XTrain1, yTrain1 = p3h.accrueFiles(filenamesTrainBact,
                                     numFiles=numFiles,
                                     key=1,
                                     n_components=n_components,
                                     imSize=imSize,
                                     methods=methods,
                                     bandwidth=bandwidth)
ImgTrain2, XTrain2, yTrain2 = p3h.accrueFiles(filenamesTrainVir,
                                     numFiles=numFiles,
                                     key=2,
                                     n_components=n_components,
                                     imSize=imSize,
                                     methods=methods,
                                     bandwidth=bandwidth)

print("Loading validation images")
ImgVal0, XVal0, yVal0 = p3h.accrueFiles(filenamesValNORMAL,
                                 key=0,
                                 n_components=n_components,
                                 imSize=imSize,
                                 methods=methods,
                                 bandwidth=bandwidth)
ImgVal1, XVal1, yVal1 = p3h.accrueFiles(filenamesValBact,
                                 key=1,
                                 n_components=n_components,
                                 imSize=imSize,
                                 methods=methods,
                                 bandwidth=bandwidth)
ImgVal2, XVal2, yVal2 = p3h.accrueFiles(filenamesValVir,
                                 key=2,
                                 n_components=n_components,
                                 imSize=imSize,
                                 methods=methods,
                                 bandwidth=bandwidth)

print("Loading test images")
ImgTest0, XTest0, yTest0 = p3h.accrueFiles(filenamesTestNORMAL,
                                   numFiles=numFiles,
                                   key=0,
                                   n_components=n_components,
                                   imSize=imSize,
                                   methods=methods,
                                   bandwidth=bandwidth)
ImgTest1, XTest1, yTest1 = p3h.accrueFiles(filenamesTestBact,
                                   numFiles=numFiles,
                                   key=1,
                                   n_components=n_components,
                                   imSize=imSize,
                                   methods=methods,
                                   bandwidth=bandwidth)
ImgTest2, XTest2, yTest2 = p3h.accrueFiles(filenamesTestVir,
                                   numFiles=numFiles,
                                   key=2,
                                   n_components=n_components,
                                   imSize=imSize,
                                   methods=methods,
                                   bandwidth=bandwidth)


X = np.array(XTrain0 + XTrain1 + XTrain2 + XVal0 + XVal1 + XVal2 + XTest0 + XTest1 + XTest2)
y = np.array(yTrain0 + yTrain1 + yTrain2 + yVal0 + yVal1 + yVal2 + yTest0 + yTest1 + yTest2)

path = "./results/features/"

if eigen_Xray_images == True:
    n_components = 4
    Img = np.array(ImgTrain0 +
                   ImgTrain1 +
                   ImgTrain2 +
                   ImgVal0 +
                   ImgVal1 +
                   ImgVal2 +
                   ImgTest0 +
                   ImgTest1 +
                   ImgTest2)
    pca = PCA(n_components = n_components)
    pca.fit(Img/255. - 0.5)
    pcaComponents = pca.components_.reshape(n_components, imSize[0], imSize[1])

    fig, axs = plt.subplots(2,2)
    axs[0,0].imshow(pcaComponents[0], cmap='plasma')
    axs[0,1].imshow(pcaComponents[1], cmap='plasma')
    axs[1,0].imshow(pcaComponents[2], cmap='plasma')
    axs[1,1].imshow(pcaComponents[3], cmap='plasma')
    plt.tight_layout()
    plt.savefig("./results/eigenImages.png", bbox_inches='tight')
    plt.show()

    exit()

# This section finds the best bandwidth for kde
if find_best_bandwidth == True:
    Img = np.array(ImgTrain0 +
                   ImgTrain1 +
                   ImgTrain2 +
                   ImgVal0 +
                   ImgVal1 +
                   ImgVal2 +
                   ImgTest0 +
                   ImgTest1 +
                   ImgTest2)

    plt.imshow(Img[0].reshape(imSize))
    plt.show()
    svc = SVC(gamma='scale')
    bandwidths = np.logspace(-3, 3, 101)
    scores = []
    for bandwidth in bandwidths:
        print("Working on Bandwidth: ", bandwidth)
        kde = []
        for i, image in enumerate(Img):
            kde.append(p3h.extractFeatures(image.reshape(imSize), methods=["kde"], bandwidth=bandwidth, n_components=n_components))
            print("Treating image ", i, end="\r")
        kdeTrain, kdeTest, yTrain, yTest = train_test_split(kde, y, test_size=0.25, shuffle=True)
        svc.fit(kdeTrain, yTrain)
        scores.append(svc.score(kdeTest, yTest))
        print(svc.score(kdeTest, yTest))
    scores = np.array(scores)
    bestScore = scores[np.max(scores) == scores]
    bestBandWidth = bandwidths[np.max(scores) == scores]
    print("Best score: ", bestScore, "Best bandwidth: ", bestBandWidth)
    plt.semilogx(bandwidths, scores)
    plt.xlabel("Bandwidth")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("./results/accBandwidth.png")
    plt.show()
        
    exit()

# This section finds the best number of components for pca
if find_best_n_components == True:
    Img = np.array(ImgTrain0 +
                   ImgTrain1 +
                   ImgTrain2 +
                   ImgVal0 +
                   ImgVal1 +
                   ImgVal2 +
                   ImgTest0 +
                   ImgTest1 +
                   ImgTest2)

    plt.imshow(Img[0].reshape(imSize))
    plt.show()
    svc = SVC(gamma='scale')
    n_components_list = np.arange(1, 21)
    scores = []
    for n_components in n_components_list:
        print("Working on ", n_components, " components")
        pca = []
        for i, image in enumerate(Img):
            pca.append(p3h.extractFeatures(image.reshape(imSize), methods=["pca"], n_components=n_components))
            print("Treating image ", i, end="\r")
        pcaTrain, pcaTest, yTrain, yTest = train_test_split(pca, y, test_size=0.25, shuffle=True)
        svc.fit(pcaTrain, yTrain)
        scores.append(svc.score(pcaTest, yTest))
        print(svc.score(pcaTest, yTest))
    scores = np.array(scores)
    bestScore = scores[np.max(scores) == scores]
    best_n_components = n_components_list[np.max(scores) == scores]
    print("Best score: ", bestScore, "Best number of components: ", best_n_components)
    plt.plot(n_components_list, scores)
    plt.xlabel("n Components")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("./results/accNumComponents.png")
    plt.show()
        
    exit()

# This section finds the best combination of features
if find_best_combo == True:
    Img = np.array(ImgTrain0 +
                   ImgTrain1 +
                   ImgTrain2 +
                   ImgVal0 +
                   ImgVal1 +
                   ImgVal2 +
                   ImgTest0 +
                   ImgTest1 +
                   ImgTest2)

    ss = StandardScaler()
    pt = QuantileTransformer() #PowerTransformer()
    svc = SVC(gamma='scale')
    methodsList = [['pca'],
                   ['kde'],
                   ['haralick'],
                   ['image'],
                   ['pca','kde'],
                   ['pca','haralick'],
                   ['pca','image'],
                   ['kde','haralick'],
                   ['kde','image'],
                   ['haralick','image'],
                   ['pca','kde','haralick'],
                   ['pca','kde','image'],
                   ['kde','haralick','image'],
                   ['pca','kde','haralick','image']]
    scores = []
    scoresSS = []
    scoresPT = []
    errors = []
    errorsSS = []
    errorsPT = []
    for methods in methodsList:
        print("Working on methods: ", methods)
        features = []
        for i, image in enumerate(Img):
            features.append(p3h.extractFeatures(image.reshape(imSize), methods=methods, bandwidth=bandwidth, n_components=n_components))
            print("Treating image ", i, end="\r")
        # Cross validation using K-Fold
        features = np.array(features)
        kf = KFold(n_splits=5, shuffle = True)
        kFoldScores = []
        kFoldScoresSS = []
        kFoldScoresPT = []
        for train_index, test_index in kf.split(features):
            featuresTrain, featuresTest = features[train_index], features[test_index]
            yTrain, yTest = y[train_index], y[test_index]

            svc.fit(featuresTrain, yTrain)
            kFoldScores.append(svc.score(featuresTest, yTest))
            print("The accuracy of just these features: ", svc.score(featuresTest, yTest))
            
            svc.fit(ss.fit_transform(featuresTrain), yTrain)
            scoreSS = svc.score(ss.fit_transform(featuresTest), yTest)
            kFoldScoresSS.append(scoreSS)
            print("The accuracy after StandardScaler: ", scoreSS)
            
            svc.fit(pt.fit_transform(featuresTrain), yTrain)
            scorePT = svc.score(pt.fit_transform(featuresTest), yTest)
            kFoldScoresPT.append(scorePT)
            print("The accuracy after QuantileTransformer: ", scorePT)
            
        kFoldScores = np.array(kFoldScores)
        scores.append(np.mean(kFoldScores))
        errors.append(np.sqrt(np.var(kFoldScores))*2/np.sqrt(5))

        kFoldScoresSS = np.array(kFoldScoresSS)
        scoresSS.append(np.mean(kFoldScoresSS))
        errorsSS.append(np.sqrt(np.var(kFoldScoresSS))*2/np.sqrt(5))

        kFoldScoresPT = np.array(kFoldScoresPT)
        scoresPT.append(np.mean(kFoldScoresPT))
        errorsPT.append(np.sqrt(np.var(kFoldScoresPT))*2/np.sqrt(5))

    scores = np.array(scores)
    errors = np.array(errors)
    bestScore = scores[np.max(scores) == scores]
    bestCombo = methodsList[np.arange(0,len(methodsList))[np.max(scores) == scores][0]]
    print("Best score: ", bestScore, "Best combo: ", bestCombo)

    scoresSS = np.array(scoresSS)
    errorsSS = np.array(errorsSS)
    bestScoreSS = scoresSS[np.max(scoresSS) == scoresSS]
    bestComboSS = methodsList[np.arange(0,len(methodsList))[np.max(scoresSS) == scoresSS][0]]
    print("Best score: ", bestScoreSS, "Best combo: ", bestComboSS)

    scoresPT = np.array(scoresPT)
    errorsPT = np.array(errorsPT)
    bestScorePT = scoresPT[np.max(scoresPT) == scoresPT]
    bestComboPT = methodsList[np.arange(0,len(methodsList))[np.max(scoresPT) == scoresPT][0]]
    print("Best score: ", bestScorePT, "Best combo: ", bestComboPT)

    x = range(len(methodsList))
    my_xticks = ['pca',
                 'kde',
                 'haralick',
                 'image',
                 'pca, kde',
                 'pca, haralick',
                 'pca, image',
                 'kde, haralick',
                 'kde, image',
                 'pca, kde, haralick',
                 'pca, kde, image',
                 'kde, haralick, image',
                 'pca, kde, haralick, image']
    plt.xticks(x, my_xticks, rotation=45)
    plt.errorbar(x, scores, errors, label="No feature scaling")
    plt.errorbar(x, scoresSS, errorsSS, label="StandardScaler")
    plt.errorbar(x, scoresPT, errorsPT, label="QuantileTransformer")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./results/accBestFeatures.png")
    plt.show()
        
    exit()

    
############################
## Quick look at PCA      ##
############################
if pca_inspect == True:
    
    Img = np.array(ImgTrain0 +
                   ImgTrain1 +
                   ImgTrain2 +
                   ImgVal0 +
                   ImgVal1 +
                   ImgVal2 +
                   ImgTest0 +
                   ImgTest1 +
                   ImgTest2)

    categories = ["Normal", "Bacterial", "Viral"]
    ss = QuantileTransformer()
    
    pca = PCA(2)
    projected = pca.fit_transform(Img/255 - 0.5)
    plt.scatter(projected[:,0], projected[:,1],
                c=y,edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('plasma', 3))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title("PCA analysis on image")
    cb = plt.colorbar()
    loc = np.arange(0, max(y), max(y)/float(len(categories)))
    cb.set_ticks(loc)
    cb.set_ticklabels(categories)
    plt.tight_layout()
    plt.savefig(path + "PCAplot.png")
    plt.show()

    print("Before scaling:")
    print(np.max(Img), np.min(Img))
    #ss = StandardScaler()
    Img = ss.fit_transform(Img)
    print("After scaling:")
    print(np.max(Img), np.min(Img))

    
    pca = PCA(2)
    projected = pca.fit_transform(Img/255 - 0.5)
    plt.scatter(projected[:,0], projected[:,1],
                c=y,edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('plasma', 3))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title("PCA analysis on image with scaling")
    cb = plt.colorbar()
    loc = np.arange(0, max(y), max(y)/float(len(categories)))
    cb.set_ticks(loc)
    cb.set_ticklabels(categories)
    plt.tight_layout()
    plt.savefig(path + "PCAplotScale.png")
    plt.show()

    
    projected2 = pca.fit_transform(X)
    plt.scatter(projected2[:,0], projected2[:,1],
                c=y,edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('plasma', 3))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title("PCA analysis on extracted features")
    cb = plt.colorbar()
    loc = np.arange(0, max(y), max(y)/float(len(categories)))
    cb.set_ticks(loc)
    cb.set_ticklabels(categories)
    plt.tight_layout()
    plt.savefig(path + "PCAplotOnFeatures.png")
    plt.show()

    print("Before scaling:")
    print(np.max(X), np.min(X))
    X = ss.fit_transform(X)
    print("After scaling:")
    print(np.max(X), np.min(X))

    projected2 = pca.fit_transform(X)
    plt.scatter(projected2[:,0], projected2[:,1],
                c=y,edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('plasma', 3))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title("PCA analysis on extracted features after scaling")
    cb = plt.colorbar()
    loc = np.arange(0, max(y), max(y)/float(len(categories)))
    cb.set_ticks(loc)
    cb.set_ticklabels(categories)
    plt.tight_layout()
    plt.savefig(path + "PCAplotOnFeaturesScaled.png")
    plt.show()

print("X and y shape: ", X.shape, y.shape)

pdX = pd.DataFrame(data=X)
pdY = pd.DataFrame(data=y)


if os.path.exists(path):
    if os.path.exists(path + "Features.pkl"):
        os.remove(path + "Features.pkl")
    if os.path.exists(path + "FeatureLabels.pkl"):
        os.remove(path + "FeatureLabels.pkl")
    try:
        pdX.to_pickle(path + "Features.pkl")
    except:
        print("Couldn't write " + path + "Features.pkl to drive")
    try:
        pdY.to_pickle(path + "FeatureLabels.pkl")
    except:
        print("Couldn't write " + path + "FeatureLabels.pkl to drive")
else:
    try:
        os.mkdir(path)
    except:
        print("Creation of the path " + path + " failed.")
    else:
        print("Successfully created the directory " + path)
    try:
        pdX.to_pickle(path + "Features.pkl")
    except:
        print("Couldn't write " + path + "Features.pkl to drive")
    try:
        pdY.to_pickle(path + "FeatureLabels.pkl")
    except:
        print("Couldn't write " + path + "FeatureLabels.pkl to drive")
