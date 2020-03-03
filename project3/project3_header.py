import glob
import seaborn as sn
import os
from PIL import Image
import numpy as np
import pickle as pk 
import pandas as pd
import mahotas as mt
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.svm import SVC
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from distutils.util import strtobool


def featureExtraction(methods=['pca','kde'], numFiles='all', imSize=(150, 150), write_to_disk=True):

    n_components = 20
    bandwidth = 61

    bact_viral_diff = False
    three_classes = False
    # pca_inspection toggled on or off
    print('look at viral-, bacterial- pneumonia and normal in three different classes [y/n]?')
    three_classes = bool(strtobool(input())) #have viral, bacterial and pneumonia in three different classes
    if three_classes:
        print('Look at only positive pneumonia to differnentiate between bacterial or viral [y/n]?')
        bact_viral_diff = bool(strtobool(input()))  #only look at the images of patients with pneumonia to try to 
                                #differnentiate between bacterial and viral. Bacterial positive, viral negative
        
    else:    
        print('set only bacterial as positive [y/n]?')
        bact_all_diff = bool(strtobool(input()))  #look at all images, only bacterial pneumonia is positive.
            
    pca_inspect = False
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



    if three_classes:
        if bact_viral_diff:
            vkey = 0
        else:
            vkey = 2
    elif bact_all_diff:
        vkey = 0
    else:
        vkey = 1


    print("Loading train images")
    if bact_viral_diff:
        pass 
    else:
        ImgTrain0, XTrain0, yTrain0 = accrueFiles(filenamesTrainNORMAL,
                                                    numFiles=numFiles,
                                                    key=0,
                                                    n_components=n_components,
                                                    imSize=imSize,
                                                    methods=methods,
                                                    bandwidth=bandwidth)

    ImgTrain1, XTrain1, yTrain1 = accrueFiles(filenamesTrainBact,
                                                numFiles=numFiles,
                                                key=1,
                                                n_components=n_components,
                                                imSize=imSize,
                                                methods=methods,
                                                bandwidth=bandwidth)
                                            
    ImgTrain2, XTrain2, yTrain2 = accrueFiles(filenamesTrainVir,
                                                numFiles=numFiles,
                                                key=vkey,
                                                n_components=n_components,
                                                imSize=imSize,
                                                methods=methods,
                                                bandwidth=bandwidth)

    print("Loading validation images")
    if bact_viral_diff:
        pass 
    else:
        ImgVal0, XVal0, yVal0 = accrueFiles(filenamesValNORMAL,
                                                key=0,
                                                n_components=n_components,
                                                imSize=imSize,
                                                methods=methods,
                                                bandwidth=bandwidth)

    ImgVal1, XVal1, yVal1 = accrueFiles(filenamesValBact,
                                            key=1,
                                            n_components=n_components,
                                            imSize=imSize,
                                            methods=methods,
                                            bandwidth=bandwidth)
    ImgVal2, XVal2, yVal2 = accrueFiles(filenamesValVir,
                                            key=vkey,
                                            n_components=n_components,
                                            imSize=imSize,
                                            methods=methods,
                                            bandwidth=bandwidth)

    print("Loading test images")
    if bact_viral_diff:
        pass 
    else:
        ImgTest0, XTest0, yTest0 = accrueFiles(filenamesTestNORMAL,
                                                numFiles=numFiles,
                                                key=0,
                                                n_components=n_components,
                                                imSize=imSize,
                                                methods=methods,
                                                bandwidth=bandwidth)
    
    ImgTest1, XTest1, yTest1 = accrueFiles(filenamesTestBact,
                                            numFiles=numFiles,
                                            key=1,
                                            n_components=n_components,
                                            imSize=imSize,
                                            methods=methods,
                                            bandwidth=bandwidth)
    ImgTest2, XTest2, yTest2 = accrueFiles(filenamesTestVir,
                                            numFiles=numFiles,
                                            key=vkey,
                                            n_components=n_components,
                                            imSize=imSize,
                                            methods=methods,
                                            bandwidth=bandwidth)


    if bact_viral_diff:
        X = np.array(XTrain1 + XTrain2 + XVal1 + XVal2 +  XTest1 + XTest2)
        y = np.array(yTrain1 + yTrain2 + yVal1 + yVal2  + yTest1 + yTest2)
        Img = np.array(ImgTrain1 + ImgTrain2 + ImgVal1 + ImgVal2 + ImgTest1 + ImgTest2)
        
    else:
        Img = np.array(ImgTrain0 + ImgTrain1 + ImgTrain2 + ImgVal0 + ImgVal1 + ImgVal2 + ImgTest0 + ImgTest1 + ImgTest2)
        X = np.array(XTrain0 + XTrain1 + XTrain2 + XVal0 + XVal1 + XVal2 + XTest0 + XTest1 + XTest2)
        y = np.array(yTrain0 + yTrain1 + yTrain2 + yVal0 + yVal1 + yVal2 + yTest0 + yTest1 + yTest2)
    
    #preprocess images
    Img = image_preprocess(Img)
    path = os.getcwd() + "/results/features/"

    ############################
    ## Quick look at PCA      ##
    ############################
    if pca_inspect == True:
        pca = PCA(2)
        plt.scatter(projected[:,0], projected[:,1],
                    c=y,edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('plasma', 2))
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.title("PCA analysis on image")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path + "PCAplot.png")
        plt.show()

        print("Before scaling:")
        print(np.max(Img), np.min(Img))
        ss = StandardScaler()
        Img = ss.fit_transform(Img)
        print("After scaling:")
        print(np.max(Img), np.min(Img))


        pca = PCA(2)
        projected = pca.fit_transform(Img)
        plt.scatter(projected[:,0], projected[:,1],
                    c=y,edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('plasma', 2))
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.title("PCA analysis on image with scaling")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path + "PCAplotScale.png")
        plt.show()



        projected2 = pca.fit_transform(X)
        plt.scatter(projected2[:,0], projected2[:,1],
                    c=y,edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('plasma', 2))
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.title("PCA analysis on extracted features")
        plt.colorbar()
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
                    cmap=plt.cm.get_cmap('plasma', 2))
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.title("PCA analysis on extracted features after scaling")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path + "PCAplotOnFeaturesScaled.png")
        plt.show()
                                
    print("X and y shape: ", X.shape, y.shape)

    pdX = pd.DataFrame(data=X)
    pdY = pd.DataFrame(data=y)
    
    if write_to_disk:
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
                os.makedirs(path)
                print("Successfully created the directory " + path)
            except:
                print("Creation of the path " + path + " failed.")
            else:
                pass
            try:
                pdX.to_pickle(path + "Features.pkl")
            except:
                print("Couldn't write " + path + "Features.pkl to drive")
                exit()
            try:
                pdY.to_pickle(path + "FeatureLabels.pkl")
            except:
                print("Couldn't write " + path + "FeatureLabels.pkl to drive")
                exit()
    if methods==['image']:
        Img = np.expand_dims(Img, axis =3)
        if write_to_disk:
            if os.path.exists(path):
                if os.path.exists(path + "Images.pkl"):
                    os.remove(path + "Images.pkl")
                try:
                    pk.dump(Img, open(path + "Images.pkl", 'wb'))
                except:
                    print("Couldn't write " + path + "Images.pkl to drive")
                    exit()
            else:
                try:
                    os.makedirs(path)
                except:
                    print("Creation of the path " + path + " failed.")
                else:
                    print("Successfully created the directory " + path)
                try:
                    pk.dump(Img, open(path + "Images.pkl", 'wb'))
                except:
                    print("Couldn't write " + path + "Images.pkl to drive")
                    exit()
        return Img, pdY

    return pdX, pdY
def image_preprocess(Img):
    #subtracts image mean from images and divides by standard deviation of each image
    processed_Img = (Img - np.mean(Img, axis=(1,2), keepdims=True))/np.std(Img, axis=(1,2), keepdims=True)
    return processed_Img

def extractFeatures(img, methods=["pca", "kde", "haralick", "image"], n_components=20, bandwidth = 60):
    featureList = []
    for method in methods:
        
        if method == "pca":
            pca = PCA(n_components=n_components, whiten = False)
            pca.fit(img)
            pcaComponents = pca.components_.ravel()
            featureList.append(pcaComponents)
            """
            plt.plot(pca.explained_variance_ratio_, label="Explained Variance")
            #plt.plot(pca.singular_values_, label="Singular Values")
            plt.title("PCA contributions in sample image")
            plt.xlabel("Features")
            plt.ylabel("Explained Variance")
            plt.legend()
            plt.savefig("Singular_value_explained_variance.png")
            plt.show()
            exit()
            """
        if method == "kde":
            kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(img)
            kdeComponents = kde.score_samples(img).ravel()
            featureList.append(kdeComponents)

        if method == "haralick":
            haralickComponents = np.mean(mt.features.haralick(img), axis=0).ravel()
            featureList.append(haralickComponents)

        if method == "image":
            featureList.append(img.ravel())
            
    features = featureList[0]
    for i in range(len(featureList)-1):
        features = np.concatenate((features, featureList[i+1]))
            
    return features    

# Help function to collect files and simultaniously perform
# preprocessing. Returns selection of features (X) and labels (y)
def accrueFiles(filenames,
                numFiles = "all",
                key=0,
                n_components=10,
                imSize="noResize",
                methods=["pca", "kde", "haralick", "image"],
                bandwidth=61):
    #initiate lists
    Img = []
    X = []
    y = []
    
    # selecting the number of files wanted
    if numFiles != "all":
        filenames = filenames[:numFiles]

    # Loop of over all files
    i = 0
    numFiles = len(filenames)
    for filename in filenames:
        i += 1
        print("File", str(i), "of", str(numFiles), "files", end="\r")
        # Save all features to features
        
        img = Image.open(filename).convert("L") # loads image and converts to grayscale
        if imSize != "noResize":
            img = img.resize(imSize) # if resize is specified resizes image
    
        img_arr = np.array(img.copy()) # copies image to a numpy array
        img.close()

        # Extract features, can choose from kde, pca and haralick
        features = extractFeatures(img = img_arr,
                                   n_components=n_components,
                                   methods=["pca", "kde", "haralick"],
                                   bandwidth = 61) # Check LDA

        # Finish by saving the images and keys
        Img.append(img_arr)
        X.append(features)
        y.append(key)
        
    print("Completed loading images")
    return Img, X, y

def cumulative_plot(y_test,y_test_pred, output, title = 'cumulative curve'):
    sort = np.argsort(-y_test_pred,axis = 0)
    y_test_pred = y_test_pred[sort]
    curve_model_1 = np.cumsum(y_test[sort])
    curve_perfect_model = np.cumsum(-np.sort(-y_test, axis = 0))
    curve_no_model = np.linspace(curve_perfect_model[-1]/len(y_test),curve_perfect_model[-1],num=len(y_test))
    area_model = auc(np.arange(len(y_test)), curve_model_1)
    area_perfect_model = auc(np.arange(len(y_test)), curve_perfect_model)
    area_no_model = auc(np.arange(len(y_test)), curve_no_model)
    cumulative_area_ratio = (area_model-area_no_model)/(area_perfect_model-area_no_model)
    plt.plot(np.arange(y_test.shape[0]), curve_perfect_model)
    plt.plot(np.arange(y_test.shape[0]), curve_model_1)
    plt.plot(np.arange(y_test.shape[0]), curve_no_model)
    plt.legend(['Perfect model','Model', 'Baseline'])
    plt.xlabel('Number of predictions')
    plt.ylabel('Cumulative number of positive outcomes')
    plt.title('Cumulative curve, area ratio: {0:.2f}'.format(cumulative_area_ratio))
    plt.savefig(output)
    plt.tight_layout()
    plt.show()
    print("Cumulative area ratio: ", cumulative_area_ratio)
    return cumulative_area_ratio


def bootstrap_mean_std(score_metric, model, yTest, XTest, n_bootstraps=100, bootstrap_size=0.5):
    '''
    uses bootstrapping to caculate the mean value and 
    standard deviation of the score metric
    score_metric(true, prediction)
    '''
    n = yTest.shape[0]
    n_train = int(n*bootstrap_size)
    score = np.zeros(n_bootstraps)
    for i in range(n_bootstraps):
        true, x_test = resample(yTest, XTest, n_samples=n_train)
        pred = model.predict(x_test)
        score[i] = score_metric(true, pred)
    return np.mean(score), np.std(score)

