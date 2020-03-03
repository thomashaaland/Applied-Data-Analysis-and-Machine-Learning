from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import oblig1_header as oh
from oblig1_header import create_X
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import sklearn as skl
# Load the terrain
terrain1 = imread('data/SRTM_data_Norway_1.tif')
terrain1 = (terrain1 - np.mean(terrain1))/np.std(terrain1)

def CV_fit(X, z, k, f=None, alpha=0, method='OLS'):
    #f is the exact function
    OLS = method == 'OLS'
    Ridge = method == 'Ridge'
    Lasso = method == 'Lasso'
    if f is None:
        f = z
    kf = oh.k_fold(k)
    kf.get_n_splits(X)
    beta = np.zeros((k, X.shape[1]))
    errors = np.zeros(k)
    betasSigma = np.zeros(beta.shape)
    i = 0
    for train_index, test_index in kf.split():
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_validation = X[train_index], X[test_index]
        z_train, z_validation = z[train_index], z[test_index]
        f_train, f_validation = f[train_index], f[test_index]
        
        if OLS:            
            beta[i,:] = oh.linFit(X_train, z_train, model='OLS', _lambda = 0)
            #zPredictsOLS[:,i] = (X_test @ betaOLS).reshape(-1) # Used validation to get good results
        elif Ridge:
            beta[i,:] = oh.linFit(X_train, z_train, model='Ridge', _lambda = alpha)
            #zPredictsRidge[:,i] = (X_test @ betaRidge).reshape(-1) # Used validation to get good results
        elif Lasso:
            clf = skl.Lasso(alpha = alpha, fit_intercept=False, max_iter=10**8, precompute=True).fit(X_train, z_train)
            beta[i,:] = clf.coef_
        else:
            raise Exception('method has to be Lasso, OLS or Ridge, not {}'.format(method))
       
        zPredicts = (X_validation @ beta[i,:])
        errors[i] = np.mean((f_validation - zPredicts)**2)
    
        if OLS:
            sigmaOLSSq = 1/(X_validation.shape[0] - 0*X_validation.shape[1]) * np.sum((z_validation - zPredicts)**2)
            sigmaBetaOLSSq = sigmaOLSSq * np.diag(np.linalg.pinv(X_validation.T @ X_validation))
            betasSigma[i,:] = np.sqrt(sigmaBetaOLSSq)

        elif Ridge:
            XInvRidge = np.linalg.pinv(X_validation.T @ X_validation + alpha * np.eye(len(beta[i,:])))
            sigmaRidgeSq = 1/(X_validation.shape[0] - 0*X_validation.shape[1]) * np.sum((z_validation - zPredicts)**2)
            sigmaBetaRidgeSq = sigmaRidgeSq * np.diag(XInvRidge @ X_validation.T @ X_validation @ XInvRidge.T)
            betasSigma[i,:] = np.sqrt(sigmaBetaRidgeSq)
        
        elif Lasso:
            pass

        i += 1
    return beta, errors, betasSigma

def fit_terrain_data(terrain1, degree=15, reg_type='OLS' ,k=5 , alpha=0, x_splits=4, y_splits=8, plot_every_area=False):
    OLS = reg_type == 'OLS'
    Ridge = reg_type == 'Ridge'
    Lasso = reg_type == 'Lasso'
    ny, nx = terrain1.shape
    mx = int(nx/x_splits)
    my = int(ny/y_splits)
    terrain1 = terrain1[:my*y_splits,:mx*x_splits]
    z_final_predict = np.zeros(terrain1.shape)
    area_error = np.zeros(y_splits*x_splits)
    for i_x in range(x_splits):
        for j_y in range(y_splits):
            #print( 'areas left=', x_splits*y_splits - (j_y + (i_x)*y_splits))
            terrain = terrain1[my*j_y:my*(j_y+1), mx*i_x:mx*(i_x+1)]
            Nx = terrain.shape[0]
            Ny = terrain.shape[1]
            x = np.zeros((mx, my))
            y = np.zeros((mx, my))
            x_line = np.linspace(0, 1, mx)
            y_line = np.linspace(0, 1, my)
            for i in range(mx):
                for j in range(my):
                    x[i,j] = x_line[i]
                    y[i,j] = y_line[j]

            x = x.flatten()
            y = y.flatten()
            z = terrain.flatten()
            
            xy = np.c_[x, y]
            X_plot = create_X(x, y, degree)
            X = create_X(x, y, degree)
            kf = oh.k_fold(k)
            X_rest, X_test, z_rest, z_test= train_test_split(X, z, test_size = int(z.shape[0]/k), shuffle=True)
            betas, errors, betaSigma = CV_fit(X_rest, z_rest, k, alpha=0,method=reg_type )
            best_i = np.argmin(errors)
            beta = betas[best_i,:]
            area_error[j_y + (i_x)*y_splits] = np.mean(errors, axis=0)
            z_plot1 = X_plot.dot(beta).reshape(terrain.shape)
            z_final_predict[my*j_y:my*(j_y+1), mx*i_x:mx*(i_x+1)] = z_plot1[:,:]
            
            if plot_every_area:
                plt.figure()
                plt.title('Terrain over Norway, area{}'.format(j_y + (i_x)*y_splits))
                plt.imshow(terrain, cmap='gray')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.figure()
                plt.title('Terrain over Norway prediction, area{}'.format(j_y + (i_x)*y_splits))
                plt.imshow(z_plot1, cmap='gray')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.show()
    _min = -2
    _max = 2.5
    fig1 =  plt.figure()
    plt.imshow(z_final_predict, cmap='gray', vmin = _min, vmax = _max)
    
    plt.xlabel('X')
    plt.ylabel('Y')

    fig2 = plt.figure()
    plt.imshow(terrain1, cmap='gray', vmin = _min, vmax = _max)
    plt.title('Terrain')
    plt.xlabel('X')
    plt.ylabel('Y')    
    #plt.show()
    mse = np.mean((z_final_predict-terrain1)**2)
    print('mse over all points=', mse)
    print('area error', np.mean(area_error))
    print('R2 score', oh.R2_score(z_final_predict, terrain1))
    return fig1, fig2
terrain = terrain1[::10, ::10]
degree = 8
fit, terrainplot = fit_terrain_data(terrain,degree=degree,x_splits=8, y_splits=16, reg_type='OLS', alpha=0)

fit.suptitle('{} degree polynomial OLS fit, to terrain split into {} smaller areas'.format(degree, 8*16))
plt.show()