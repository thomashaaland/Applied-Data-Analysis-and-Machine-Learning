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
import sklearn.linear_model as skl
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from matplotlib.collections import PolyCollection

np.random.seed(2019) # Random seed to guarantee reproducibility

# Load the terrain
terrain1 = imread('data/SRTM_data_Norway_1.tif')
terrain1 = (terrain1 - np.mean(terrain1))
terrain1 = terrain1/np.std(terrain1)

plt.figure()
plt.imshow(terrain1, cmap='gray')
plt.title('Terrain')
plt.xlabel('X')
plt.ylabel('Y')

y_start = 1500
x_start = 900
y_start = 800
x_start = 800
n_n = 200
nth = 10
terrain1 = terrain1[y_start:y_start + n_n,x_start:x_start + n_n]
terrain1 = terrain1[::nth, ::nth]



plt.figure()
plt.imshow(terrain1, cmap='gray')
plt.title('Terrain choice')
plt.xlabel('X')
plt.ylabel('Y')

x_splits = 1
y_splits = 1
ny, nx = terrain1.shape
mx = int(nx/x_splits)
my = int(ny/y_splits)
terrain1 = terrain1[:my*y_splits,:mx*x_splits]

degree = 6

k = 5 # k when running through lambda values
poly_deg_k = 10 #k when running over different poly degrees
minDegree = 0
maxDegree = 25
maxDegree += 1
minLambda = -7 #-7
maxLambda = 1 #-4

numLambdas = 50 # 50 was good number
degrees = np.arange(minDegree,maxDegree)
lambdas = np.logspace(maxLambda,minLambda,numLambdas)

for i_x in range(x_splits):
    for j_y in range(y_splits):
        area_n = j_y + (i_x)*y_splits
        #print( 'areas left=', x_splits*y_splits - area_n)
        terrain = terrain1[my*j_y:my*(j_y+1), mx*i_x:mx*(i_x+1)]
        Nx = terrain.shape[0]
        Ny = terrain.shape[1]
        x_m = np.zeros((mx, my))
        y_m = np.zeros((mx, my))
        x_line = np.linspace(0, 1, mx)
        y_line = np.linspace(0, 1, my)
        for i in range(mx):
            for j in range(my):
                x_m[i,j] = x_line[i]
                y_m[i,j] = y_line[j]

        x = x_m.flatten()
        y = y_m.flatten()
        z = terrain.flatten()
        
        xy = np.c_[x, y]
        X_plot = create_X(x, y, degree)
        X = create_X(x, y, degree)
        #X, X_trash, x ,x_trash, y, t_trash, z, z_trash = train_test_split(X, x, y, z, test_size=0.8, shuffle=True)
        #splitting the data in train and test data
        X_rest, X_test, x_rest, x_test, y_rest, y_test, z_rest, z_test= train_test_split(X, x, y, z, test_size = 3/5 , shuffle=True)
        f_rest = z_rest
        number_of_train = len(z_rest)
        print('number of z_values in train', number_of_train)

        lamErrOLS = np.zeros(lambdas.shape[0])
        lamErrRidge = np.zeros(lambdas.shape[0])
        lamErrLasso = np.zeros(lambdas.shape[0])

        numBetas = X_rest.shape[1]

        betasOLS = np.empty((numLambdas, numBetas))
        betasRidge = np.empty((numLambdas, numBetas))
        betasLasso = np.empty((numLambdas, numBetas))

        betasSigmaOLS = np.empty((numLambdas, numBetas))
        betasSigmaRidge = np.empty((numLambdas, numBetas))
        betasSigmaLasso = np.empty((numLambdas, numBetas))


        kf = oh.k_fold(n_splits=k, shuffle=True)
        kf.get_n_splits(X_rest)


        
        for nlam, _lambda in enumerate(lambdas):
            print(_lambda)
            ######### KFold! #############

            errorsOLS = np.empty(k)
            zPredictsOLS = np.empty((int(z.shape[0]/k)))
            betasOLSTemp = np.empty((k, numBetas))
            betasSigmaOLSTemp = np.empty((k, numBetas))
            
            errorsRidge = np.empty(k)
            zPredictsRidge = np.empty((int(z.shape[0]/k)))
            betasRidgeTemp = np.empty((k, numBetas))
            betasSigmaRidgeTemp = np.empty((k, numBetas))
            
            errorsLasso = np.empty(k)
            zPredictsLasso = np.empty((int(z.shape[0]/k)))
            betasLassoTemp = np.empty((k, numBetas))
            betasSigmaLassoTemp = np.empty((k, numBetas))
            
            zTests = np.empty((int(z.shape[0]/k)))
            i = 0

            for train_index, test_index in kf.split():
                X_train, X_validation = X_rest[train_index], X_rest[test_index]
                x_train, x_validation = x_rest[train_index], x_rest[test_index]
                y_train, y_validation = y_rest[train_index], y_rest[test_index]
                z_train, z_validation = z_rest[train_index], z_rest[test_index]
                f_train, f_validation = f_rest[train_index], f_rest[test_index]

                # OLS, Finding the best lambda
                betaOLS = oh.linFit(X_train, z_train, model='OLS', _lambda = _lambda)
                betasOLSTemp[i] = betaOLS.reshape(-1)
                zPredictsOLS = (X_validation @ betaOLS)
                errorsOLS[i] = np.mean((f_validation - zPredictsOLS)**2)
                sigmaOLSSq = 1/(X_validation.shape[0] - 0*X_validation.shape[1]) * np.sum((z_validation - zPredictsOLS)**2)
                sigmaBetaOLSSq = sigmaOLSSq * np.diag(np.linalg.pinv(X_validation.T @ X_validation))
                betasSigmaOLSTemp[i] = np.sqrt(sigmaBetaOLSSq)
                
                
                # Ridge, Finding the best lambda
                betaRidge = oh.linFit(X_train, z_train, model='Ridge', _lambda = _lambda)
                betasRidgeTemp[i] = betaRidge.reshape(-1)
                zPredictsRidge = (X_validation @ betaRidge)
                errorsRidge[i] = np.mean((f_validation - zPredictsRidge)**2)
                sigmaRidgeSq = 1/(X_validation.shape[0] - 0*X_validation.shape[1]) * np.sum((z_validation - zPredictsRidge)**2)
                XInvRidge = np.linalg.pinv(X_validation.T @ X_validation + _lambda * np.eye(len(betaRidge)))
                sigmaBetaRidgeSq = sigmaRidgeSq * np.diag(XInvRidge @ X_validation.T @ X_validation @ XInvRidge.T)
                betasSigmaRidgeTemp[i] = np.sqrt(sigmaBetaRidgeSq)
                
                # Lasso, Finding the best lambda

                lasso = skl.Lasso(alpha = _lambda, fit_intercept=False, max_iter=10**2, precompute=True).fit(X_train, z_train)
                betaLasso = lasso.coef_
                betasLassoTemp[i] = betaLasso.reshape(-1)
                zPredictsLasso = lasso.predict(X_validation)
                errorsLasso[i] = np.mean((f_validation - zPredictsLasso)**2)
                i += 1
            #ax.scatter3D(x_validation, y_validation, zPredictsLasso, '.')
            #plt.show()
            betasOLS[nlam] = np.mean(betasOLSTemp,axis=0)
            betasRidge[nlam] = np.mean(betasRidgeTemp,axis=0)
            betasLasso[nlam] = np.mean(betasLassoTemp,axis=0)
            betasSigmaOLS[nlam] = np.mean(betasSigmaOLSTemp, axis=0)
            betasSigmaRidge[nlam] = np.mean(betasSigmaRidgeTemp, axis = 0)
            lamErrOLS[nlam] = min(errorsOLS)
            lamErrRidge[nlam] = min(errorsRidge)
            lamErrLasso[nlam] = min(errorsLasso)
            #lamErrOLS[nlam] = np.mean(errorsOLS)
            #lamErrRidge[nlam] = np.mean(errorsRidge)
            lamErrLasso[nlam] = np.mean(errorsLasso)

        
        
        def plot_from_beta(beta):
            plt.figure()
            xplot = np.linspace(0,1,n_n)
            yplot = np.linspace(0,1,n_n)
            xplot, yplot = np.meshgrid(xplot,yplot)

            zplot = np.zeros((n_n, n_n))
            for i in range(n_n):
                for j in range(n_n):
                    Xplot = create_X(xplot[i,j], yplot[i,j],degree)
                    zplot[i,j] = Xplot.dot(beta)
            fig = plt.imshow(zplot.T, cmap='gray')
            plt.xlabel('X')
            plt.ylabel('Y')
            return fig
        fig = plot_from_beta(betaOLS)
        plt.title('Prediction of the terrain data from OLS regression. polynomial degree {}'.format(degree))
        plt.show()
        '''
        ############################
        # Write to file the errors #
        ############################
        write2OLS = open("betasOLS.txt", "w+")
        # FORMAT: lambda -> beta0 -> beta1 -> beta...
        write2OLS.write("lambdas\tbetas\t\t\tsigmaBeta\t\tconfidence interval\n")
        for i in range(betasOLS.shape[0]):
            lineOLS = str(lambdas[i])+"\t"
            for j in range(betasOLS.shape[1]):
                lineOLS += str(betasOLS[i][j]) + "\t" + str(betasSigmaOLS[i][j]) + "\t" + str(betasOLS[i][j] - 2*betasSigmaOLS[i][j]) + "\t" + str(betasOLS[i][j] + 2*betasSigmaOLS[i][j]) + "\n\t"
            lineOLS += "\n"

        write2OLS.write(lineOLS)
        write2OLS.close()

        write2Ridge = open("betasRidge.txt", "w+")
        # FORMAT: lambda -> beta0 -> beta1 -> beta...
        write2Ridge.write("lambdas\tbetas\tsigmaBeta\tconfidence interval")
        for i in range(betasRidge.shape[0]):
            lineRidge = str(lambdas[i])+"\t"
            for j in range(betasRidge.shape[1]):
                lineRidge += str(betasRidge[i][j]) + "\t" + str(betasSigmaRidge[i][j]) + "\t" + str(betasRidge[i][j] - 2*betasSigmaRidge[i][j]) + "\t" + str(betasRidge[i][j] + 2*betasSigmaRidge[i][j]) + "\n\t"
            lineRidge += "\n"

        write2Ridge.write(lineRidge)
        write2Ridge.close()

        write2Lasso = open("betasLasso.txt", "w+")
        # FORMAT: lambda -> beta0 -> beta1 -> beta...
        write2Lasso.write("lambdas\tbetas")
        for i in range(betasLasso.shape[0]):
            lineLasso = str(lambdas[i])+"\t"
            for j in range(betasLasso.shape[1]):
                lineLasso += str(betasLasso[i][j]) + "\n\t"
            lineLasso += "\n"

        write2Lasso.write(lineRidge)
        write2Lasso.close()
        
        
        '''
        ##########################################
        ## Plot beta with errorbar              ##
        ##########################################



        """
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        error = []
        verts = []
        print(betasSigmaOLS.T[0])
        for i in range(betasOLS.shape[1]):
            verts.append([list(zip(lambdas, betasOLS.T[i]))])
            error.append(betasSigmaOLS.T[i])
        poly = PolyCollection(verts)
        poly.set_alpha(0.8)

        print(error)
        ax.add_collection3d(poly, zs=error, zdir='Z')
        ax.set_xlabel("lambdas")
        ax.set_xlim3d(min.lambdas, max.lambdas)
        ax.set_ylim3d(min.betasOLS, max.betasOLS)
        ax.set_zlabel("Z")
        ax.set_zlim3d(min.betasSigmaOLS, max.betasSigmaOLS)
        plt.show()

            #plt.errorbar(lambdas, betasOLS.T[i], betasSigmaOLS.T[i], errorevery=9)
        plt.title("OLS varying with lambda")
        plt.xscale("log")
        plt.xlabel("Lambdas")
        plt.ylabel("Beta error")
        plt.savefig("betasOLS.png")
        plt.show()
        """
        """
        for i in range(betasOLS.shape[1]):
            plt.errorbar(lambdas, betasOLS.T[i], betasSigmaOLS.T[i], errorevery=10)
        plt.title("OLS varying with lambda")
        plt.xlabel("Lambdas")
        plt.ylabel("Beta error")
        plt.savefig("betasOLS.png")
        plt.xscale("log")
        plt.show()

        for i in range(betasRidge.shape[1]):
            plt.errorbar(lambdas, betasRidge.T[i], betasSigmaRidge.T[i], errorevery=10)
        plt.title("Ridge varying with lambda")
        plt.xlabel("Lambdas")
        plt.ylabel("Beta error")
        plt.savefig("betasRidge.png")
        plt.xscale("log")
        plt.show()

        plt.plot(lambdas, betasLasso)
        plt.title("Lasso varying with lambda")
        plt.xlabel("Lambdas")
        plt.ylabel("Beta error")
        plt.savefig("betasLasso.png")
        plt.xscale("log")
        plt.show()

            
        plt.loglog(lambdas, lamErrOLS,label="OLS")
        plt.loglog(lambdas, lamErrRidge,label="Ridge")
        plt.loglog(lambdas, lamErrLasso,label="Lasso")
        plt.xlabel("Lambda")
        plt.ylabel("Error")
        plt.legend()
        plt.savefig("bestLambdaCompare.png")
        plt.show()
        """

        for i in range(betasOLS.shape[1]):
            error = 2*betasSigmaOLS.T[i]
            plt.plot(lambdas, betasOLS.T[i])
            plt.fill_between(lambdas, betasOLS.T[i]-error, betasOLS.T[i]+error, alpha = 0.2)
        plt.title("OLS varying with lambda")
        plt.xlabel("Lambdas")
        plt.ylabel("Beta error")
        plt.savefig("betasOLS.png")
        plt.xscale("log")
        plt.show()

        for i in range(betasRidge.shape[1]):
            error = 2*betasSigmaRidge.T[i]
            plt.plot(lambdas, betasRidge.T[i])
            plt.fill_between(lambdas, betasRidge.T[i]-error, betasRidge.T[i]+error, alpha = 0.2)
        plt.title("Ridge varying with lambda")
        plt.xlabel("Lambdas")
        plt.ylabel("Beta error")
        plt.savefig("betasRidge.png")
        plt.xscale("log")
        plt.show()

        plt.plot(lambdas, betasLasso)
        plt.title("Lasso varying with lambda")
        plt.xlabel("Lambdas")
        plt.ylabel("Beta error")
        plt.savefig("betasLasso.png")
        plt.xscale("log")
        plt.show()

            
        plt.plot(lambdas, lamErrOLS,label="OLS")
        plt.plot(lambdas, lamErrRidge,label="Ridge")
        plt.plot(lambdas, lamErrLasso,label="Lasso")
        plt.xlabel("Lambda")
        plt.ylabel("Error")
        plt.xscale('log')
        plt.title('MSE vs lambda for polynomial degree={}, number of train values={}'.format(degree, number_of_train))
        plt.legend()
        plt.tight_layout()
        plt.savefig("bestLambdaCompare.png")
        plt.show()

        lambdaRidge = np.array([lambdas[lamErrRidge == min(lamErrRidge)]])
        lambdaLasso = np.array([lambdas[lamErrLasso == min(lamErrLasso)]])
        lambdas = np.array([0])
        print('Lasso lambda =', lambdaLasso)
                                
        ##### Start the loop over many degrees of polynomial ########
        #x_train, x_test, y_train, y_test, z_train, z_test, f_train, f_test = train_test_split(x, y, z, f, test_size=0.2)
        variancesOLS = np.zeros((lambdas.shape[0], degrees.shape[0]))
        biasesOLS = np.zeros((lambdas.shape[0], degrees.shape[0]))
        biasefsOLS = np.zeros((lambdas.shape[0], degrees.shape[0]))
        errorsOLS = np.zeros((lambdas.shape[0], degrees.shape[0]))

        variancesRidge = np.zeros((lambdas.shape[0], degrees.shape[0]))
        biasesRidge = np.zeros((lambdas.shape[0], degrees.shape[0]))
        biasefsRidge = np.zeros((lambdas.shape[0], degrees.shape[0]))
        errorsRidge = np.zeros((lambdas.shape[0], degrees.shape[0]))

        variancesLasso = np.zeros((lambdas.shape[0], degrees.shape[0]))
        biasesLasso = np.zeros((lambdas.shape[0], degrees.shape[0]))
        biasefsLasso = np.zeros((lambdas.shape[0], degrees.shape[0]))
        errorsLasso = np.zeros((lambdas.shape[0], degrees.shape[0]))

        k = poly_deg_k
        x_rest, x_test, y_rest, y_test, z_rest, z_test, f_rest, f_test = train_test_split(x, y, z, z, test_size = int(z.shape[0]/k), shuffle=True)        
        z_test = np.expand_dims(z_test, axis=1)
        kf = oh.k_fold(n_splits=k-1, shuffle=True)
        kf.get_n_splits(x_rest)
        for nlam, _lambda in enumerate(lambdas):
            for degree in degrees:
                ######### KFold! #############
                print('degree=', degree)               
                X_rest = oh.create_X(x_rest.ravel(), y_rest.ravel(), degree, intercept=False)
                X_test = oh.create_X(x_test.ravel(), y_test.ravel(), degree, intercept=False)
                
                
                #kf = KFold(n_splits=k-1, random_state=None, shuffle=True)
                zPredictsOLS = np.empty((int(z.shape[0]/k), k))
                zPredictsRidge = np.empty((int(z.shape[0]/k), k))
                zPredictsLasso = np.empty((int(z.shape[0]/k), k))
                i = 0
                #for train_index, test_index in kf.split(X_rest):
                for train_index, test_index in kf.split():
                    #print("TRAIN:", train_index, "TEST:", test_index)
                    X_train, X_validation = X_rest[train_index], X_rest[test_index]
                    x_train, x_validation = x_rest[train_index], x_rest[test_index]
                    y_train, y_validation = y_rest[train_index], y_rest[test_index]
                    z_train, z_validation = z_rest[train_index], z_rest[test_index]
                    f_train, f_validation = f_rest[train_index], f_rest[test_index]

                    # OLS, Finding the best lambda
                    betaOLS = oh.linFit(X_train, z_train, model='OLS', _lambda = _lambda)
                    zPredictsOLS[:,i] = (X_test @ betaOLS).reshape(-1) # Used validation to get good results
                
                    # Ridge, Finding the best lambda
                    betaRidge = oh.linFit(X_train, z_train, model='Ridge', _lambda = lambdaRidge)
                    zPredictsRidge[:,i] = (X_test @ betaRidge).reshape(-1) # Used validation to get good results

                    # Lasso, Finding the best lambda

                    lasso = skl.Lasso(alpha = lambdaLasso, fit_intercept=False, max_iter=10**2, precompute=True)
                    if X_train.shape[1] == 0:
                        zPredictsLasso[:,i] = np.zeros(len(X_test)) # Used validation to get good results
                    else:
                        lasso.fit(X_train, z_train)
                        zPredictsLasso[:,i] = lasso.predict(X_test).reshape(-1) # Used validation to get good results

                            
                    i += 1
                    #ax.scatter3D(x_test, y_test, zPredicts[:,i-1], '.')
                # To do: Need to perform an analysis of the betas from OLS, Ridge and Lasso. Plot the values and compare for some degree
                # Also need to do a comparison like above, but for different lambdas
                
                varianceOLS = np.mean( np.var(zPredictsOLS, axis=1, keepdims=True) )
                biasOLS = np.mean( ( z_test - np.mean( zPredictsOLS, axis=1, keepdims=True ) )**2 )
                biasfOLS = np.mean( ( f_test - np.mean( zPredictsOLS, axis=1, keepdims=True ) )**2 )
                errorOLS = np.mean( np.mean( ( z_test - zPredictsOLS)**2, axis=1, keepdims=True ) )

                variancesOLS[nlam][degree] = varianceOLS
                biasesOLS[nlam][degree] = biasOLS
                biasefsOLS[nlam][degree] = biasfOLS
                errorsOLS[nlam][degree] = errorOLS

                varianceRidge = np.mean( np.var(zPredictsRidge, axis=1, keepdims=True) )
                biasRidge = np.mean( ( z_test - np.mean( zPredictsRidge, axis=1, keepdims=True ) )**2 )
                biasfRidge = np.mean( ( f_test - np.mean( zPredictsRidge, axis=1, keepdims=True ) )**2 )
                errorRidge = np.mean( np.mean( ( z_test - zPredictsRidge)**2, axis=1, keepdims=True ) )
                            
                variancesRidge[nlam][degree] = varianceRidge
                biasesRidge[nlam][degree] = biasRidge
                biasefsRidge[nlam][degree] = biasfRidge
                errorsRidge[nlam][degree] = errorRidge
                            
                varianceLasso = np.mean( np.var(zPredictsLasso, axis=1, keepdims=True) )
                biasLasso = np.mean( ( z_test - np.mean( zPredictsLasso, axis=1, keepdims=True ) )**2 )
                biasfLasso = np.mean( ( f_test - np.mean( zPredictsLasso, axis=1, keepdims=True ) )**2 )
                errorLasso = np.mean( np.mean( ( z_test - zPredictsLasso)**2, axis=1, keepdims=True ) )
                
                variancesLasso[nlam][degree] = varianceLasso
                biasesLasso[nlam][degree] = biasLasso
                biasefsLasso[nlam][degree] = biasfLasso
                errorsLasso[nlam][degree] = errorLasso


        fig = plt.figure()
        plt.plot(np.mean(errorsOLS, axis=0), label="MSE")
        plt.plot(np.mean(variancesOLS, axis=0), label="Variance")
        plt.plot(np.mean(biasesOLS, axis=0), label="Bias")
        plt.ylim([0,2])
        plt.title("Bias Variance tradeoff for OLS")
        plt.xlabel("Degree")
        plt.ylabel("Error")
        plt.legend()
        plt.savefig("biasVarianceOLS.png")
        plt.show()

        fig = plt.figure()
        plt.plot(np.mean(errorsRidge, axis=0), label="MSE")
        plt.plot(np.mean(variancesRidge, axis=0), label="Variance")
        plt.plot(np.mean(biasesRidge, axis=0), label="Bias")
        plt.ylim([0,2])
        plt.title("Bias Variance tradeoff for Ridge, lambda={:.7f}".format(lambdaRidge[0][0]))
        plt.xlabel("Degree")
        plt.ylabel("Error")
        plt.legend()
        plt.savefig("biasVarianceRidge.png")
        plt.show()

        fig = plt.figure()
        plt.plot(np.mean(errorsLasso, axis=0), label="MSE")
        plt.plot(np.mean(variancesLasso, axis=0), label="Variance")
        plt.plot(np.mean(biasesLasso, axis=0), label="Bias")
        plt.ylim([0,2])
        plt.title("Bias Variance tradeoff for Lasso, lambda={:.7f}".format(lambdaLasso[0][0]))
        plt.xlabel("Degree")
        plt.ylabel("Error")
        plt.legend()
        plt.savefig("biasVarianceLasso.png")
        plt.show()
