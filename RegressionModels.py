#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
import pre_hot as pre
from sklearn.externals import joblib
import os
from sklearn.metrics import r2_score
def fit_model(X,Y,model,filename):
    if (os.path.exists(filename)):
        resmodel=joblib.load(filename)
        #print("ffffffff")
    else:

        resmodel=model.fit(X,Y)


        joblib.dump(resmodel,filename)

    return resmodel
def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)


def plot_regression_line(x, b):
    # plotting the actual points as scatter plot
    #plt.scatter(x, y, color="m",
     #           marker="o", s=30)

    # predicted response vector
    y_pred = b[0] + b[1] * x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels
    #plt.xlabel(featurename)
    #plt.ylabel('average_rate')

    # function to show plot
    plt.show()

def plot(X_test,prediction):

    for i in range(len(X_test.columns)):
        x = X_test[X_test.columns[i]]
        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(prediction, axis=1)
        plt.scatter(x, y)

        plt.xlabel(X_test.columns[i], fontsize=20)
        plt.ylabel('average_rate', fontsize=20)
        '''
        plt.plot(x, prediction, color='red', linewidth=3)
        plt.show()
        '''

        # estimating coefficients
        b = estimate_coef(x, y)

        # plotting regression line
        plot_regression_line(x, b)
        # plotting the regression line
        # plt.plot(x, b, color="g")

        plt.show()

def LinearRegressionMultivariable(X_train, y_train,X_test, y_test,rootFolder=''):

    r = linear_model.LinearRegression()
    #r.fit(X_train, y_train)
    r = fit_model(X_train,y_train,r,rootFolder+'LinearRegression')
    prediction = r.predict(X_test)
    #print(np.array(prediction[0:50]))
    w = r.coef_
    from sklearn.metrics import accuracy_score
    #print('accuracy',accuracy_score(list(y_test),list(prediction)))


    print('___________Linear Regression______________')
    #print('Co-efficient of linear regression', r.coef_)
    print('Intercept of linear regression model', r.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
    print('Score of linear regression',r.score(X_test,y_test))
    print('R2 Score of Linear Regression',r2_score(y_test,prediction))
    #for i in range(0, 10):
        #print('actual:', y_test.iloc[i], ', pred:', prediction[i])
    print('__________________________________________')

    return r,metrics.mean_squared_error(y_test, prediction)


from sklearn.preprocessing import PolynomialFeatures


def PolynomialRegression(X_train, y_train,X_test, y_test):

    poly_features = PolynomialFeatures(degree=2)

    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)

    # fit the transformed features to Linear Regression
    poly_model = linear_model.LinearRegression(normalize=True)
    #poly_model.fit(X_train_poly, y_train)
    poly_model = fit_model(X_train_poly,y_train,poly_model,'PolynomialRegression')
    # predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)
    # predicting on test data-set
    prediction = poly_model.predict(poly_features.fit_transform(X_test))
    print('___________Polynomial Regression______________')

    #print('Co-efficient of Polynomial regression', poly_model.coef_)
    print('Intercept of Polynomial regression model', poly_model.intercept_)
    print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
    print('Score of Polynomial', poly_model.score(poly_features.fit_transform(X_test), y_test))
    print('R2 Score of Polynomial Regression', r2_score(y_test, prediction))
    print('__________________________________________')
    return poly_model, metrics.mean_squared_error(y_test, prediction)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsRegressor
def DecisionTreeRegression(X_train, y_train,X_test, y_test,rootFolder=''):
    tree = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, random_state=100,
                                 min_samples_split=10,
                                 min_samples_leaf=10)
    #tree.fit(X_train, y_train)
    tree = fit_model(X_train, y_train,tree,rootFolder+'TreeRegression')
    predict = tree.predict(X_test)
    score = tree.score(X_test, y_test)
    MSE = mean_squared_error(predict, y_test)
    print('___________Decision Tree Regression______________')

    print('mean squared error (Tree)= ', MSE)
    print('Score = ', score)
    print('R2 Score', r2_score(y_test, predict))
    print('__________________________________________')
    export_graphviz(tree, out_file="Tree.dot")
    return tree,MSE
def KNNRegression(X_train, y_train,X_test, y_test,n_neighbors=None,rootFolder=''):
    K = 1
    Min_K = -1
    min = 100000
    KNN = None
    if n_neighbors is None:
        for i in range(100):
            KNN = KNeighborsRegressor(n_neighbors=K)
            KNN.fit(X_train, y_train)
            predict = KNN.predict(X_test)
            MSE = mean_squared_error(predict, y_test)
            #print("When K = ", K, 'MSE = ', MSE)
            if MSE < min:
                min = MSE
                Min_K = i
            K += 1
    else:
        Min_K = n_neighbors
    KNN = KNeighborsRegressor(n_neighbors=Min_K)
    KNN.fit(X_train, y_train)
    predict = KNN.predict(X_test)
    print('___________KNN Regression______________')

    print("Minimum Error When K = ", Min_K)
    print("Minimun MSE = ", min)
    print("Score",KNN.score(X_test,y_test))
    print('R2 Score', r2_score(y_test, predict))

    print('__________________________________________')
    return KNN,min

def PickBestModel(X_train, y_train,X_test, y_test,rootFolder=''):
    MSE = []
    Models = []
    bestModel = None
    linearRegression , linearRegressionMSE = LinearRegressionMultivariable(X_train, y_train,X_test, y_test,rootFolder)
    MSE.append(linearRegressionMSE)
    Models.append(linearRegression)


    DTRegression , DTMSE = DecisionTreeRegression(X_train, y_train,X_test, y_test,rootFolder)
    MSE.append(DTMSE)
    Models.append(DTRegression)
    '''
    KNNReg,KNNRegMSE = KNNRegression(X_train, y_train,X_test, y_test)
    MSE.append(KNNRegMSE)
    Models.append(KNNReg)
    '''
    bestModelIndex = MSE.index(min(MSE))
    bestModel = Models[bestModelIndex]
    print('Least MSE',min(MSE))

    return bestModel

def predict(model,X_test):
    return model.predict(X_test)


def Run(plotting = False,rootFolder='',train_val_split=True):
    datasetpath = 'regression_train_data.csv'
    regressiondatasetscaler = 'regressiondataset_scaler'
    X_train, y_train = pre.ReadDataset('tmdb_5000_movies_train.csv', 'tmdb_5000_credits_train.csv','vote_average')
    if train_val_split:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20)
    else:
        X_test, y_test = pre.ReadDataset('tmdb_5000_movies_testing_regression.xlsx', 'tmdb_5000_credits_test.csv',
                                         'vote_average')

    if (os.path.exists(rootFolder+datasetpath)):
        X_train = pd.read_csv(rootFolder+datasetpath)
        X_train = X_train.drop(labels=[X_train.columns[0]], axis=1)
        scaler = joblib.load(rootFolder+regressiondatasetscaler)

    else:
        X_train.index = range(len(X_train))
        X_train, scaler = pre.prefull(X_train)
        X_train.to_csv(rootFolder+datasetpath)

        joblib.dump(scaler, rootFolder+regressiondatasetscaler)




    data = pd.concat([X_train, y_train], axis=1)

    X_test.index = range(len(X_test))
    #X_train,scaler = pre.prefull(X_train)
    X_test = pre.pre_test(X_test,X_train,scaler)

    bestModel = PickBestModel(X_train, y_train, X_test, y_test,rootFolder)
    print('The Best Model fitting the data is', bestModel)

    xTest = X_test


    ypred = predict(bestModel, xTest)
    print('MSE', mean_squared_error(y_test, ypred))
    if plotting == True:
        plot(X_test, ypred)
