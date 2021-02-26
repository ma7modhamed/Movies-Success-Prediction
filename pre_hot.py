import pandas as pd
import numpy as np
from numpy import  newaxis
import json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#import matplotlib.pyplot as plt; plt.rcdefaults()
#from ttictoc import TicToc


def ReadDataset(DatasetPathMoviesFile,DatasetPathCreditsFile,targetName=None):
    #targetName=None):
    X = None
    Y = []
    m=DatasetPathMoviesFile.split('.')
    #print(m[1])
    if m[1]=='csv':
        MoviesDataFrame = pd.read_csv(DatasetPathMoviesFile)
    else:
        MoviesDataFrame=pd.read_excel(DatasetPathMoviesFile)
        #print(MoviesDataFrame)

    CreditsDataFrame = pd.read_csv(DatasetPathCreditsFile, usecols=['movie_id', 'title', 'cast', 'crew'])
    print (CreditsDataFrame.columns)

    #df = pd.read_csv("test.csv", usecols=['Wheat', 'Oil'])
    print (CreditsDataFrame.columns)

    data = pd.merge(MoviesDataFrame, CreditsDataFrame, left_on='id', right_on='movie_id', how='inner')
    if targetName is not None:
        colsToDrop = [targetName]
        Y = data [targetName]
        #colsToDrop = ['rate']
        #target = data['rate']


        X = data.drop(labels=colsToDrop, axis=1)

        return X,Y
    else:
        X = data
        return X

def class_label(target):
    #target = data['rate']
    Y=[]
    for val in target:
        if (val == 'High'):
            Y.append(3)
        if (val == 'Intermediate'):
            Y.append(2)
        if (val == 'Low'):
            Y.append(1)
    Y = pd.DataFrame(Y)

    Y = np.ravel(Y)
    return Y


new_feature=[]
'''def CategoriesCount(feature,selected):
    categories =[] #dict()
    for j in range(len(feature)):
        elements = json.loads(feature.iloc[j])
        #print(elements)
        for i in range(len(elements)):
            if elements[i][selected] not in categories:
                categories.append(elements[i][selected] )#= 1
            #categories[elements[i][selected]] += 1
    #print("categories=",categories)
    #print("len catag=",len(categories))
    return categories'''
'''def encod(values):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print("len encode =",len(onehot_encoded))
    return onehot_encoded
'''
'''
def encod (List_c):
    lb_make = LabelEncoder()
    cat_df_flights_sklearn= lb_make.fit_transform(List_c)
    #print(cat_df_flights_sklearn)
    return cat_df_flights_sklearn

def binary(X,feature,selected):
    newFeature = []
    for i in range(len(feature)):
        rowinfeature = []
        elements = json.loads(feature.iloc[i])
        L=list(X.columns.values)
        for j in range(len(elements)):
            if elements[j][selected] in L:
                X.insert(elements[j][selected])
                print("create")
        X[elements[j][selected]] = np.where(feature.str.contains(elements[j][selected]), 1, other=0)
    return X
def replacewithlable (feature,selected):
    category=CategoriesCount(feature,selected)
    lable=encod(category)
    newFeature = []
    for i in range(len(feature)):
        rowinfeature = []
        elements = json.loads(feature.iloc[i])
        for j in range(len(elements)):
            if elements[j][selected] in category:
                ind=category.index(elements[j][selected])
                rowinfeature.append(lable[ind])
        newFeature.append(np.sum(rowinfeature))#/len(elements))
    print ("newFeature=",newFeature)
    print("len",len(newFeature))
    return newFeature
def Replace(featureName,selected,X):
    feature = X[featureName]
    #categories = CategoriesCount(feature,selected)
    newFeature = replacewithlable(feature,selected)
    loc = list(X.columns.values).index(featureName)
    X = X.drop(labels=[featureName], axis=1)
    X.insert(loc, featureName, newFeature, True)
    NX = X
   # print ("Replac=",NX)
    return NX
'''

def CategoriesCount(feature,selected):
    categories = dict()
    for j in range(len(feature)):
        try:
            elements = json.loads(feature.iloc[j])
        except:
            print('cannot load json')
            continue
        for i in range(len(elements)):
            if elements[i][selected] not in categories.keys():
                categories[elements[i][selected]] = 1
            categories[elements[i][selected]] += 1
    #print("count_cat")
    return categories

def topfeat(list_catagories):
    top=[]
    v = list(list_catagories.values())
    key = list(list_catagories.keys())
    if(len(key)>100):
        for i in range(100):
            ind_max=v.index(max(v))
            top.append(key[ind_max])
            new_feature.append(key[ind_max])
            #print(key[ind_max],v[ind_max])
            v[ind_max]=0
        #print("top")
        return top
    else:
        for i in range(len(key)):
            new_feature.append(key[i])
        #print("top_else")
        return key

def fill_new_feature(featureName,selected,key,X):
    #row=dict()
    feature = X[featureName]
    for i in range(len(feature)):
        #rowinfeature = dict#[]
        try:
            elements = json.loads(feature.iloc[i])
        except:
            print('cannot load json')
            continue
        for j in range(len(elements)):
            M=elements[j][selected]
            if M in key:
                X.loc[i,M]=1
    X = X.drop(labels=[featureName], axis=1)
    #print("fill_new_feature")
    return X

def newcolum(X,key):
    for i in range(len(key)):
        X[key[i]]=0
    #print("newcol=")
    return X

def prefull(X):
    #scaler = StandardScaler()
    scaler = MinMaxScaler()
    colsToDrop = ['homepage', 'id', 'original_language', 'title_x', 'title_y', 'overview', 'original_title',
                  'status', 'tagline', 'movie_id']
    X = X.drop(labels=colsToDrop, axis=1)
    colsToReplace = {'genres': 'name',
                     'keywords': 'name',
                     'production_companies': 'name',
                     'production_countries': 'name',
                     'spoken_languages': 'name',
                     'cast': 'name',
                     'crew': 'name'}
    X['release_date'] = (pd.DatetimeIndex(X['release_date']).year)

    XtoNormalize = X.drop(labels=list(colsToReplace.keys()), axis=1)

    #new_feature=[]
    for key, value in colsToReplace.items():

        #print(key)
        feature=X[key]
        categories = CategoriesCount(feature, value)
        N = topfeat(categories)
        #new_feature.append(N)
        X = newcolum(X, N)
        #
        X=fill_new_feature(key,value, N, X)
        #
        print("-------------------------------")

    #print("new=====",new_feature)
    for i in range(len(X.columns)):
        if X.columns[i] not in  new_feature :
             X[X.columns[i]].fillna(X[X.columns[i]].mean(), inplace=True)
             X[X.columns[i]].replace(0, X[X.columns[i]].mean(), inplace=True)
             #StandardScaler.fit_transform(X[X.columns[i]])    #print(X)

    normalizedColumns = list(XtoNormalize.columns)

    XtoNormalize = X[normalizedColumns]
    XtoNormalize = scaler.fit_transform(XtoNormalize)

    XtoNormalize = pd.DataFrame(XtoNormalize,columns=normalizedColumns)
    #print(X['budget'])
    X = X.drop(labels=normalizedColumns,axis=1)
    X = pd.concat([X,XtoNormalize],axis=1)
    #print(X['budget'])

    return X,scaler


def pre_test(test,train,scaler):
    #scaler = StandardScaler()
    colsToDrop = ['homepage', 'id', 'original_language', 'title_x', 'title_y', 'overview', 'original_title',
                  'status', 'tagline', 'movie_id']
    test = test.drop(labels=colsToDrop, axis=1)
    colsToReplace = {'genres': 'name',
                     'keywords': 'name',
                     'production_companies': 'name',
                     'production_countries': 'name',
                     'spoken_languages': 'name',
                     'cast': 'name',
                     'crew': 'name'}
    test['release_date'] = (pd.DatetimeIndex(test['release_date']).year)
    XtoNormalize = test.drop(labels=list(colsToReplace.keys()), axis=1)
    new_test_feature=[]
    #Check feature found or not
    for i in range(len(train.columns)):
        if train.columns[i] not in  test.columns :
            test[train.columns[i]]=0
            new_test_feature.append(train.columns[i])

    #pre colum of jsion

    for key, value in colsToReplace.items():
        feature = test[key]

        for i in range(len(feature)):

            #print(feature.iloc[i])
            try:
                elements = json.loads(feature.iloc[i])
            except:
                print('cannot load json')
                print('value is',feature.iloc[i])

                print('feature',key,'length is',len(feature))

                continue

            for j in range(len(elements)):
                M = elements[j][value]
                if M in test.columns:
                    test.loc[i, M] = 1

        test = test.drop(labels=[key], axis=1)



    #date

    # print("new=====",new_feature)

    for i in range(len(test.columns)):
        if test.columns[i] not in new_test_feature:
            test[test.columns[i]].fillna(test[test.columns[i]].mean(), inplace=True)
            test[test.columns[i]].replace(0, test[test.columns[i]].mean(), inplace=True)
            # StandardScaler.fit_transform(X[X.columns[i]])    #print(X)

    normalizedColumns = list(XtoNormalize.columns)

    XtoNormalize = test[normalizedColumns]
    print (XtoNormalize.shape)
    XtoNormalize = scaler.transform(XtoNormalize)


    XtoNormalize = pd.DataFrame(XtoNormalize, columns=normalizedColumns)

    test = test.drop(labels=normalizedColumns, axis=1)
    test = pd.concat([test, XtoNormalize], axis=1)

    #print("test=",test.shape)
    return test


#TrainTimeList=[]

'''
colsToReplace = {'genres': 'name',
                     'keywords': 'name',
                     'production_companies': 'name',
                    'spoken_languages': 'name',
                    'cast': 'name',
                      'crew': 'name'}
#}
#from sklearn import preprocessing
#for key, value in colsToReplace.items():'''
'''X,Y=ReadDataset('tmdb_5000_movies_classification.csv', 'tmdb_5000_credits.csv')
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
feature=X['keywords']
#print('key',key)

colsToReplace = {'genres': 'name',
                 'keywords': 'name',
                 'production_companies': 'name',
                 'production_countries': 'name',
                 'spoken_languages': 'name',
                 'cast': 'name',
                 'crew': 'name'}

for key, value in colsToReplace.items():
    feature = X[key]
    categories = CategoriesCount(feature, value)
    N = topfeat(categories)
    X = newcolum(X, N)
    X_ = fill_new_feature(key, value, N, X)
    print("-------------------------------")
#categories = CategoriesCount(feature,'name')
N=topfeat(categories)
X=newcolum(X,N)
X=fill_new_feature('keywords','name',N,X)
#N = encod(categories);
#M=replacewithlable (feature,'name',N,categories)
#X=binary(X,feature,'name')
#print(X)
    #encod(categories)


#'''