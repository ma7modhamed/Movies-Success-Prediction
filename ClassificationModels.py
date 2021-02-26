from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt; plt.rcdefaults()
from ttictoc import TicToc
import pre_hot as pre
import numpy as np
t = TicToc()
import  os
from sklearn.externals import joblib
import pandas as pd

TrainTimeList=[]

def fit_model(X,Y,model,filename):
    if (os.path.exists(filename)):
        resmodel=joblib.load(filename)
        #print("ffffffff")
    else:
        t.tic()
        resmodel=model.fit(X,Y)
        t.toc()
        TrainTimeList.append(t.elapsed)
        print("total train =",t.elapsed)
        joblib.dump(resmodel,filename)

    return resmodel

#classsssifier
from sklearn.linear_model import LogisticRegression
def Class_model(X_T ,Y_T,X_Test,Y_test,rootFolder=''):
    TestTimeList=[]
    accracy_bar=[]
    print('-------Logistic Regression-------')
    logistic = LogisticRegression(C=10000,multi_class='auto',solver='liblinear')
    #logistic.fit(X_T,Y_T)
    logistic = fit_model(X_T,Y_T,logistic,rootFolder+'LogisticRegression')
    y = logistic.predict(X_Test)
    acc = accuracy_score(Y_test, y) * 100
    print("Accuracy Logistic Regression : ", acc)
    accracy_bar.append(acc)
    print("Confusion Matrix for Logistic Regression : ")
    print(confusion_matrix(Y_test, y))

    ''''
    print("      ----------- KNN ------------")
    model = KNeighborsClassifier(n_neighbors=20)
    t.tic()
    #model.fit(X_T,Y_T)
    model = fit_model(X_T,Y_T,model,rootFolder+'Knn')
    t.toc()
    TrainTimeList.append(t.elapsed)
    print("total train =", t.elapsed)
    t.tic()
    y=model.predict(X_Test)
    t.toc()

    TestTimeList.append(t.elapsed)
    print("total test =", t.elapsed)

    acc = accuracy_score(Y_test, y) * 100
    print("Accuracy Knn : ", acc)
    accracy_bar.append(acc)
    print("Confusion Matrix for Knn : ")
    print(confusion_matrix(Y_test, y))

    '''
    print("      ------- Tree----------")
    tree = DecisionTreeClassifier(random_state=100, max_depth=15, min_samples_leaf=4)
    R_tree= fit_model(X_T, Y_T, tree, rootFolder+"tree_model_15")
    t.tic()
    y_pred = R_tree.predict(X_Test)
    t.toc()
    TestTimeList.append(t.elapsed)
    print("total test =", t.elapsed )
    acc = accuracy_score(Y_test, y_pred) * 100
    print("Accuracy tree : ", acc)
    accracy_bar.append(acc)
    print("Confusion Matrix for tree : ")
    print(confusion_matrix(Y_test, y_pred))

    print("    ---------- SVM ------------")
    clf = svm.SVC(kernel='linear',gamma='auto')
    R=fit_model(X_T,Y_T,clf,rootFolder+"svm_model_linear ")

    t.tic()
    y_p=R.predict(X_Test)
    t.toc()
    TestTimeList.append(t.elapsed)
    print("total test =", t.elapsed)
    acc=accuracy_score(Y_test, y_p) * 100
    print("Accuracy : ", acc)
    accracy_bar.append(acc)

    print("Confusion Matrix for Svm: ")
    print(confusion_matrix(Y_test, y_p))


    print("      ----- addabos tree ---------")

    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15),
                             algorithm="SAMME.R",
                             n_estimators=50)
    Res_bdt = fit_model(X_T, Y_T, bdt, rootFolder+"addbos_tree_model_15")

    t.tic()
    y_prediction = Res_bdt.predict(X_Test)
    t.toc()
    TestTimeList.append(t.elapsed)
    print("total test =",t.elapsed)

    acc=accuracy_score(Y_test, y_prediction) * 100
    print("Accuracy addapost tree : ", acc)
    accracy_bar.append(acc)

 #   print("Train=",TrainTimeList)
    #plot_time(TrainTimeList)
#    print("Test=",TestTimeList)

    #plot_time(TestTimeList ,'Test')
    #plot_time(accracy_bar,'Accuracy')


def ModelWithPCA(X_Train ,Y_T,X_Test,Y_test,n_components,rootFolder=''):
    print("    ------------ PCA ----------------")

    train_pca,test_pca=pca(X_Train,X_Test,n_components)

    print('After Applying PCA With n_components = ',n_components)

    print('The data feature was',X_Train.shape,'has reduced to be',train_pca.shape)
    test_Time_List=[]
    accuracy=[]
    print(" ******** SVM _PCA :")
    svm_pca = svm.SVC(decision_function_shape='ovr', gamma='auto')
    print(train_pca.shape)
    svm_result = fit_model(train_pca, Y_T, svm_pca, rootFolder+ "svm_pca_"+str(n_components))
    #clf.fit(train_pca,Y_Train)
    t.tic()
    y_p = svm_result.predict(test_pca)
    t.toc()
    test_Time_List.append(t.elapsed)
    acc=accuracy_score(Y_test, y_p) * 100
    accuracy.append(acc)
    print("PCA--Accuracy svm : ",acc )
    print("confusion_matrix :")
    print(confusion_matrix(Y_test, y_p))

    print(" ******** Tree _PCA :")
    tree_pca=DecisionTreeClassifier()#criterion = "entropy", min_samples_leaf=5)
    tree_result = fit_model(train_pca, Y_T, tree_pca, rootFolder+"tree_pca_"+str(n_components))
    t.tic()
    y_p = tree_result.predict(test_pca)
    t.toc()
    test_Time_List.append(t.elapsed)
    acc=accuracy_score(Y_test, y_p) * 100
    accuracy.append(acc)
    print("PCA--Accuracy tree : ", acc)
    print("confusion_matrix :")
    print(confusion_matrix(Y_test, y_p))

    print(" ******** KNN _PCA :")
    knn_pca= KNeighborsClassifier(n_neighbors=1)
    knn_result = fit_model(train_pca, Y_T, knn_pca,rootFolder+ "knnpca_"+str(n_components))
    t.tic()
    y_p = knn_result.predict(test_pca)
    t.toc()
    test_Time_List.append(t.elapsed)
    acc=accuracy_score(Y_test, y_p) * 100
    accuracy.append(acc)
    print("PCA--Accuracy knn : ", acc)
    print("confusion_matrix :")
    print(confusion_matrix(Y_test, y_p))

    print(" ******** AdaBoost _PCA :")
    bdt_pca= AdaBoostClassifier(DecisionTreeClassifier(max_depth=15),
                             algorithm="SAMME.R",
                             n_estimators=50)
    bdt_result = fit_model(train_pca, Y_T, bdt_pca,rootFolder+ "bdt_pca_"+str(n_components))
    t.tic()
    y_p = bdt_result.predict(test_pca)
    t.toc()
    test_Time_List.append(t.elapsed)
    acc=accuracy_score(Y_test, y_p) * 100
    accuracy.append(acc)
    print("PCA--Accuracy tree-adabost : ", acc)
    print("confusion_matrix :")
    print(confusion_matrix(Y_test, y_p))


    #plot_time(TrainTimeList,"Train")
    plot_time(test_Time_List,"Test")
    plot_time(accuracy,"Accuracy")


def plot_time(draw_list,str):

    objects = ('Svm', 'Tree', 'KNN', 'Adapost_tree')
    y_pos = np.arange(len(objects))
    performance = draw_list
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel(str+'in second')
    plt.title('Models')
    plt.show()


def pca(train,test,n_components):
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test=scaler.transform(test)
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(train)
    principalComponents_test=pca.transform(test)
    #print("pca",pca.explained_variance_ratio_)
    print("train",principalComponents.shape)
    print("test",principalComponents_test.shape)
    return  principalComponents,principalComponents_test


def Run(rootFolder='',train_val_split=True,pca_n_components=0.98):
    print("--------------------------------------------------")
    print("------------------ Classification Model -------------------")
    datasetpath = 'classification_train_data.csv'
    datasetscaler = 'datasetscaler'
    X_train, y_train = pre.ReadDataset('tmdb_5000_movies_classification.csv', 'tmdb_5000_credits.csv', 'rate')

    if train_val_split:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20)
    else:
        X_test, y_test = pre.ReadDataset('tmdb_5000_movies_testing_classification.xlsx',
                                         'tmdb_5000_credits_test.csv', 'rate')

    if (os.path.exists(rootFolder+datasetpath)):
        X_train = pd.read_csv(rootFolder+datasetpath)
        X_train = X_train.drop(labels=[X_train.columns[0]],axis=1)
        scaler = joblib.load(rootFolder+datasetscaler)
    else:
        X_train.index = range(len(X_train))
        X_train, scaler = pre.prefull(X_train)
        X_train.to_csv(rootFolder+datasetpath)

        joblib.dump(scaler,rootFolder+datasetscaler)

    #print(scaler)

    y_train = pre.class_label(y_train)

    #X_test, y_test = pre.ReadDataset('samples_tmdb_5000_movies_testing_classification.xlsx','samples_tmdb_5000_credits_test.csv','rate')
    y_test = pre.class_label(y_test)

    X_test.index = range(len(X_test))
    X_test=pre.pre_test(X_test,X_train,scaler)

    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20,shuffle=True)
    print ("train after_pre=",X_train.shape)
    print("test after pre=",X_test.shape)

    Class_model(X_train, y_train,X_test, y_test ,rootFolder)
    ModelWithPCA(X_train, y_train, X_test, y_test,pca_n_components,rootFolder)