import RegressionModels as RegModels
import ClassificationModels as ClassModels
import os

def Run(rootFolder='',train_val_split=True):

    RegModels.Run(rootFolder=rootFolder,train_val_split=train_val_split)
    ClassModels.Run(rootFolder=rootFolder,train_val_split=train_val_split,pca_n_components=0.98)

Run('Train_Val_100_0/',False)