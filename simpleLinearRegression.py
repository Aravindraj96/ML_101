# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 00:50:25 2021

@author: adgbh
"""
"""

Import the required Libraries Numpy, Matplotlib and pandas.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""

Import a Dataset

Returns a Dependent and independent Matrix.

"""

dataset = pd.read_csv("Data.csv")
feature_matrix = dataset.iloc[:,:-1].values #Feature Matrix
print (feature_matrix)
dependent_matrix = dataset.iloc[:,-1].values  #Dependent Matrix
#print (y)
print ('#####################################################################')
"""

Taking are of missing data.

Returns Feature Matrix without missing data.

Missing data is filled by getting the average of all the other data

"""

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(feature_matrix[:, 1:3])
feature_matrix[:, 1:3] = imputer.transform(feature_matrix[:, 1:3])
print (feature_matrix)

"""

Encode Categorical Data using One Hot Encoding

"""
print ('#####################################################################')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
colTrans = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[0])],remainder='passthrough')
feature_matrix = np.array(colTrans.fit_transform(feature_matrix))
print (feature_matrix)
print ('#####################################################################')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dependent_matrix = le.fit_transform(dependent_matrix)
print (dependent_matrix)
print ('#####################################################################')

"""

Splitting Traning and Test Data set

"""
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(feature_matrix, dependent_matrix, test_size=0.2, random_state=1)
print (X_train)
print ('#####################################################################')
print (X_test)
print ('#####################################################################')
print (Y_test)
print ('#####################################################################')
print (Y_train)
print ('#####################################################################')


"""

Feature Scaling

"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print (X_train)
print ('#####################################################################')
print(X_test)
print ('#####################################################################')