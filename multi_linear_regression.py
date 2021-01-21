# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 09:18:15 2021

@author: adgbh
"""

"""

Import Libraries

"""
import pandas as pa
import numpy as np
import matplotlib.pyplot as plt
"""

import Datasets

"""
dataset = pa.read_csv("50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
"""

Encode catagorical data

"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])],remainder='passthrough')
x = np.array(ct.fit_transform(x))
"""

Seperate test and train datasets

"""
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

"""

Train the Multiple Linear regression model wi the training set

"""
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
"""

Predicting the test result

"""
Y_pred = regressor.predict(X_test)
np.set_printoptions(precision = 2)
print (np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test), 1)),1))
"""

Predicting for a value 

"""
rd = float(input("Enter R&D Cost :"))
ad = float(input("Enter Administration cost : "))
mk = float(input("Enter Marketing cost : "))
print (regressor.predict([[1,0,0,rd,ad,mk]]))