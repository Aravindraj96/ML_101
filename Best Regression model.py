# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 09:08:53 2021

@author: adgbh
"""
"""

Import Libraries

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""

Import Datasets

"""
dataset = pd.read_csv("Data_1.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
"""

Train test split

"""
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 200)
"""

MultiVariate Linear Regression

"""
from sklearn.linear_model import LinearRegression
Linear_regressor = LinearRegression()
Linear_regressor.fit(x_train, y_train)
y_pred_MR = Linear_regressor.predict(x_test)
np.set_printoptions(precision=2)
concat_result= (y_pred_MR.reshape(len(y_pred_MR),1), y_test.reshape(len(y_test),1))
print (np.concatenate(concat_result,1))
"""

Polynomial Regression

"""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
Polynomial_Regressor = PolynomialFeatures(degree = 4)
x_poly = Polynomial_Regressor.fit_transform(x_train)
reg = LinearRegression()
reg.fit(x_poly, y_train)
y_pred_PR = reg.predict(Polynomial_Regressor.transform(x_test))
np.set_printoptions(precision=2)
concat_result_PR = (y_pred_PR.reshape(len(y_pred_PR),1), y_test.reshape(len(y_test),1))
print (np.concatenate(concat_result_PR,1))
"""

Support Vector Regression

"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train_FS = sc_x.fit_transform(x_train)
y_train_FS = y_train.reshape(len(y_train),1)
y_train_FS = sc_y.fit_transform(y_train_FS)
from sklearn.svm import SVR
SVR_Regressor = SVR(kernel = 'rbf')
SVR_Regressor.fit(x_train_FS, y_train_FS)
y_pred_SVR = sc_y.inverse_transform(SVR_Regressor.predict(sc_x.transform(x_test)))
np.set_printoptions(precision=2)
concat_result_SVR = (y_pred_SVR.reshape(len(y_pred_SVR),1), y_test.reshape(len(y_test),1))
print (np.concatenate(concat_result_SVR,1))
"""

Decision tree algorithm

"""
from sklearn.tree import DecisionTreeRegressor
DTR_Regressor = DecisionTreeRegressor(random_state=200)
DTR_Regressor.fit(x_train, y_train)
y_pred_DTR = DTR_Regressor.predict(x_test)
np.set_printoptions(precision=2)
concat_result_DTR = (y_pred_DTR.reshape(len(y_pred_DTR),1), y_test.reshape(len(y_test),1))
print (np.concatenate(concat_result_DTR,1))
"""

Random Forest Algorithm

"""
from sklearn.ensemble import RandomForestRegressor
RFR_Regressor = RandomForestRegressor(n_estimators=100, random_state=200)
RFR_Regressor.fit(x_train, y_train)
y_pred_RFR = RFR_Regressor.predict(x_test)
np.set_printoptions(precision=2)
concat_result_RFR = (y_pred_RFR.reshape(len(y_pred_RFR),1), y_test.reshape(len(y_test),1))
print (np.concatenate(concat_result_RFR,1))
"""

Computing the accuracy of various models 

"""
print("Multi Variate Regression")
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred_MR))
print("Plynomial Regression")
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred_PR))
print("Support Vector Regression")
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred_SVR))
print("Decision tree Regression")
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred_DTR))
print("Random Forest Regression")
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred_RFR))
