# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 09:08:07 2021

@author: adgbh
"""

"""

Import Libraries

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""

Import datasets

"""
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
"""

Training the linear regression model on the whole dataset

"""
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)
"""

Plotting the linear model on Graph

"""
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Level vs Salary - Training set')
plt.xlabel('Job Level')
plt.ylabel('salary')
plt.show
"""

Training the Polynomial Regresion model

"""
from sklearn.preprocessing import PolynomialFeatures
p_regressor = PolynomialFeatures(degree = 4)
X_poly = p_regressor.fit_transform(x)
l_regressor = LinearRegression()
l_regressor.fit(X_poly, y)
"""

Plotting the linear model on Graph

"""
plt.scatter(x, y, color = 'green')
plt.plot(x, l_regressor.predict(p_regressor.fit_transform(x)), color = 'black')
plt.title('Level vs Salary - Training set Polynomial')
plt.xlabel('Job Level')
plt.ylabel('salary')
plt.show
"""

Predict the salary level

"""
print (regressor.predict([[6.5]]))
print (l_regressor.predict(p_regressor.fit_transform([[6.5]])))
