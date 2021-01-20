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

dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
"""

Splitting train and test data

"""
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0 )
"""

Simple Linear Model - Training the train set

"""
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
"""

Simple Linear Regression - Test Set prediction

"""
Y_pred = regressor.predict(X_test)
print (Y_pred)
print (Y_test)
"""

Plotting the Training set result

"""
import matplotlib.pyplot as mpl
mpl.scatter(X_train, Y_train, color='red')
mpl.plot(X_train, regressor.predict(X_train), color = 'blue')
mpl.title('Salary vs Experience - Training set')
mpl.xlabel('Years of Experience')
mpl.ylabel('salary')
mpl.show
"""

Plotting the Training set result

"""
mpl.scatter(X_test, Y_test, color='green')
mpl.plot(X_test, regressor.predict(X_test), color = 'blue')
mpl.title('Salary vs Experience - Test set')
mpl.xlabel('Years of Experience')
mpl.ylabel('salary')
mpl.show
"""

Print the Salary for a specific year of experience

"""
yoe = int(input("Enter the Years of Experience"))
print (regressor.predict([[yoe]]))
print (regressor.coef_)
print (regressor.intercept_)