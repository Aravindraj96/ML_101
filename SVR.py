# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 18:31:50 2021

@author: adgbh
"""
"""

import libraries

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""

Import dataset

"""
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
"""

Reshaping the label or Dependent Matrix

"""
print (x)
print ('#####################################################################')
print (y)
print ('#####################################################################')
y = y.reshape(len(y),1) # reshaping y since sklearn accepts(feature scaling) only 2d array 
print (y)
print ('#####################################################################')
"""

Feature scaling

Using two standard scalers because we have to reverse the scaling to the original value where x and y are differently scaled

"""

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x[:, :])
sc_y = StandardScaler()
y = sc_y.fit_transform(y[:, :])
print (x)
print ('#####################################################################')
print (y)
print ('#####################################################################')
"""

Training the SVR Model

"""
from sklearn.svm import SVR
svr_regressor = SVR(kernel = 'rbf') #Refers to the Guassian RBF kernel for the regressor
svr_regressor.fit(x, y)
print (sc_y.inverse_transform(svr_regressor.predict(sc_x.transform([[6.5]])))) # using the standard scaler to get the predicted salary in original scale
"""

plotting the results on a graph

"""
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y) , color='red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(svr_regressor.predict(x)), color='blue')
plt.xlabel('Job Level')
plt.ylabel('Salary')
plt.title('SVR Model to predict salary')
plt.show()
"""

Smoothing the curve

"""
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y) , color='black')
plt.plot(x_grid, sc_y.inverse_transform(svr_regressor.predict(sc_x.transform(x_grid))), color='green') # we use a transform here because the x_grid is reverted to unscaled value
plt.xlabel('Job Level')
plt.ylabel('Salary')
plt.title('SVR Model to predict salary')
plt.show()