# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:14:06 2021

@author: adgbh
"""
"""

import Libraries

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""

import Dataset

"""
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
"""

Training the Decision tree model

"""
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x, y)
"""

Predicting a new value

"""
print (regressor.predict([[6.5]]))
"""

Visualising the Decision tree Regression

"""
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Decision Tree Algorithm')
plt.xlabel('Job Level')
plt.ylabel('Salaries')
