# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 08:03:10 2021

@author: adgbh
"""
"""

import libraries

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""

Import Data set

"""
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
"""

Training the Random state model

"""
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) # n_estimator determines the number of trees
regressor.fit(x, y)
"""

Predict the value

"""
print(regressor.predict([[6.5]]))
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