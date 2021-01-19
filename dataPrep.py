# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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



"""