from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression
# from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


x = [i for i in range(21)[1:]]

# actual and error data - experimental and simulated
y1 = [0.8, 3.9, 9.1, 16, 24.7, 35.8, 49, 63.8, 81.2, 99.8, 121,144, 169.2, 196,225.4,256, 289.4, 324,361.3,400]
# error data only - simulated only
y2 = [0.8, 3.9, 9.1, 15.8, 24.7, 35.8, 49.1, 63.8, 81.2, 99.8, 121.3,143.4, 169.2, 195.8,225.4,255.8, 289.4, 324,361.3,400.2]

# print(len(x), len(y1))
# =============================================================================
# a = [1,2,3]
# print(a[1:2]) -- returns array 2
# print(a[1])  -- returns num 2
# =============================================================================

# =============================================================================
# data =[]
# for i,j in zip(x,y1):
#     data.append([i,j])
# =============================================================================
    
data =[]
for i in x:
    data.append([i])
    
# print(data)

poly = PolynomialFeatures(degree = 2) 
x_poly = poly.fit_transform(data)
# print(x_poly)
poly.fit(x_poly,y1)

lin = LinearRegression() 
lin.fit(x_poly, y1) 

# Visualising the Polynomial Regression results 
plt.scatter(x, y1, color = 'blue') 

x = np.array(x).reshape(-1,1)
plt.plot(x, lin.predict(poly.fit_transform(x)), color = 'red') 
plt.title('Polynomial Regression') 
plt.xlabel('x-input') 
plt.ylabel('y1-with exp data') 

plt.show() 

# ===============================================================================
# for y2
poly2 = PolynomialFeatures(degree = 2) 
x_poly = poly2.fit_transform(data)
# print(x_poly)
poly2.fit(x_poly,y2)

lin2 = LinearRegression() 
lin2.fit(x_poly, y2) 

# Visualising the Polynomial Regression results 
plt.scatter(x, y2, color = 'blue') 

x = np.array(x).reshape(-1,1)
plt.plot(x, lin2.predict(poly.fit_transform(x)), color = 'red') 
plt.title('Polynomial Regression') 
plt.xlabel('x-input') 
plt.ylabel('y2-with sim data') 

plt.show() 


# =============================================================================
# PREDICTION USING POLYNOMIAL REGRESSION
# =============================================================================
x_new = [[25],[26],[27]]
plt.figure()
plt.plot(x_new, lin2.predict(poly2.fit_transform(x_new)),color='blue', label = 'sim-only')
plt.plot(x_new, lin.predict(poly.fit_transform(x_new)),color='red', label = 'sim+exp')
plt.legend()
plt.show()
# =============================================================================
# DATA WITH EXPERIMENTAL RESULTS IMPUTED GIVES BETTER PREDICTIONS
# =============================================================================
print(lin2.predict(poly2.fit_transform(x_new)))
print(lin.predict(poly.fit_transform(x_new)))
