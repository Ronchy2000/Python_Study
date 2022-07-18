# -*- coding: utf-8 -*-
# @Time    : 2022/7/18 10:01
# @Author  : Ronchy_LU
# @Email   : rongqi1949@gmail.com
# @File    : Random Forest Regresion.py
# @Software: PyCharm

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Salaries.csv')

column_head = list(df.columns.values)
print("column_head: ",column_head)


x= df.iloc[:,1:-1].astype(int)   #ignore last column
y= df.iloc[:, -1 :].astype(int) #ignore all columns except the last one

print("df['level']:",df['level'])
print('------------------------------')
print("df.ix[0]:",df.iloc[0])
print('------------------------------')
# print("type(x):",type(x))   #DataFrame

#----------------------------------------------------------------------------------
# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.ensemble import RandomForestRegressor

# create regressor object
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

# fit the regressor with x and y data
regressor.fit(x, y)

print(regressor)

Y_pred = regressor.predict(np.array([6.5]).reshape(1, 1)) # test the output by changing values
print(Y_pred)

# Visualising the Random Forest Regression results

# arrange for creating a range of values
# from min value of x to max
# value of x with a difference of 0.01
# between two consecutive values
print("min(x):",min(x))
print("min(x) type:",type(min(x)))
print("max(x):",max(x))
print("max(x) type:",type(max(x)))
#--------------------------------------
#str  -> int
'''
X_grid = np.arange(int(min(x)), int(max(x)),0.01)

# reshape for reshaping the data into a len(X_grid)*1 array,
# i.e. to make a column out of the X_grid value
X_grid = X_grid.reshape((len(X_grid), 1))

# Scatter plot for original data
plt.scatter(x, y, color = 'blue')

# plot predicted data
plt.plot(X_grid, regressor.predict(X_grid),
		color = 'green')
plt.title('Random Forest Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
'''
