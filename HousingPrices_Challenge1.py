# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:53:20 2018

@author: 134476
"""

# Multiple Linear Regression to predict housing prices using Backward elimination

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data_Meetup2.csv')
#Checking first 5 rows of dataset
dataset.head()

#checking if any value is null
print(sum(dataset.isnull().any()))

#summary statistics of dataset
dataset.describe()
dataset.info()

# Removing id,date columns from dataset
dataset = dataset.drop(['id','date'],axis=1)

#Checking corrleation between variables of dataset
dataset.corr()

#creating heat map to check correlations.
plt.figure(figsize=(5,5))
ax = sns.heatmap(dataset.corr(),annot = True)
ax.set_title('correlation matrix',fontsize = 25)

# Indicating dependent variable Price as y
y = dataset['price']
y = np.array(y)

# Predictor variables
X = dataset.drop(['price'],axis=1)
X = np.array(X)

import statsmodels.formula.api as sm
X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#removing X5 and running backward elimination again
X_opt = X[:,[0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#if included this accuracy will be 73.15%
#X_opt = X[:,[0,1,2,5,6,7,8,9,10,11,12,13,14,15,16,17]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_opt_train, X_opt_test, y_opt_train, y_opt_test = train_test_split(X_opt, y, test_size = 0.1,random_state=2)

#Fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_opt_train, y_opt_train)

#Predicting the test set results
y_pred = regressor.predict(X_opt_test)


#Regression Evaluation Metrics
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from math import sqrt

meanAbsoluteError = mean_absolute_error(y_opt_test,y_pred)
print('Mean Absolute Error:',meanAbsoluteError)

meanSquaredError = mean_squared_error(y_opt_test,y_pred)
print('mean squared error :', meanSquaredError)

rootMeanSquaredError = sqrt(meanSquaredError)
print('root mean squared error :', rootMeanSquaredError)

r2_training = regressor.score(X_opt_train,y_opt_train)
print('R2_training:',r2_training)

r2_test = regressor.score(X_opt_test,y_opt_test)
print('R2_Test :', r2_test)

# Distribution and comparison of test values and Predicted values
ax1 = sns.distplot(pd.DataFrame(y_opt_test),hist = False, color = 'r',label = 'Actual Value')
sns.distplot(pd.DataFrame(y_pred),hist=False,color='b',label='fitted values',ax=ax1)





