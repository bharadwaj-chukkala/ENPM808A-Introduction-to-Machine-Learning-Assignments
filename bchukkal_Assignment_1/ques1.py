# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 21:58:33 2022

@author: Bharadwaj Chukkala | 118341705 | bchukkal@umd.edu
"""

import string
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

## Loading the Data ##
Sales_Data = pd.read_csv(r"C:\Users\bhara\OneDrive\Desktop\ENPM808A\Week1_Assignment\mlr05.csv")
print(Sales_Data)
print('\v')

print(Sales_Data.describe())
print('\v')

## Prepare/Segregate Data
X = Sales_Data[['X2', 'X3', 'X4', 'X5', 'X6']] #Input with 5 dimensions
Y = Sales_Data['X1'] #Corresponding Output

## Splitting DataSet into Testing And Training sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=7, train_size =20, shuffle=False, random_state=0)

## Training the Regression Model ##
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

## Checking the weights chosen to fit the model
weights = regressor.coef_
weight_data = zip(X.columns, weights)
weights_dataframe = pd.DataFrame(weight_data, columns = ['Feature','Weight'])
print(weights_dataframe)
print('\v')

## Output Predictions for X_Test ##
Y_predict = regressor.predict(X_test)

## Comparing the Predicted Values with Test Values to see closeness
Closeness_Data = zip(Y_predict, Y_test) 
Closeness_dataframe = pd.DataFrame(Closeness_Data, columns = ['Predicted Output', 'Actual Output'])
print(Closeness_dataframe)
print('\v')

## Calculating Erro values betweeen Predicted Outputs and actual Test Outputs
print('Mean Absolute Error:', mean_absolute_error(Y_test, Y_predict))
print('Mean Squared Error:', mean_squared_error(Y_test, Y_predict))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_test, Y_predict)))



