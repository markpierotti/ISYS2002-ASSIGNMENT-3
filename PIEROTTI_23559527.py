#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:45:34 2024

@author: mark_pierotti
"""

# import libraries
# the following libraries provide the tools necessary for data wrangling,
# normalisation, indexation, date standardisation, pre-processing,
# and a few things ill add aling the way

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import InputLayer
from sklearn.metrics import mean_squared_error
# import dataset. changed file location from assessment 2 to asesssment 3 folder
# pd.read_csv() function makes csv a dataframe
data = pd.read_csv('/Users/pierotti/SCU/ISYS2002_Data Wrangling and Advanced Analytics/Assignment 3/GlobalTemperatures.csv')

# drop() function to exclude rows 1 as they are headers, include row 1202. 
# *note - data.drop(index = range(2, 1203), inplace=True) by itself threw an error: not found in axis.
# took a while but worked out where logically it has to be placed, also changed brackets and removed comma, replace with colon
# the range was being screwy as well, played around with the upper limit until it started at correct date.
# also went back to early modules to ensure consistant code formatting/syntax. Apply data.drop to Uncertainty columns

data.drop(index = data.index[0:1200], inplace=True)
data.drop(columns=['LandAverageTemperatureUncertainty', 'LandMaxTemperatureUncertainty', 'LandMinTemperatureUncertainty', 'LandAndOceanAverageTemperatureUncertainty'], inplace = True)
#print(data)


# check for duplicates in dataset
# no duplicates found, so code commented out
data.drop_duplicates(inplace=True)
print(data)
# Number of duplicate rows: 0

# Rename 'dt' column to 'Date' and added spaces to other column headers with data.rename from pandas - logic was wrong, relocated data.rename
# to where it is now, before index is set

data.rename(columns = {'dt': 'Date'}, inplace = True)
data.rename(columns = {'LandAverageTemperature': 'Land Average Temperature'}, inplace = True)
data.rename(columns = {'LandAverageTemperatureUncertainty': 'Land  Average Temperature Uncertainty'}, inplace = True)
data.rename(columns = {'LandMaxTemperature': 'Land Max Temperature'}, inplace = True)
data.rename(columns = {'LandMaxTemperatureUncertainity': 'Land Max Temperature Uncertainity'}, inplace = True)
data.rename(columns = {'LandMinTemperature': 'Land Min Temperature'}, inplace = True)
data.rename(columns = {'LandMinTemperatureUncertainity': 'Land Min Temperature Uncertainity'}, inplace = True)
data.rename(columns = {'LandAndOceanAverageTemperature': 'Land And Ocean Average Temperature'}, inplace = True)
data.rename(columns = {'LandAndOceanAverageTemperatureUncertainity': 'Land And Ocean Average Temperature Uncertainity'}, inplace = True)

# print to check if it works. It does work, print code commented out.
#print(data)

# apply to_datetime to convert date to a string - pd.to_datetime(data['Date']) included hours, 
# .dt.date attribute added to remove hours, leaving only yyyy-mm-dd

data['Date'] = pd.to_datetime(data['Date']).dt.date

# print to see if it works. It does work, print code commented out.
#print(data)
#

# apply indexing to column A - 'Date'. pandas library already loaded. note: changed 'dt' to 'Date' to
# reflect the change of data.rename relocation in sequence

data.set_index('Date', inplace=True)

# print to see if it works. It does work, print code commented out. *when code is commented out, index
# added to dataframe as individual column, code modifies Date column to be Row Label of Dataframe
# print(data)


# normalize temperature columns [0,1] using MinMaxScaler - excluding normalisation of related uncertainity columns
# as they will not be used as output variables

scaler = MinMaxScaler()
data[['Land Average Temperature', 'Land Max Temperature', 'Land Min Temperature', 'Land And Ocean Average Temperature']] = scaler.fit_transform(
data[['Land Average Temperature', 'Land Max Temperature', 'Land Min Temperature', 'Land And Ocean Average Temperature']])

# print to see if it works. It does work, print code commented out.
#print(data)

# Neural Network part

# Predictors (X) and Target Variable (y) defined

X = data[['Land Average Temperature', 'Land Max Temperature', 'Land Min Temperature', 'Land And Ocean Average Temperature']].values
y = data[['Land Average Temperature']].values

# Standardisation of features *NOTE - running the MinMaxScaler below threw an error: ValueError: Expected a 2-dimensional container but got <class 'pandas.core.series.Series'> instead. Pass a DataFrame containing a single row (i.e. single sample) or a single column (i.e. single feature) instead.
# asked chatGPT to explain the error by copying the output in the iPython Console - this is referred to in the assessment in Appendix A
scaler = MinMaxScaler() 	
scaler.fit(X)
X_norm = scaler.transform(X)
# chatGPT code addition below


# Select the target variable and ensure it's a DataFrame
y = data[['Land Average Temperature']] # Note the double brackets

# Initalise the Scaler
scaler = MinMaxScaler() 	
# Fit the scaler
scaler.fit(y)
# Transform the data
y_scaled = scaler.transform(y)

# split dataset into train, validation and test sets
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.2, random_state=42)

# Create the model
model = Sequential()
model.add(InputLayer(input_shape=(4)))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(1, activation='linear'))

# Model compilation
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.fit(X_train, y_train ,batch_size = 20, epochs = 5, verbose=1)

y_pred = model.predict(X_test)
MSE = mean_squared_error(y_test, y_pred)
print("Testing MSE: ", MSE)

# y_pred_train = model.predict(X_train)
# y_pred_val = model.predict(X_val)
# y_pred_test = model.predict(X_test)
# print("Training MSE: ", mean_squared_error(y_train, y_pred_train))
# print("Validation MSE: ", mean_squared_error(y_val, y_pred_val))
# print("Testing MSE: ", mean_squared_error(y_test, y_pred_test))