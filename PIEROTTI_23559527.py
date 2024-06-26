#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:45:34 2024

@author: mark_pierotti
"""

# Import libraries
# The following libraries provide the tools necessary for data wrangling,
# normalisation, indexation, date standardisation, pre-processing,
# and a few things ill add aling the way

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from sklearn.metrics import mean_squared_error, mean_absolute_error

# import dataset. changed file location from assessment 2 to asesssment 3 folder. pd.read_csv() function makes csv a dataframe
data = pd.read_csv('/Users/pierotti/SCU/ISYS2002_Data Wrangling and Advanced Analytics/Assignment 3/GlobalTemperatures.csv')

# visualise raw data as a Line plot for Land Average Temperature over time, same as Assignment 1 and 2
plt.figure(figsize=(14, 8))
plt.plot(data['dt'], data['LandAverageTemperature'], label='Land Average Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Raw Land Average Temperature Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.xticks(pd.date_range(start=data.index.min(), end=data.index.max(), freq='25Y'), rotation=45)
plt.legend()
plt.show()



# Drop() function to exclude rows 1 as they are headers, include row 1202. 
# *note - data.drop(index = range(2, 1203), inplace=True) by itself threw an error: not found in axis.
# took a while but worked out where logically it has to be placed, also changed brackets and removed comma, replace with colon
# the range was being screwy as well, played around with the upper limit until it started at correct date.
# also went back to early modules to ensure consistant code formatting/syntax. Apply data.drop to Uncertainty columns
data.drop(index = data.index[0:1200], inplace=True)

# drop() function to remove unnecessary columns
data.drop(columns=[
    'LandAverageTemperatureUncertainty',
    'LandMaxTemperatureUncertainty',
    'LandMinTemperatureUncertainty',
    'LandAndOceanAverageTemperatureUncertainty'], inplace = True)

# Drop_duplicate rows / values if found
data.drop_duplicates(inplace=True)
print(data) # Number of duplicate rows: 0


# Rename 'dt' column to 'Date' and added spaces to other column headers with data.rename from pandas - logic was wrong, relocated data.rename
# to where it is now, before index is set

data.rename(columns = {'dt': 'Date'}, inplace = True)
data.rename(columns = {'LandAverageTemperature': 'Land Average Temperature'}, inplace = True)
# data.rename(columns = {'LandAverageTemperatureUncertainty': 'Land  Average Temperature Uncertainty'}, inplace = True)
# data.rename(columns = {'LandMaxTemperature': 'Land Max Temperature'}, inplace = True)
# data.rename(columns = {'LandMaxTemperatureUncertainity': 'Land Max Temperature Uncertainity'}, inplace = True)
# data.rename(columns = {'LandMinTemperature': 'Land Min Temperature'}, inplace = True)
# data.rename(columns = {'LandMinTemperatureUncertainity': 'Land Min Temperature Uncertainity'}, inplace = True)
# data.rename(columns = {'LandAndOceanAverageTemperature': 'Land And Ocean Average Temperature'}, inplace = True)
# data.rename(columns = {'LandAndOceanAverageTemperatureUncertainity': 'Land And Ocean Average Temperature Uncertainity'}, inplace = True)

# Print to check if it works. It does work, print code commented out.
print(data)

# Apply to_datetime to convert date to a string - pd.to_datetime(data['Date']) included hours, 
# .dt.date attribute added to remove hours, leaving only yyyy-mm-dd

# Connvert column to datetime - indocating the format is mixed
data['Date'] = pd.to_datetime(data['Date'], format = 'mixed', dayfirst=True).dt.date

# Apply indexing to column A - 'Date'.
data.set_index('Date', inplace=True)

# Print to see if it works. It does work, print code commented out.
print(data)

# Normalize temperature columns [0,1] using MinMaxScaler - excluding normalisation of related uncertainity columns
# as they will not be used as output variables

scaler = MinMaxScaler()
data[['Land Average Temperature', 'Land Max Temperature', 'Land Min Temperature', 'Land And Ocean Average Temperature']] = scaler.fit_transform(
data[['Land Average Temperature', 'Land Max Temperature', 'Land Min Temperature', 'Land And Ocean Average Temperature']])

# print to see if it works. It does work, print code commented out.
print(data)

# Visualize the processed data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Land Average Temperature'], label='Land Average Temperature', marker='o')
#plt.plot(data.index, data['Land Max Temperature'], label='Land Max Temperature', marker='o')
#plt.plot(data.index, data['Land Min Temperature'], label='Land Min Temperature', marker='o')
#plt.plot(data.index, data['Land And Ocean Average Temperature'], label='Land And Ocean Average Temperature', marker='o')




# Neural Network part

# Predictors (X) and Target Variable (y) defined from dataset
X = data[['Land Average Temperature', 'Land Max Temperature', 'Land Min Temperature', 'Land And Ocean Average Temperature']].values
y = data[['Land Average Temperature']].values

# Feature scaling using MinMaxScaler *NOTE - running the MinMaxScaler below threw an error: ValueError: Expected a 2-dimensional container but got <class 'pandas.core.series.Series'> instead. Pass a DataFrame containing a single row (i.e. single sample) or a single column (i.e. single feature) instead.
# Asked chatGPT to explain the error by copying the output in the iPython Console - this is referred to in the assessment in Appendix A
scaler = MinMaxScaler() 	
scaler.fit(X)
X_norm = scaler.transform(X)


# chatGPT code addition below


# Select the target variable and ensure it's a DataFrame
y = data[['Land Average Temperature']] 

# Initalise the Scaler
scaler = MinMaxScaler()

# Fit the scaler
scaler.fit(y)

# Transform the data
y_scaled = scaler.transform(y)
# Declare how many times the training sqquence will run i.e. 5.
num_runs = 5  
for run in range(num_runs):
    print(f"Run {run + 1}:")
    
# Split dataset into train, validation and test sets
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.2, random_state=42)

# Create a sequential neural network model with input, hidden and output layers
model = Sequential()
model.add(InputLayer(input_shape=(4)))
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=20, activation='relu'))
model.add(Dense(1, activation='linear'))

# Model compilation using 'mean squared error' for loss, 'adam' optimiser for gradient descent
# and Performance metrics using mean absolute square.
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# Model training
model.fit(X_train, y_train, batch_size = 20, epochs = 5, verbose=1)

# Model evaluation
y_pred = model.predict(X_test)
MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)
print("Training MSE: ", mean_squared_error(y_train, y_pred_train))
print("Validation MSE: ", mean_squared_error(y_val, y_pred_val))
print("Testing MSE: ", mean_squared_error(y_test, y_pred_test))



# Epoch trials to establish best model with least error for training to then test the data with.
# epochs_trial = [3, 5, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18]
# erros = []
# for epoch in epochs_trial:
#     model = Sequential()
#     model.add(InputLayer(input_shape=(4)))
#     model.add(Dense(units=10, activation='relu'))
#     model.add(Dense(units=10, activation='relu'))
#     model.add(Dense(1, activation='linear'))
    
#     model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
#     model.fit(X_train, y_train ,batch_size = 20, epochs = 5, verbose=1)
    
#     y_pred = model.predict(X_val)
#     MAE = mean_absolute_error(y_val, y_pred)
#     print("Validation Error for epoch: ", epoch, " is ", MAE)
#     erros.append(MAE)

# best_epoch = epochs_trial[np.where(erros == min(erros))[0][0]]
# print("Best epoch is: ", best_epoch)

# Experiment to see what happens if it runs 100 epochs(!) ive optimised the M1 Apple Silicon chip to leverage the GPU and Neural Network Architecture, this should be interesting.
epochs_trial = list(range(1, 101))
erros = []
for epoch in epochs_trial:
    model = Sequential()
    model.add(InputLayer(input_shape=(4)))
    model.add(Dense(units=33, activation='relu'))
    model.add(Dense(units=2, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    model.fit(X_train, y_train ,batch_size = 20, epochs = 5, verbose=1)
    
    y_pred = model.predict(X_val)
    MAE = mean_absolute_error(y_val, y_pred)
    print("Validation Error for epoch: ", epoch, " is ", MAE)
    erros.append(MAE)

best_epoch = epochs_trial[np.where(erros == min(erros))[0][0]]
print("Best epoch is: ", best_epoch)