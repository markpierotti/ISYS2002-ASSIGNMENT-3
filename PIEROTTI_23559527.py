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
from sklearn.preprocessing import MinMaxScaler

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
#duplicates = data.duplicated()
#print(f'Number of duplicate rows: {duplicates.sum()}')
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

