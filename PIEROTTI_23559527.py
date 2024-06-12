#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:45:34 2024

@author: mark_pierotti
"""

#import libraries
#the following libraries provide the tools necessary for data wrangling,
#normalisation, indexation, date standardisation, pre-processing,
#and a few things ill add aling the way
import pandas as pd

#import dataset. changed file location from assessment 2 to asesssment 3 folder
data = pd.read_csv('/Users/pierotti/SCU/ISYS2002_Data Wrangling and Advanced Analytics/Assignment 3/GlobalTemperatures.csv')

# check for duplicates in dataset - if found, use data.drop_duplicates() method
# no duplicates found, so code commented out
#duplicates = data.duplicated()
#print(f'Number of duplicate rows: {duplicates.sum()}')
# Number of duplicate rows: 0

# Rename 'dt' column to 'Date' with data.rename from pandas - logic was wrong, relocated data.rename
# to where it is now, before index is set
data.rename(columns = {'dt': 'Date'}, inplace = True)
# print to check if it works
print(data)

# apply to_datetime to convert date to a string - pd.to_datetime(data['Date']) included hours, 
# .dt.date attribute added to remove hours, leaving only yyyy-mm-dd
data['Date'] = pd.to_datetime(data['Date']).dt.date
print(data)
#

#apply indexing to column A - 'dt'. pandas library already loaded. note: changed 'dt' to 'Date' to
# reflect the change of data.rename relocation in sequence
data.set_index('Date', inplace=True)


