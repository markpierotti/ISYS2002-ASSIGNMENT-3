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

#import dataset. 
data = pd.read_csv('/Users/pierotti/SCU/ISYS2002_Data Wrangling and Advanced Analytics/Assignment 2/GlobalTemperatures.csv')

# check for duplicates in dataset - if found, use data.drop_duplicates() method
duplicates = data.duplicated()
print(f'Number of duplicate rows: {duplicates.sum()}')
# Number of duplicate rows: 0

#apply indexing to column A - 'dt'. pandas library already loaded
data.set_index('dt', inplace=True)

#I THINK THIS WORKS :D