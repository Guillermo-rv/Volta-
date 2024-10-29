# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:14:08 2024

@author: guill
"""

""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.impute import SimpleImputer
import numpy as np

st160 = pd.read_excel("ST0160_20062024.xlsx")


st160.shape # (4982, 663)

#%% CLEAN COLUMS, AND RENAME

# Drop columns that are entirely NaN
st160 = st160.dropna(axis=1, how='all')  # Ya veremos


def clean_column_name_v3(col):
    if 'ProcessResults' in col:
        prefix = 'ProcessResults_'
        col = col.replace(
            "['data'][0]['norm']['ProcessResults']['", "").replace("']", "").split("']['")
    else:
        prefix = ''
        col = col.replace(
            "['data'][0]['norm']['", "").replace("']", "").split("']['")

    cleaned_parts = []
    seen_parts = set()
    for part in col:
        if part.isdigit():
            cleaned_parts[-1] += f"_{part}"
        else:
            if part in seen_parts:
                part = f"{part}_duplicate"
            seen_parts.add(part)
            cleaned_parts.append(part.replace("['", "_").replace("']", ""))

    clean_name = prefix + "_".join([cleaned_parts[i] for i in range(
        len(cleaned_parts)) if i == 0 or cleaned_parts[i] != cleaned_parts[i - 1]])

    return clean_name


st160.columns = [clean_column_name_v3(col) for col in st160.columns]


# %% CONVERT TO DATETIME

# DATETIME
def identify_datetime_columns(df):
    datetime_columns = []
    for col in df.columns:
        if df[col].astype(str).str.contains(r'\d{4}-\d{2}-\d{2}').any():
            datetime_columns.append(col)
    return datetime_columns

# Function to convert columns to datetime


def convert_to_datetime(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

# Function to remove timezone information


def remove_timezone(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].dt.tz_localize(None)
    return df


# Identify all potential datetime columns
identified_datetime_columns = identify_datetime_columns(st160)

# Convert these identified columns to datetime
st160 = convert_to_datetime(st160, identified_datetime_columns)

# Remove timezone information from these datetime columns
st160 = remove_timezone(st160, identified_datetime_columns)

st160.columns.tolist()

#%%
'''
Placing_[0]_PositionIndex   106117670_001
Placing_[1]_PositionIndex   106117670_002
Placing_[2]_PositionIndex   106117688_001
Placing_[3]_PositionIndex   106117688_002
Placing_[4]_PositionIndex   106117688_003
Placing_[5]_PositionIndex   106117688_004
Placing_[6]_PositionIndex   106117688_008
Placing_[7]_PositionIndex   106117688_006
Placing_[8]_PositionIndex   106117688_007
Placing_[9]_PositionIndex   106117688_005
Placing_[10]_PositionIndex  106117699_001
Placing_[11]_PositionIndex  106117699_002

Connector number 1   -->  106117670_001 
Connector number 2   -->  106117670_002 
Connector number 3   -->  106117668_001
Connector number 4   -->  106117668_004
Connector number 5   -->  106117668_005
Connector number 6   -->  106117668_008
Connector number 7   -->  106117668_002
Connector number 8   -->  106117668_003
Connector number 9   -->  106117668_006
Connector number 10 -->  106117668_007
Connector number 11 -->  106117669_001
Connector number 12 -->  106117669_002

300033529_001 corresponds with the screw number 1
300033529_002 corresponds with the screw number 2

Search_0_PositionIndex   106117670_002  
Search_1_PositionIndex   300033529_002
Search_2_PositionIndex   106117670_001
Search_3_PositionIndex   300033529_001
Search_4_PositionIndex   106117669_002
Search_5_PositionIndex   106117669_001
Search_6_PositionIndex   106117668_002
Search_7_PositionIndex   106117668_003
Search_8_PositionIndex   106117668_006
Search_9_PositionIndex   106117668_007
Search_10_PositionIndex  106117668_008
Search_11_PositionIndex  106117668_005
Search_12_PositionIndex  106117668_004
Search_13_PositionIndex  106117668_001

Positions_0_RegularConsumption   106117670_001
Positions_1_RegularConsumption   106117670_002
Positions_2_RegularConsumption   106117688_001
Positions_3_RegularConsumption   106117688_002
Positions_4_RegularConsumption   106117688_003
Positions_5_RegularConsumption   106117688_004
Positions_6_RegularConsumption   106117688_008
Positions_7_RegularConsumption   106117688_006
Positions_8_RegularConsumption   106117688_007
Positions_9_RegularConsumption   106117688_005
Positions_10_RegularConsumption  106117699_001
Positions_11_RegularConsumption  106117699_002

ProcessResults_Placing_Positions_Positions[0]_PositionIndex -->> 106117670_001
ProcessResults_MaterialConsumption_AdditionalConsumption_AdditionalConsumption[0]_PositionIndex --> empty, erase it


ProcessResults_MaterialConsumption_RegularConsumption_RegularConsumption[{i}]_PositionIndex From 0 to 11
ProcessResults_Screwing_30432_Screws_Screws[{i}]_PositionIndex' From 0 to 1
ProcessResults_PositionSearch2D_Search_Search[{i}]_PositionIndex' From 0 o 13
ProcessResults_Placing_Positions_Positions[{i}]_PositionIndex' From 0 to 11

Connector 1 and 2 are placed by robot R10 and screw 1 and 2 as well
Connectors 3 till 12 are placed by robot R10


'''


'''

PLACING COLUMNS
    
Example: ProcessResults_Placing_Positions_Positions[{i}]_Result

ProcessResults_Placing_Positions_Positions[0]_Result --> change to --> ProcessResults_Placing_Positions_Positions[1]_Result
ProcessResults_Placing_Positions_Positions[1]_Result --> change to --> ProcessResults_Placing_Positions_Positions[2]_Result 
ProcessResults_Placing_Positions_Positions[2]_Result --> change to --> ProcessResults_Placing_Positions_Positions[3]_Result
ProcessResults_Placing_Positions_Positions[3]_Result --> change to --> ProcessResults_Placing_Positions_Positions[7]_Result
ProcessResults_Placing_Positions_Positions[4]_Result --> change to --> ProcessResults_Placing_Positions_Positions[8]_Result
ProcessResults_Placing_Positions_Positions[5]_Result --> change to --> ProcessResults_Placing_Positions_Positions[4]_Result
ProcessResults_Placing_Positions_Positions[6]_Result --> change to --> ProcessResults_Placing_Positions_Positions[6]_Result
ProcessResults_Placing_Positions_Positions[7]_Result --> change to --> ProcessResults_Placing_Positions_Positions[9]_Result
ProcessResults_Placing_Positions_Positions[8]_Result --> change to --> ProcessResults_Placing_Positions_Positions[10]_Result
ProcessResults_Placing_Positions_Positions[9]_Result --> change to --> ProcessResults_Placing_Positions_Positions[5]_Result
ProcessResults_Placing_Positions_Positions[10]_Result --> change to --> ProcessResults_Placing_Positions_Positions[11]_Result
ProcessResults_Placing_Positions_Positions[11]_Result --> change to --> ProcessResults_Placing_Positions_Positions[12]_Result


all the rest of columns named like ProcessResults_Placing_Positions_Positions[{i}]  for i in range(12) should do the same

and is because this explnation:
Placing_[0]_PositionIndex   106117670_001
Placing_[1]_PositionIndex   106117670_002
Placing_[2]_PositionIndex   106117688_001
Placing_[3]_PositionIndex   106117688_002
Placing_[4]_PositionIndex   106117688_003
Placing_[5]_PositionIndex   106117688_004
Placing_[6]_PositionIndex   106117688_008
Placing_[7]_PositionIndex   106117688_006
Placing_[8]_PositionIndex   106117688_007
Placing_[9]_PositionIndex   106117688_005
Placing_[10]_PositionIndex  106117699_001
Placing_[11]_PositionIndex  106117699_002


POSITIONSEARCH
Example 2: f'ProcessResults_PositionSearch2D_Search_Search[{i}]_Position[0]_Result 


ProcessResults_PositionSearch2D_Search_Search[0]_Position[0]_Result --> change to -->  ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_Result
ProcessResults_PositionSearch2D_Search_Search[1]_Position[0]_Result --> change to -->  ProcessResults_PositionSearch2D_Search_Search_Screw[2]_Position[0]_Result
ProcessResults_PositionSearch2D_Search_Search[2]_Position[0]_Result --> change to -->  ProcessResults_PositionSearch2D_Search_Search_Placing[1]_Position[0]_Result
ProcessResults_PositionSearch2D_Search_Search[3]_Position[0]_Result --> change to -->  ProcessResults_PositionSearch2D_Search_Search_Screw[1]_Position[0]_Result
ProcessResults_PositionSearch2D_Search_Search[4]_Position[0]_Result --> change to -->  ProcessResults_PositionSearch2D_Search_Search_Placing[12]_Position[0]_Result
ProcessResults_PositionSearch2D_Search_Search[5]_Position[0]_Result --> change to -->  ProcessResults_PositionSearch2D_Search_Search_Placing[11]_Position[0]_Result
ProcessResults_PositionSearch2D_Search_Search[6]_Position[0]_Result --> change to -->  ProcessResults_PositionSearch2D_Search_Search_Placing[7]_Position[0]_Result
ProcessResults_PositionSearch2D_Search_Search[7]_Position[0]_Result --> change to -->  ProcessResults_PositionSearch2D_Search_Search_Placing[8]_Position[0]_Result
ProcessResults_PositionSearch2D_Search_Search[8]_Position[0]_Result --> change to -->  ProcessResults_PositionSearch2D_Search_Search_Placing[9]_Position[0]_Result
ProcessResults_PositionSearch2D_Search_Search[9]_Position[0]_Result --> change to -->  ProcessResults_PositionSearch2D_Search_Search_Placing[10]_Position[0]_Result
ProcessResults_PositionSearch2D_Search_Search[10]_Position[0]_Result --> change to --> ProcessResults_PositionSearch2D_Search_Search_Placing[6]_Position[0]_Result
ProcessResults_PositionSearch2D_Search_Search[11]_Position[0]_Result --> change to --> ProcessResults_PositionSearch2D_Search_Search_Placing[5]_Position[0]_Result
ProcessResults_PositionSearch2D_Search_Search[12]_Position[0]_Result --> change to --> ProcessResults_PositionSearch2D_Search_Search_Placing[4]_Position[0]_Result
ProcessResults_PositionSearch2D_Search_Search[13]_Position[0]_Result --> change to --> ProcessResults_PositionSearch2D_Search_Search_Placing[3]_Position[0]_Result

all the rest of columns named like ProcessResults_PositionSearch2D_Search_Search[{i}] should do the same for i in range(14)


Search_0_PositionIndex   106117670_002  
Search_1_PositionIndex   300033529_002
Search_2_PositionIndex   106117670_001
Search_3_PositionIndex   300033529_001
Search_4_PositionIndex   106117669_002
Search_5_PositionIndex   106117669_001
Search_6_PositionIndex   106117668_002
Search_7_PositionIndex   106117668_003
Search_8_PositionIndex   106117668_006
Search_9_PositionIndex   106117668_007
Search_10_PositionIndex  106117668_008
Search_11_PositionIndex  106117668_005
Search_12_PositionIndex  106117668_004
Search_13_PositionIndex  106117668_001

Connector number 1   -->  106117670_001 
Connector number 2   -->  106117670_002 
Connector number 3   -->  106117668_001
Connector number 4   -->  106117668_004
Connector number 5   -->  106117668_005
Connector number 6   -->  106117668_008
Connector number 7   -->  106117668_002
Connector number 8   -->  106117668_003
Connector number 9   -->  106117668_006
Connector number 10 -->   106117668_007
Connector number 11 -->   106117669_001
Connector number 12 -->   106117669_002

300033529_001 corresponds with the screw number 1
300033529_002 corresponds with the screw number 2

SCREW
Example 3: ProcessResults_Screwing_30432_Screws_Screws[{i}]_Stages[0]_Angle_AV

ProcessResults_Screwing_30432_Screws_Screws[0]_Stages[0]_Angle_AV --> change to -->  ProcessResults_Screwing_30432_Screws_Screws[1]_Stages[0]_Angle_AV 
ProcessResults_Screwing_30432_Screws_Screws[1]_Stages[0]_Angle_AV --> change to -->  ProcessResults_Screwing_30432_Screws_Screws[2]_Stages[0]_Angle_AV 

all the rest of the columns with the structure ProcessResults_Screwing_30432_Screws_Screws[{i}] should do the same for i in range(2)
as
300033529_001 corresponds with the screw number 1
300033529_002 corresponds with the screw number 2

'''




#%% TIMESTAMP COLUMNS

# Create duration columns by subtracting StartTime from EndTime for st160
for i in range(0, 13):  # Adjust the range based on your specific columns
    start_col = f'ProcessResults_ProcessTimes_ProcessTimes[{i}]_StartTime'
    end_col = f'ProcessResults_ProcessTimes_ProcessTimes[{i}]_EndTime'
    if start_col in st160.columns and end_col in st160.columns:
        # Convert to datetime
        st160[start_col] = pd.to_datetime(st160[start_col], errors='coerce')
        st160[end_col] = pd.to_datetime(st160[end_col], errors='coerce')
        
        # Calculate duration in seconds
        st160[f'ProcessResults_ProcessTimes_ProcessTimes[{i}]_Duration'] = (st160[end_col] - st160[start_col]).dt.total_seconds()

# Drop original StartTime and EndTime columns only for ProcessTimes
st160 = st160.drop(columns=[f'ProcessResults_ProcessTimes_ProcessTimes[{i}]_StartTime' for i in range(0, 13) if f'ProcessResults_ProcessTimes_ProcessTimes[{i}]_StartTime' in st160.columns] + 
                      [f'ProcessResults_ProcessTimes_ProcessTimes[{i}]_EndTime' for i in range(0, 13) if f'ProcessResults_ProcessTimes_ProcessTimes[{i}]_EndTime' in st160.columns])


#%% CLEANING

# General columns remain unchanged
general_columns = [
    'ProcessResults_GeneralData_TotalProcessingTime', 'ProcessResults_Info_CarrierID',
    'ProcessResults_Info_CarrierRoundtrips', 'ProcessResults_Info_ECartID',
    'ProcessResults_Info_TimeStamp', 'ProductID', 'Result', '_data[0]_result',
    '_station', '__ts_time',
    *[f'ProcessResults_ProcessTimes_ProcessTimes[{i}]_Duration' for i in range(13)]
]

# Placing columns
placing_columns = [
    *[f'ProcessResults_Placing_Positions_Positions[{i}]_Result' for i in range(12)],
    *[f'ProcessResults_Placing_Positions_Positions[{i}]_PositionIndex' for i in range(12)],
    *[f'ProcessResults_Placing_Positions_Positions[{i}]_TimeStamp' for i in range(12)],
    'ProcessResults_Placing_Result'
]

# PositionSearch2D columns
position_search2d_columns = [
    'ProcessResults_PositionSearch2D_Result',
    *[f'ProcessResults_PositionSearch2D_Search_Search[{i}]_PositionIndex' for i in range(14)],
    *[f'ProcessResults_PositionSearch2D_Search_Search[{i}]_Position[0]_Y' for i in range(14)],
    *[f'ProcessResults_PositionSearch2D_Search_Search[{i}]_Position[0]_X' for i in range(14)],
    *[f'ProcessResults_PositionSearch2D_Search_Search[{i}]_Position[0]_Result' for i in range(14)],
    *[f'ProcessResults_PositionSearch2D_Search_Search[{i}]_Position[2]_X' for i in range(14)],
    *[f'ProcessResults_PositionSearch2D_Search_Search[{i}]_Position[2]_Y' for i in range(14)],
    *[f'ProcessResults_PositionSearch2D_Search_Search[{i}]_Position[2]_Result' for i in range(14)]
]

# Screw columns remain unchanged
screw_columns = [
    'ProcessResults_Screwing_30432_Result',
    *[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result' for i in range(2)],
    *[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_NV' for i in range(2)],
    *[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_LT' for i in range(2)],
    *[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_UT' for i in range(2)],
    *[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV' for i in range(2)],
    *[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_NV' for i in range(2)],
    *[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_LT' for i in range(2)],
    *[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV' for i in range(2)],
    *[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Timestamp' for i in range(2)],
    *[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_UT' for i in range(2)],
    *[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_PositionIndex' for i in range(2)],
    *[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Stages[0]_Angle_AV' for i in range(2)],
    *[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Stages[0]_Torque_AV' for i in range(2)],
    *[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Stages[1]_Angle_AV' for i in range(2)],
    *[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Stages[1]_Torque_AV' for i in range(2)]
]

# Print the columns for verification
print("PositionSearch2D Columns:")
for col in position_search2d_columns:
    print(col)

print("\nPlacing Columns:")
for col in placing_columns:
    print(col)

print("\nGeneral Columns:")
for col in general_columns:
    print(col)

print("\nScrew Columns:")
for col in screw_columns:
    print(col)


# Combine all columns
columns_to_keep = general_columns + placing_columns + position_search2d_columns + screw_columns
columns_to_keep = [col for col in columns_to_keep if col in st160.columns]

# Filter the DataFrame
st160_filter = st160[columns_to_keep]

st160_filter.shape # 4982, 192

st160_filter_3 = st160_filter.copy()
st160_filter_4 = st160_filter_3.copy()

# Print the filtered columns for verification
print("Filtered Columns:")
for col in st160_filter.columns:
    print(col)




#%% RENAME SCREW

# Function to rename Screwing columns based on direct replacement
def rename_screwing_columns_directly(df):
    new_columns = {}
    for col in df.columns:
        if 'ProcessResults_Screwing_30432_Screws_Screws[0]' in col:
            new_columns[col] = col.replace('ProcessResults_Screwing_30432_Screws_Screws[0]', 'ProcessResults_Screwing_30432_Screws_Screws[1]')
        elif 'ProcessResults_Screwing_30432_Screws_Screws[1]' in col:
            new_columns[col] = col.replace('ProcessResults_Screwing_30432_Screws_Screws[1]', 'ProcessResults_Screwing_30432_Screws_Screws[2]')
    df.rename(columns=new_columns, inplace=True)
    return df

# Apply the function to rename screwing columns
rename_screwing_columns_directly(st160_filter)

# Verify the renaming of Screwing columns
screwing_columns = [col for col in st160_filter.columns if 'ProcessResults_Screwing_30432_Screws_Screws' in col]

print("\nScrewing Columns:")
for col in screwing_columns:
    print(col)

#%% RENAME PLACING

# Function to rename Placing columns based on direct replacement
def rename_placing_columns_directly(df):
    new_columns = {}
    for col in df.columns:
        if 'ProcessResults_Placing_Positions_Positions[0]' in col:
            new_columns[col] = col.replace('ProcessResults_Placing_Positions_Positions[0]', 'ProcessResults_Placing_Positions_Positions[1]')
        elif 'ProcessResults_Placing_Positions_Positions[1]' in col:
            new_columns[col] = col.replace('ProcessResults_Placing_Positions_Positions[1]', 'ProcessResults_Placing_Positions_Positions[2]')
        elif 'ProcessResults_Placing_Positions_Positions[2]' in col:
            new_columns[col] = col.replace('ProcessResults_Placing_Positions_Positions[2]', 'ProcessResults_Placing_Positions_Positions[3]')
        elif 'ProcessResults_Placing_Positions_Positions[3]' in col:
            new_columns[col] = col.replace('ProcessResults_Placing_Positions_Positions[3]', 'ProcessResults_Placing_Positions_Positions[7]')
        elif 'ProcessResults_Placing_Positions_Positions[4]' in col:
            new_columns[col] = col.replace('ProcessResults_Placing_Positions_Positions[4]', 'ProcessResults_Placing_Positions_Positions[8]')
        elif 'ProcessResults_Placing_Positions_Positions[5]' in col:
            new_columns[col] = col.replace('ProcessResults_Placing_Positions_Positions[5]', 'ProcessResults_Placing_Positions_Positions[4]')
        elif 'ProcessResults_Placing_Positions_Positions[6]' in col:
            new_columns[col] = col.replace('ProcessResults_Placing_Positions_Positions[6]', 'ProcessResults_Placing_Positions_Positions[6]')
        elif 'ProcessResults_Placing_Positions_Positions[7]' in col:
            new_columns[col] = col.replace('ProcessResults_Placing_Positions_Positions[7]', 'ProcessResults_Placing_Positions_Positions[9]')
        elif 'ProcessResults_Placing_Positions_Positions[8]' in col:
            new_columns[col] = col.replace('ProcessResults_Placing_Positions_Positions[8]', 'ProcessResults_Placing_Positions_Positions[10]')
        elif 'ProcessResults_Placing_Positions_Positions[9]' in col:
            new_columns[col] = col.replace('ProcessResults_Placing_Positions_Positions[9]', 'ProcessResults_Placing_Positions_Positions[5]')
        elif 'ProcessResults_Placing_Positions_Positions[10]' in col:
            new_columns[col] = col.replace('ProcessResults_Placing_Positions_Positions[10]', 'ProcessResults_Placing_Positions_Positions[11]')
        elif 'ProcessResults_Placing_Positions_Positions[11]' in col:
            new_columns[col] = col.replace('ProcessResults_Placing_Positions_Positions[11]', 'ProcessResults_Placing_Positions_Positions[12]')
    df.rename(columns=new_columns, inplace=True)
    return df

# Apply the function to rename placing columns
rename_placing_columns_directly(st160_filter)

# Verify the renaming of Placing columns
placing_columns = [col for col in st160_filter.columns if 'ProcessResults_Placing_Positions_Positions' in col]

print("\nPlacing Columns:")
for col in placing_columns:
    print(col)
 


#%% RENAME SEARCH

# Function to rename Search columns based on direct replacement
def rename_search_columns_directly(df):
    new_columns = {}
    for col in df.columns:
        if 'ProcessResults_PositionSearch2D_Search_Search[0]' in col:
            new_columns[col] = col.replace('ProcessResults_PositionSearch2D_Search_Search[0]', 'ProcessResults_PositionSearch2D_Search_Search_Placing[2]')
        elif 'ProcessResults_PositionSearch2D_Search_Search[1]' in col:
            new_columns[col] = col.replace('ProcessResults_PositionSearch2D_Search_Search[1]', 'ProcessResults_PositionSearch2D_Search_Search_Screw[2]')
        elif 'ProcessResults_PositionSearch2D_Search_Search[2]' in col:
            new_columns[col] = col.replace('ProcessResults_PositionSearch2D_Search_Search[2]', 'ProcessResults_PositionSearch2D_Search_Search_Placing[1]')
        elif 'ProcessResults_PositionSearch2D_Search_Search[3]' in col:
            new_columns[col] = col.replace('ProcessResults_PositionSearch2D_Search_Search[3]', 'ProcessResults_PositionSearch2D_Search_Search_Screw[1]')
        elif 'ProcessResults_PositionSearch2D_Search_Search[4]' in col:
            new_columns[col] = col.replace('ProcessResults_PositionSearch2D_Search_Search[4]', 'ProcessResults_PositionSearch2D_Search_Search_Placing[12]')
        elif 'ProcessResults_PositionSearch2D_Search_Search[5]' in col:
            new_columns[col] = col.replace('ProcessResults_PositionSearch2D_Search_Search[5]', 'ProcessResults_PositionSearch2D_Search_Search_Placing[11]')
        elif 'ProcessResults_PositionSearch2D_Search_Search[6]' in col:
            new_columns[col] = col.replace('ProcessResults_PositionSearch2D_Search_Search[6]', 'ProcessResults_PositionSearch2D_Search_Search_Placing[7]')
        elif 'ProcessResults_PositionSearch2D_Search_Search[7]' in col:
            new_columns[col] = col.replace('ProcessResults_PositionSearch2D_Search_Search[7]', 'ProcessResults_PositionSearch2D_Search_Search_Placing[8]')
        elif 'ProcessResults_PositionSearch2D_Search_Search[8]' in col:
            new_columns[col] = col.replace('ProcessResults_PositionSearch2D_Search_Search[8]', 'ProcessResults_PositionSearch2D_Search_Search_Placing[9]')
        elif 'ProcessResults_PositionSearch2D_Search_Search[9]' in col:
            new_columns[col] = col.replace('ProcessResults_PositionSearch2D_Search_Search[9]', 'ProcessResults_PositionSearch2D_Search_Search_Placing[10]')
        elif 'ProcessResults_PositionSearch2D_Search_Search[10]' in col:
            new_columns[col] = col.replace('ProcessResults_PositionSearch2D_Search_Search[10]', 'ProcessResults_PositionSearch2D_Search_Search_Placing[6]')
        elif 'ProcessResults_PositionSearch2D_Search_Search[11]' in col:
            new_columns[col] = col.replace('ProcessResults_PositionSearch2D_Search_Search[11]', 'ProcessResults_PositionSearch2D_Search_Search_Placing[5]')
        elif 'ProcessResults_PositionSearch2D_Search_Search[12]' in col:
            new_columns[col] = col.replace('ProcessResults_PositionSearch2D_Search_Search[12]', 'ProcessResults_PositionSearch2D_Search_Search_Placing[4]')
        elif 'ProcessResults_PositionSearch2D_Search_Search[13]' in col:
            new_columns[col] = col.replace('ProcessResults_PositionSearch2D_Search_Search[13]', 'ProcessResults_PositionSearch2D_Search_Search_Placing[3]')
    df.rename(columns=new_columns, inplace=True)
    return df

# Apply the function to rename search columns
rename_search_columns_directly(st160_filter)

# Verify the renaming of Search columns
search_columns = [col for col in st160_filter.columns if 'ProcessResults_PositionSearch2D_Search_Search' in col]

print("\nSearch Columns:")
for col in search_columns:
    print(col)


#%% Export to import you have to do it

st160_filter.shape # 4982, 190

print("Filtered Columns:")
for col in st160_filter.columns:
    print(col)



st160_filter.to_excel("st160_filter.xlsx", index=False)

st160_filter = pd.read_excel("st160_filter.xlsx")



#%% #%% SCREWING Y ANGLE UMBRALES *IMP AV QUE SALE SOLO ES LO MISMO QUE STAGE 2 Y ES EL QUE HAY QUE COMPARAR CON UT Y LT.


'''
ANGLE
NV = 1800 --> debería ser 40 o algo así pero... una mierda, pero Net value tampoco es un valor útil.
LT = 1
UT= 100
AV = --> entorno a 40

In angle the lower Tolerance is not right so we should find it
The UT is 100 whic is good

Torque
NV = 4,5
LT = 4,05
UT = 4,95
AV = --> entorno a 4,5

Thersholds in Screw:
    LT = 4,05
    UT = 4,95

lT = lOWER TOLERANCE
UT = UPPER TOLERANCE
NV= NET VALUE
AV = ACTUAL VALUE

'''


import matplotlib.pyplot as plt

# Define new tolerance limits for ST 160
angle_nv, angle_lt, angle_ut = 1800, 1, 100
torque_nv, torque_lt, torque_ut = 4.5, 4.05, 4.95

# Extracting angle and torque AV values along with results
angle_av_data = st160_filter[[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV' for i in range(1, 3)]]
torque_av_data = st160_filter[[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV' for i in range(1, 3)]]
results_data = st160_filter[[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result' for i in range(1, 3)]]

# Flattening the data
angle_av_flat = angle_av_data.values.flatten()
torque_av_flat = torque_av_data.values.flatten()
results_flat = results_data.values.flatten()

# Generating x-axis values for screws
screws = list(range(1, 3)) * len(st160_filter)

# Creating a mask for OK and NOK results
ok_mask = results_flat == 1
nok_mask = results_flat == 2

# Plot for Angle AV
plt.figure(figsize=(15, 7))
plt.scatter(screws, angle_av_flat, edgecolors='black', c=['cyan' if ok else 'red' for ok in ok_mask], label='OK', linewidths=0.5)
plt.scatter(screws, angle_av_flat, edgecolors='black', c=['red' if nok else 'cyan' for nok in nok_mask], label='NOK', linewidths=0.5)
plt.axhline(y=angle_lt, color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance')
plt.axhline(y=angle_ut, color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
plt.xlabel('Screw')
plt.ylabel('Angle AV')
plt.title('Distribution of Angle OK and NOK Values, ST 160')
plt.ylim(0, 200)  # Zoom in on the relevant range

# Create custom legend
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=10, markeredgecolor='black', label='OK'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, markeredgecolor='black', label='NOK'),
    plt.Line2D([0], [0], color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance'),
    plt.Line2D([0], [0], color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
]
plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), title='Legend')
plt.show()

# Plot for Torque AV
plt.figure(figsize=(15, 7))
plt.scatter(screws, torque_av_flat, edgecolors='black', c=['cyan' if ok else 'red' for ok in ok_mask], label='OK', linewidths=0.5)
plt.scatter(screws, torque_av_flat, edgecolors='black', c=['red' if nok else 'cyan' for nok in nok_mask], label='NOK', linewidths=0.5)
plt.axhline(y=torque_lt, color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance')
plt.axhline(y=torque_ut, color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
plt.xlabel('Screw')
plt.ylabel('Torque AV')
plt.title('Distribution of Torque OK and NOK Values, ST 160')
plt.ylim(3.5, 5.5)  # Zoom in on the relevant range

# Create custom legend
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=10, markeredgecolor='black', label='OK'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, markeredgecolor='black', label='NOK'),
    plt.Line2D([0], [0], color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance'),
    plt.Line2D([0], [0], color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
]
plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), title='Legend')
plt.show()


#%%

import pandas as pd
import matplotlib.pyplot as plt

# Assuming st160_filter is already loaded
# st160_filter = pd.read_excel('path_to_your_file.xlsx')

# Define the columns
angle_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV' for i in range(1, 3)]
torque_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV' for i in range(1, 3)]
result_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result' for i in range(1, 3)]

# Extract the relevant data
angle_av_data = st160_filter[angle_av_columns]
torque_av_data = st160_filter[torque_av_columns]
results_data = st160_filter[result_columns]

# Flatten the data
angle_av_flat = angle_av_data.values.flatten()
torque_av_flat = torque_av_data.values.flatten()
results_flat = results_data.values.flatten()

# Filter the OK results
ok_mask = results_flat == 1

# Calculate the Upper Tolerance (UT) and Lower Tolerance (LT) for Angle and Torque
angle_ut = angle_av_flat[ok_mask].max()
angle_lt = angle_av_flat[(ok_mask) & (angle_av_flat > 0)].min()
torque_ut = torque_av_flat[ok_mask].max()
torque_lt = torque_av_flat[(ok_mask) & (torque_av_flat > 0)].min()

# Generate x-axis values for screws
screws = list(range(1, 3)) * len(st160_filter)

# Plot for Angle AV
plt.figure(figsize=(15, 7))
plt.scatter(screws, angle_av_flat, edgecolors='black', c=['cyan' if ok else 'red' for ok in ok_mask], linewidths=0.5)
plt.axhline(y=angle_lt, color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance')
plt.axhline(y=angle_ut, color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
plt.xlabel('Screw')
plt.ylabel('Angle AV')
plt.title('Distribution of Angle OK and NOK Values, ST 160')
plt.ylim(0, 200)  # Zoom in on the relevant range

# Create custom legend
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=10, markeredgecolor='black', label='OK'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, markeredgecolor='black', label='NOK'),
    plt.Line2D([0], [0], color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance'),
    plt.Line2D([0], [0], color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
]
plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), title='Legend')
plt.show()

# Plot for Torque AV
plt.figure(figsize=(15, 7))
plt.scatter(screws, torque_av_flat, edgecolors='black', c=['cyan' if ok else 'red' for ok in ok_mask], linewidths=0.5)
plt.axhline(y=torque_lt, color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance')
plt.axhline(y=torque_ut, color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
plt.xlabel('Screw')
plt.ylabel('Torque AV')
plt.title('Distribution of Torque OK and NOK Values, ST 160')
plt.ylim(3.5, 5.5)  # Zoom in on the relevant range

# Create custom legend
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=10, markeredgecolor='black', label='OK'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, markeredgecolor='black', label='NOK'),
    plt.Line2D([0], [0], color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance'),
    plt.Line2D([0], [0], color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
]
plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), title='Legend')
plt.show()


#%%

import pandas as pd
import matplotlib.pyplot as plt

# Assuming st160_filter is already loaded
# st160_filter = pd.read_excel('path_to_your_file.xlsx')

# Define the columns
angle_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV' for i in range(1, 3)]
torque_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV' for i in range(1, 3)]
result_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result' for i in range(1, 3)]

# Extract the relevant data
angle_av_data = st160_filter[angle_av_columns]
torque_av_data = st160_filter[torque_av_columns]
results_data = st160_filter[result_columns]

# Flatten the data
angle_av_flat = angle_av_data.values.flatten()
torque_av_flat = torque_av_data.values.flatten()
results_flat = results_data.values.flatten()

# Filter the OK results
ok_mask = results_flat == 1

# Calculate the Upper Tolerance (UT) and Lower Tolerance (LT) for Angle and Torque
angle_ut = angle_av_flat[ok_mask].max()
angle_lt = angle_av_flat[(ok_mask) & (angle_av_flat > 0)].min()
torque_ut = torque_av_flat[ok_mask].max()
torque_lt = torque_av_flat[(ok_mask) & (torque_av_flat > 0)].min()

# Print the calculated values
print(f"Angle UT: {angle_ut}, Angle LT: {angle_lt}")
print(f"Torque UT: {torque_ut}, Torque LT: {torque_lt}")

# Generate x-axis values for screws
screws = list(range(1, 3)) * len(st160_filter)

# Plot for Angle AV
plt.figure(figsize=(15, 7))
plt.scatter(screws, angle_av_flat, edgecolors='black', c=['cyan' if ok else 'red' for ok in ok_mask], linewidths=0.5)
plt.axhline(y=angle_lt, color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance')
plt.axhline(y=angle_ut, color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
plt.xlabel('Screw')
plt.ylabel('Angle AV')
plt.title('Distribution of Angle OK and NOK Values, ST 160')
plt.ylim(0, 200)  # Zoom in on the relevant range

# Create custom legend
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=10, markeredgecolor='black', label='OK'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, markeredgecolor='black', label='NOK'),
    plt.Line2D([0], [0], color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance'),
    plt.Line2D([0], [0], color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
]
plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), title='Legend')
plt.show()

# Plot for Torque AV
plt.figure(figsize=(15, 7))
plt.scatter(screws, torque_av_flat, edgecolors='black', c=['cyan' if ok else 'red' for ok in ok_mask], linewidths=0.5)
plt.axhline(y=torque_lt, color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance')
plt.axhline(y=torque_ut, color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
plt.xlabel('Screw')
plt.ylabel('Torque AV')
plt.title('Distribution of Torque OK and NOK Values, ST 160')
plt.ylim(4.25, 4.75)  # Zoom in on the relevant range

# Create custom legend
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=10, markeredgecolor='black', label='OK'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, markeredgecolor='black', label='NOK'),
    plt.Line2D([0], [0], color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance'),
    plt.Line2D([0], [0], color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
]
plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), title='Legend')
plt.show()

'''
Angle UT: 99.577095, Angle LT: 14.77757
Torque UT: 4.549838, Torque LT: 4.49916363

'''


#%%

import pandas as pd
import matplotlib.pyplot as plt

# Assuming st160_filter is already loaded
# st160_filter = pd.read_excel('path_to_your_file.xlsx')

# Define the columns
angle_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV' for i in range(1, 3)]
torque_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV' for i in range(1, 3)]
result_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result' for i in range(1, 3)]

# Extract the relevant data
angle_av_data = st160_filter[angle_av_columns]
torque_av_data = st160_filter[torque_av_columns]
results_data = st160_filter[result_columns]

# Flatten the data
angle_av_flat = angle_av_data.values.flatten()
torque_av_flat = torque_av_data.values.flatten()
results_flat = results_data.values.flatten()

# Filter the OK results
ok_mask = results_flat == 1

# Calculate the Upper Tolerance (UT) and Lower Tolerance (LT) for Angle and Torque
angle_ut = angle_av_flat[ok_mask].max()
angle_lt = angle_av_flat[(ok_mask) & (angle_av_flat > 0)].min()
torque_ut = torque_av_flat[ok_mask].max()
torque_lt = torque_av_flat[(ok_mask) & (torque_av_flat > 0)].min()

# Generate x-axis values for screws
screws = list(range(1, 3)) * len(st160_filter)

# Plot for Angle AV
plt.figure(figsize=(15, 7))
plt.scatter(screws, angle_av_flat, edgecolors='black', c=['cyan' if ok else 'red' for ok in ok_mask], linewidths=0.5)
plt.axhline(y=angle_lt, color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance')
plt.axhline(y=angle_ut, color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
plt.xlabel('Screw')
plt.ylabel('Angle AV')
plt.title('Distribution of Angle OK and NOK Values, ST 160')
plt.ylim(0, 200)  # Zoom in on the relevant range

# Create custom legend
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=10, markeredgecolor='black', label='OK'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, markeredgecolor='black', label='NOK'),
    plt.Line2D([0], [0], color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance'),
    plt.Line2D([0], [0], color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
]
plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), title='Legend')
plt.show()

# Plot for Torque AV
plt.figure(figsize=(15, 7))
plt.scatter(screws, torque_av_flat, edgecolors='black', c=['cyan' if ok else 'red' for ok in ok_mask], linewidths=0.5)
plt.axhline(y=torque_lt, color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance')
plt.axhline(y=torque_ut, color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
plt.xlabel('Screw')
plt.ylabel('Torque AV')
plt.title('Distribution of Torque OK and NOK Values, ST 160')
plt.ylim(3.5, 5.5)  # Zoom in on the relevant range

# Create custom legend
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=10, markeredgecolor='black', label='OK'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, markeredgecolor='black', label='NOK'),
    plt.Line2D([0], [0], color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance'),
    plt.Line2D([0], [0], color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
]
plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), title='Legend')
plt.show()


#%%

import pandas as pd
import matplotlib.pyplot as plt



# Define the columns
angle_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV' for i in range(1, 3)]
torque_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV' for i in range(1, 3)]
result_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result' for i in range(1, 3)]

# Extract the relevant data
angle_av_data = st160_filter[angle_av_columns]
torque_av_data = st160_filter[torque_av_columns]
results_data = st160_filter[result_columns]

# Flatten the data
angle_av_flat = angle_av_data.values.flatten()
torque_av_flat = torque_av_data.values.flatten()
results_flat = results_data.values.flatten()

# Filter the OK results
ok_mask = results_flat == 1

# Calculate the Upper Tolerance (UT) and Lower Tolerance (LT) for Angle and Torque
angle_ut = angle_av_flat[ok_mask].max()
angle_lt = angle_av_flat[(ok_mask) & (angle_av_flat > 0)].min()
torque_ut = torque_av_flat[ok_mask].max()
torque_lt = torque_av_flat[(ok_mask) & (torque_av_flat > 0)].min()

# Print the calculated values
print(f"Angle UT: {angle_ut}, Angle LT: {angle_lt}")
print(f"Torque UT: {torque_ut}, Torque LT: {torque_lt}")

# Generate x-axis values for screws
screws = list(range(1, 3)) * len(st160_filter)

# Plot for Angle AV
plt.figure(figsize=(15, 7))
plt.scatter(screws, angle_av_flat, edgecolors='black', c=['cyan' if ok else 'red' for ok in ok_mask], linewidths=0.5)
plt.axhline(y=angle_lt, color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance')
plt.axhline(y=angle_ut, color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
plt.xlabel('Screw')
plt.ylabel('Angle AV')
plt.title('Distribution of Angle OK and NOK Values, ST 160')
plt.ylim(0, 200)  # Zoom in on the relevant range

# Create custom legend
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=10, markeredgecolor='black', label='OK'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, markeredgecolor='black', label='NOK'),
    plt.Line2D([0], [0], color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance'),
    plt.Line2D([0], [0], color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
]
plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), title='Legend')
plt.show()

# Plot for Torque AV
plt.figure(figsize=(15, 7))
plt.scatter(screws, torque_av_flat, edgecolors='black', c=['cyan' if ok else 'red' for ok in ok_mask], linewidths=0.5)
plt.axhline(y=torque_lt, color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance')
plt.axhline(y=torque_ut, color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
plt.xlabel('Screw')
plt.ylabel('Torque AV')
plt.title('Distribution of Torque OK and NOK Values, ST 160')
plt.ylim(4.25, 4.75)  # Zoom in on the relevant range

# Create custom legend
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=10, markeredgecolor='black', label='OK'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, markeredgecolor='black', label='NOK'),
    plt.Line2D([0], [0], color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance'),
    plt.Line2D([0], [0], color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
]
plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), title='Legend')
plt.show()

'''
Angle UT: 99.577095, Angle LT: 14.77757
Torque UT: 4.549838, Torque LT: 4.49916363

'''

#%% Correlation Angle ok Torque NOK and viceversa. THERE ARE NO VALUES.- NO CORRELATION

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming st160_filter is already loaded
# st160_filter = pd.read_excel('path_to_your_file.xlsx')

# Define the columns
angle_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV' for i in range(1, 3)]
torque_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV' for i in range(1, 3)]
result_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result' for i in range(1, 3)]
position_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_PositionIndex' for i in range(1, 3)]

# Extract the relevant data
angle_av_data = st160_filter[angle_av_columns]
torque_av_data = st160_filter[torque_av_columns]
result_data = st160_filter[result_columns]
position_data = st160_filter[position_columns]

# Flatten the data
angle_av_flat = angle_av_data.values.flatten()
torque_av_flat = torque_av_data.values.flatten()
results_flat = result_data.values.flatten()
position_flat = position_data.values.flatten()

# Filter the OK results
ok_mask = results_flat == 1

# Calculate the Upper Tolerance (UT) and Lower Tolerance (LT) for Angle and Torque
angle_ut = angle_av_flat[ok_mask].max()
angle_lt = angle_av_flat[(ok_mask) & (angle_av_flat > 0)].min()
torque_ut = torque_av_flat[ok_mask].max()
torque_lt = torque_av_flat[(ok_mask) & (torque_av_flat > 0)].min()

# Print the calculated values
print(f"Angle UT: {angle_ut}, Angle LT: {angle_lt}")
print(f"Torque UT: {torque_ut}, Torque LT: {torque_lt}")

# Generate x-axis values for screws
screws = list(range(1, 3)) * len(st160_filter)

# Plot for Angle AV
plt.figure(figsize=(15, 7))
plt.scatter(screws, angle_av_flat, edgecolors='black', c=['cyan' if ok else 'red' for ok in ok_mask], linewidths=0.5)
plt.axhline(y=angle_lt, color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance')
plt.axhline(y=angle_ut, color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
plt.xlabel('Screw')
plt.ylabel('Angle AV')
plt.title('Distribution of Angle OK and NOK Values, ST 160')
plt.ylim(0, 200)  # Zoom in on the relevant range

# Create custom legend
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=10, markeredgecolor='black', label='OK'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, markeredgecolor='black', label='NOK'),
    plt.Line2D([0], [0], color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance'),
    plt.Line2D([0], [0], color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
]
plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), title='Legend')
plt.show()

# Plot for Torque AV
plt.figure(figsize=(15, 7))
plt.scatter(screws, torque_av_flat, edgecolors='black', c=['cyan' if ok else 'red' for ok in ok_mask], linewidths=0.5)
plt.axhline(y=torque_lt, color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance')
plt.axhline(y=torque_ut, color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
plt.xlabel('Screw')
plt.ylabel('Torque AV')
plt.title('Distribution of Torque OK and NOK Values, ST 160')
plt.ylim(4.25, 4.75)  # Zoom in on the relevant range

# Create custom legend
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=10, markeredgecolor='black', label='OK'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, markeredgecolor='black', label='NOK'),
    plt.Line2D([0], [0], color='#40B6C0', linestyle='--', linewidth=2, label='Lower Tolerance'),
    plt.Line2D([0], [0], color='#D90429', linestyle='--', linewidth=2, label='Upper Tolerance')
]
plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1), title='Legend')
plt.show()

# Initialize lists to hold discrepancies data
angle_list = []
torque_list = []
position_list = []

# Loop through each screw and check for discrepancies
for i in range(1, 3):
    angle_av = angle_av_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV']
    torque_av = torque_av_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV']
    actual_result = result_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result']
    position_index = position_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_PositionIndex']
    
    # Identify rows where AV is 0 and result is OK (1)
    discrepancy_mask = ((angle_av < angle_lt) | (angle_av > angle_ut)) & (actual_result == 1)

    angle_list.extend(angle_av[discrepancy_mask].values)
    torque_list.extend(torque_av[discrepancy_mask].values)
    position_list.extend(position_index[discrepancy_mask].values)

# Create a DataFrame for discrepancies
discrepancy_df = pd.DataFrame({
    'Angle_AV': angle_list,
    'Torque_AV': torque_list,
    'PositionIndex': position_list
})

# Display the discrepancies
print("Discrepancies:")
print(discrepancy_df)

# Calculate the correlation between torque AV and angle AV for the discrepancies
correlation = discrepancy_df[['Angle_AV', 'Torque_AV']].corr()

# Display the correlation
print("Correlation between Torque AV and Angle AV for discrepancies:")
print(correlation)

# Plotting the correlation heatmap for clarity
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation between Torque AV and Angle AV for Discrepancies')
plt.show()

# Visualize the discrepancies using a scatter plot with zoomed-in view
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Torque_AV', y='Angle_AV', data=discrepancy_df, hue='PositionIndex', palette='viridis')
plt.title('Scatter Plot of Torque AV vs. Angle AV for Discrepancies')
plt.xlabel('Torque AV')
plt.ylabel('Angle AV')
plt.show()

#%% Discrepancies Between Search and Placing Results for Positions

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# st160_filter = pd.read_excel('path_to_your_file.xlsx')

# Define relevant columns
search_x_columns = [
    'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_X',
    'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[2]_X'
]
search_y_columns = [
    'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_Y',
    'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[2]_Y'
]
placing_x_columns = [
    'ProcessResults_Placing_Positions_Positions[2]_PositionIndex',
    'ProcessResults_Placing_Positions_Positions[2]_Result'
]
placing_y_columns = [
    'ProcessResults_Placing_Positions_Positions[2]_PositionIndex',
    'ProcessResults_Placing_Positions_Positions[2]_Result'
]

# Filter out noise values
st160_filter = st160_filter[(st160_filter[search_x_columns[0]] != 99) &
                            (st160_filter[search_x_columns[1]] != 99) &
                            (st160_filter[search_y_columns[0]] != 99) &
                            (st160_filter[search_y_columns[1]] != 99) &
                            (st160_filter[placing_x_columns[1]] != 99) &
                            (st160_filter[placing_y_columns[1]] != 99)]

# Extract the relevant data
search_x_data = st160_filter[search_x_columns]
search_y_data = st160_filter[search_y_columns]
placing_x_data = st160_filter[placing_x_columns]
placing_y_data = st160_filter[placing_y_columns]

# Calculate the differences between Search and Placing results
st160_filter['Diff_X_0'] = st160_filter['ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_X'] - st160_filter['ProcessResults_Placing_Positions_Positions[2]_Result']
st160_filter['Diff_X_2'] = st160_filter['ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[2]_X'] - st160_filter['ProcessResults_Placing_Positions_Positions[2]_Result']
st160_filter['Diff_Y_0'] = st160_filter['ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_Y'] - st160_filter['ProcessResults_Placing_Positions_Positions[2]_Result']
st160_filter['Diff_Y_2'] = st160_filter['ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[2]_Y'] - st160_filter['ProcessResults_Placing_Positions_Positions[2]_Result']

# Calculate basic statistics
diff_x_0_stats = st160_filter['Diff_X_0'].describe()
diff_x_2_stats = st160_filter['Diff_X_2'].describe()
diff_y_0_stats = st160_filter['Diff_Y_0'].describe()
diff_y_2_stats = st160_filter['Diff_Y_2'].describe()

print("Difference X 0 Stats:\n", diff_x_0_stats)
print("Difference X 2 Stats:\n", diff_x_2_stats)
print("Difference Y 0 Stats:\n", diff_y_0_stats)
print("Difference Y 2 Stats:\n", diff_y_2_stats)

# Visualize the distribution of differences
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.histplot(st160_filter['Diff_X_0'], kde=True)
plt.title('Distribution of Differences in X Position (0)')

plt.subplot(2, 2, 2)
sns.histplot(st160_filter['Diff_X_2'], kde=True)
plt.title('Distribution of Differences in X Position (2)')

plt.subplot(2, 2, 3)
sns.histplot(st160_filter['Diff_Y_0'], kde=True)
plt.title('Distribution of Differences in Y Position (0)')

plt.subplot(2, 2, 4)
sns.histplot(st160_filter['Diff_Y_2'], kde=True)
plt.title('Distribution of Differences in Y Position (2)')

plt.tight_layout()
plt.show()

# Correlation between differences and Placing result
correlation_matrix = st160_filter[['Diff_X_0', 'Diff_X_2', 'Diff_Y_0', 'Diff_Y_2', 'ProcessResults_Placing_Positions_Positions[2]_Result']].corr()
print("Correlation matrix:\n", correlation_matrix)

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Differences and Placing Result')
plt.show()

# Temporal analysis
st160_filter['Timestamp'] = pd.to_datetime(st160_filter['ProcessResults_Info_TimeStamp'])

plt.figure(figsize=(14, 7))
plt.plot(st160_filter['Timestamp'], st160_filter['Diff_X_0'], label='Diff_X_0', alpha=0.7)
plt.plot(st160_filter['Timestamp'], st160_filter['Diff_X_2'], label='Diff_X_2', alpha=0.7)
plt.plot(st160_filter['Timestamp'], st160_filter['Diff_Y_0'], label='Diff_Y_0', alpha=0.7)
plt.plot(st160_filter['Timestamp'], st160_filter['Diff_Y_2'], label='Diff_Y_2', alpha=0.7)
plt.legend()
plt.title('Temporal Analysis of Position Differences')
plt.xlabel('Timestamp')
plt.ylabel('Difference')
plt.xticks(rotation=45)
plt.show()

'''

Difference X 0 Stats:
 count    4968.000000
mean        0.759461
std        13.068683
min       -15.900000
25%       -13.700000
50%        11.600000
75%        12.600000
max        13.900000
Name: Diff_X_0, dtype: float64
Difference X 2 Stats:
 count    4968.000000
mean       -1.100060
std         0.588654
min        -3.100000
25%        -1.500000
50%        -1.000000
75%        -0.700000
max         2.600000
Name: Diff_X_2, dtype: float64
Difference Y 0 Stats:
 count    4968.000000
mean        9.239392
std        91.734824
min       -96.300000
25%       -93.100000
50%        90.500000
75%        91.400000
max        93.900000
Name: Diff_Y_0, dtype: float64
Difference Y 2 Stats:
 count    4968.000000
mean       -1.349436
std         1.226149
min        -4.700000
25%        -2.400000
50%        -1.500000
75%        -0.200000
max         1.400000
Name: Diff_Y_2, dtype: float64

Correlation matrix:
                                                     Diff_X_0  ...  ProcessResults_Placing_Positions_Positions[2]_Result
Diff_X_0                                            1.000000  ...                                           0.189274   
Diff_X_2                                           -0.618859  ...                                          -0.664606   
Diff_Y_0                                            0.999168  ...                                           0.212697   
Diff_Y_2                                           -0.876151  ...                                          -0.408283   
ProcessResults_Placing_Positions_Positions[2]_R...  0.189274  ...                                           1.000000   


Histograms of Differences:

The histograms for Diff_X_0 and Diff_Y_0 show a bimodal distribution, with significant clusters around -15 and +15 for Diff_X_0 and around -100 and +100 for Diff_Y_0.
The histograms for Diff_X_2 and Diff_Y_2 are more normally distributed, centered around -1 with some variance.
Correlation Matrix:

There is a high correlation between Diff_X_0 and Diff_Y_0 (0.999), suggesting these differences might be measuring a similar phenomenon or be subject to the same source of error.
There is also a moderate correlation between Diff_X_2 and Diff_Y_2 (0.72).
The placing result (ProcessResults_Placing_Positions_Positions[2]_Result) is negatively correlated with Diff_X_2 (-0.66) and Diff_Y_2 (-0.41), indicating that as the differences in X_2 and Y_2 increase, the likelihood of a negative placing result also increases.
Temporal Analysis:

The temporal analysis plot shows consistent values over time for Diff_X_0, Diff_X_2, and Diff_Y_0.
Diff_Y_2 has a slight trend indicating that the differences are changing over time, which might suggest a gradual shift or drift in the process.
Potential Improvements and Further Analysis
Analyzing Noise/Errors:

Since 99 values are identified as noise or errors, it would be useful to filter these out and re-analyze the data. This can potentially clear up the patterns and correlations.
Clustering Analysis:

Given the bimodal distribution in Diff_X_0 and Diff_Y_0, a clustering analysis (e.g., K-means clustering) could help in identifying distinct groups within the data, which might be due to different operating conditions or error states.
Time Series Decomposition:

For the temporal analysis, performing a time series decomposition could help in separating the trend, seasonality, and noise components, providing more insight into the temporal patterns.
'''

#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
# st160_filter = pd.read_excel('path_to_your_file.xlsx')

# Filter out noise/errors
filtered_data = st160_filter[
    (st160_filter['ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_Y'] != 99) &
    (st160_filter['ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_X'] != 99) &
    (st160_filter['ProcessResults_PositionSearch2D_Search_Search_Placing[0]_Position[0]_Y'] != 99) &
    (st160_filter['ProcessResults_PositionSearch2D_Search_Search_Placing[0]_Position[0]_X'] != 99)
]

# Calculate differences
filtered_data['Diff_X_0'] = filtered_data['ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_X'] - filtered_data['ProcessResults_PositionSearch2D_Search_Search_Placing[0]_Position[0]_X']
filtered_data['Diff_Y_0'] = filtered_data['ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_Y'] - filtered_data['ProcessResults_PositionSearch2D_Search_Search_Placing[0]_Position[0]_Y']
filtered_data['Diff_X_2'] = filtered_data['ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_X']
filtered_data['Diff_Y_2'] = filtered_data['ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_Y']

# Standardize data for clustering
scaler = StandardScaler()
data_for_clustering = scaler.fit_transform(filtered_data[['Diff_X_0', 'Diff_Y_0', 'Diff_X_2', 'Diff_Y_2']])

# Perform K-means clustering
kmeans = KMeans(n_clusters=2)
filtered_data['Cluster'] = kmeans.fit_predict(data_for_clustering)

# Plot clusters
sns.pairplot(filtered_data, hue='Cluster', vars=['Diff_X_0', 'Diff_Y_0', 'Diff_X_2', 'Diff_Y_2'])
plt.show()

# Re-evaluate correlations
correlation_matrix = filtered_data[['Diff_X_0', 'Diff_Y_0', 'Diff_X_2', 'Diff_Y_2', 'ProcessResults_Placing_Positions_Positions[2]_Result']].corr()

# Plot new correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('New Correlation Matrix without Noise')
plt.show()


'''
Histograms of Differences:

X Position (0): The histogram shows two main clusters of differences, around -15 and 15, with few values in between. This suggests a bimodal distribution.
X Position (2): The differences are more normally distributed around -1, with a slight skew towards the positive side.
Y Position (0): Similar to X Position (0), this histogram shows two distinct clusters, around -100 and 100, indicating a strong bimodal distribution.
Y Position (2): The differences are more spread out, with clusters around -3 and -1, indicating a more complex distribution with potential outliers.
Correlation Matrix:

The differences in X and Y positions (both 0 and 2) are strongly correlated with each other, indicating that discrepancies in one dimension are often mirrored in the other.
The correlation between differences and the placing result is moderate, suggesting that the differences do have an impact on the placing result but are not the sole determinants.
Temporal Analysis:

The temporal analysis shows that the differences remain relatively stable over time, with occasional spikes. This stability indicates that the discrepancies are not significantly influenced by temporal factors, such as time of day or specific dates.
Cluster Analysis:

The clustering analysis reveals two distinct groups (clusters 0 and 1). These clusters likely represent two different types of discrepancies or errors.
Cluster 0 appears to have larger differences, while cluster 1 has smaller differences, which could be indicative of different sources or types of errors.
New Correlation Matrix Without Noise:

After removing noise (values of 99), the correlation matrix shows strong correlations between all differences, and a moderate correlation with the placing result. This further reinforces the importance of the discrepancies in determining the placing result.

'''

#%% Clustering Analysis

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load your data into df
# df = pd.read_csv('your_data_file.csv')  # Uncomment and modify to load your data

# Assuming df is already defined
# df = pd.read_excel('st160_filter.xlsx')  # Uncomment and modify to load your data

df = st160_filter.copy()

# Columns to use
position_columns = [
    'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_Y',
    'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[2]_Y',
    'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_X',
    'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[2]_X',
    'ProcessResults_Placing_Positions_Positions[2]_Result',
    'ProcessResults_Info_TimeStamp'
]

# Ensure df contains these columns
df = df[position_columns]

# Calculate the differences
df['Diff_X_0'] = df['ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_X']
df['Diff_X_2'] = df['ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[2]_X']
df['Diff_Y_0'] = df['ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_Y']
df['Diff_Y_2'] = df['ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[2]_Y']

# Remove noise (99 values)
df = df.replace(99, pd.NA).dropna()

# Apply KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(df[['Diff_X_0', 'Diff_X_2', 'Diff_Y_0', 'Diff_Y_2']])
df['Cluster'] = kmeans.labels_

# Separate data by clusters
cluster_0 = df[df['Cluster'] == 0]
cluster_1 = df[df['Cluster'] == 1]

# Summary statistics for each cluster
summary_stats_0 = cluster_0.describe()
summary_stats_1 = cluster_1.describe()

print("Summary Statistics for Cluster 0:")
print(summary_stats_0)
print("\nSummary Statistics for Cluster 1:")
print(summary_stats_1)

# Visualize distributions
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
sns.histplot(cluster_0['Diff_X_0'], kde=True, bins=30)
plt.title('Cluster 0 - Distribution of Differences in X Position (0)')

plt.subplot(2, 2, 2)
sns.histplot(cluster_1['Diff_X_0'], kde=True, bins=30)
plt.title('Cluster 1 - Distribution of Differences in X Position (0)')

plt.subplot(2, 2, 3)
sns.histplot(cluster_0['Diff_X_2'], kde=True, bins=30)
plt.title('Cluster 0 - Distribution of Differences in X Position (2)')

plt.subplot(2, 2, 4)
sns.histplot(cluster_1['Diff_X_2'], kde=True, bins=30)
plt.title('Cluster 1 - Distribution of Differences in X Position (2)')

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
sns.histplot(cluster_0['Diff_Y_0'], kde=True, bins=30)
plt.title('Cluster 0 - Distribution of Differences in Y Position (0)')

plt.subplot(2, 2, 2)
sns.histplot(cluster_1['Diff_Y_0'], kde=True, bins=30)
plt.title('Cluster 1 - Distribution of Differences in Y Position (0)')

plt.subplot(2, 2, 3)
sns.histplot(cluster_0['Diff_Y_2'], kde=True, bins=30)
plt.title('Cluster 0 - Distribution of Differences in Y Position (2)')

plt.subplot(2, 2, 4)
sns.histplot(cluster_1['Diff_Y_2'], kde=True, bins=30)
plt.title('Cluster 1 - Distribution of Differences in Y Position (2)')

plt.tight_layout()
plt.show()

# Temporal analysis for each cluster
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(cluster_0['ProcessResults_Info_TimeStamp'], cluster_0['Diff_X_0'], label='Diff_X_0', alpha=0.7)
plt.plot(cluster_0['ProcessResults_Info_TimeStamp'], cluster_0['Diff_X_2'], label='Diff_X_2', alpha=0.7)
plt.plot(cluster_0['ProcessResults_Info_TimeStamp'], cluster_0['Diff_Y_0'], label='Diff_Y_0', alpha=0.7)
plt.plot(cluster_0['ProcessResults_Info_TimeStamp'], cluster_0['Diff_Y_2'], label='Diff_Y_2', alpha=0.7)
plt.title('Cluster 0 - Temporal Analysis of Position Differences')
plt.xlabel('Timestamp')
plt.ylabel('Difference')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(cluster_1['ProcessResults_Info_TimeStamp'], cluster_1['Diff_X_0'], label='Diff_X_0', alpha=0.7)
plt.plot(cluster_1['ProcessResults_Info_TimeStamp'], cluster_1['Diff_X_2'], label='Diff_X_2', alpha=0.7)
plt.plot(cluster_1['ProcessResults_Info_TimeStamp'], cluster_1['Diff_Y_0'], label='Diff_Y_0', alpha=0.7)
plt.plot(cluster_1['ProcessResults_Info_TimeStamp'], cluster_1['Diff_Y_2'], label='Diff_Y_2', alpha=0.7)
plt.title('Cluster 1 - Temporal Analysis of Position Differences')
plt.xlabel('Timestamp')
plt.ylabel('Difference')
plt.legend()

plt.tight_layout()
plt.show()

# Additional analysis to identify potential root causes
additional_columns = ['ProcessResults_Info_CarrierID', 'ProcessResults_Info_ECartID']

cluster_0_additional = cluster_0[additional_columns + ['Diff_X_0', 'Diff_X_2', 'Diff_Y_0', 'Diff_Y_2']]
cluster_1_additional = cluster_1[additional_columns + ['Diff_X_0', 'Diff_X_2', 'Diff_Y_0', 'Diff_Y_2']]

# Summary statistics for additional factors
print("Cluster 0 - Additional Factors")
print(cluster_0_additional.describe())

print("\nCluster 1 - Additional Factors")
print(cluster_1_additional.describe())

# Visualize the distribution of additional factors
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
sns.countplot(x='ProcessResults_Info_CarrierID', data=cluster_0_additional)
plt.title('Cluster 0 - Distribution of Carrier ID')

plt.subplot(2, 2, 2)
sns.countplot(x='ProcessResults_Info_CarrierID', data=cluster_1_additional)
plt.title('Cluster 1 - Distribution of Carrier ID')

plt.subplot(2, 2, 3)
sns.countplot(x='ProcessResults_Info_ECartID', data=cluster_0_additional)
plt.title('Cluster 0 - Distribution of ECart ID')

plt.subplot(2, 2, 4)
sns.countplot(x='ProcessResults_Info_ECartID', data=cluster_1_additional)
plt.title('Cluster 1 - Distribution of ECart ID')

plt.tight_layout()
plt.show()



'''
Cluster Distributions
Cluster 0 (Blue)

X Position (0): Most differences are around 13.5, with a relatively normal distribution.
Y Position (0): Most differences are around 92.5, with a slight left skew.
X Position (2): Most differences are around -0.25, with a slight right skew.
Y Position (2): Most differences are around -1.5, with a slight right skew.
Cluster 1 (Orange)

X Position (0): Most differences are around -13, with a sharp peak.
Y Position (0): Most differences are around -92.5, with a slight right skew.
X Position (2): Most differences are around 1, with a sharp peak.
Y Position (2): Most differences are around 1, with a slight left skew.
Temporal Analysis
Cluster 0: Differences are mostly consistent over time with slight fluctuations.
Cluster 1: Differences are consistent, with notable deviations in the Y positions indicating possible systematic issues or changes in process.
Summary Statistics
Cluster 0 shows relatively higher values in the Y Position (0) with a mean of 92.5 and in X Position (0) with a mean of 13.5.
Cluster 1 shows negative values for Y Position (0) and X Position (0), indicating a possible systematic error or difference in calibration or positioning.
Correlation Analysis
The correlation matrix after removing noise:

The high correlation (1.0) among the differences in positions suggests that the measurements are related, but there is a 0.22 correlation between these differences and the placing result, indicating a moderate relationship.
Recommendations
Investigate Systematic Differences:

The sharp peaks in Cluster 1 for X and Y positions indicate systematic differences. These might be due to calibration issues, sensor alignment, or consistent errors in the process.
Process and Calibration Checks:

Conduct a thorough review of the calibration procedures and ensure all sensors and actuators are properly aligned and functioning correctly.
Monitoring and Real-time Adjustments:

Implement real-time monitoring and adjustments to ensure any deviations are quickly corrected. This can help in maintaining consistent quality and reducing the occurrence of NOK results.
Further Investigation:

Dive deeper into the specific instances where the differences are extreme and see if there are specific conditions or triggers causing these anomalies.

'''

#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Assuming df is already defined
# df = pd.read_excel('st160_filter.xlsx')  # Uncomment and modify to load your data

df = st160_filter.copy()

# Columns to use
position_columns = [
    'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_Y',
    'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[2]_Y',
    'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_X',
    'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[2]_X',
    'ProcessResults_Placing_Positions_Positions[2]_Result',
    'ProcessResults_Info_TimeStamp',
    'ProcessResults_GeneralData_TotalProcessingTime',
    'ProcessResults_Info_CarrierID',
    'ProcessResults_Info_ECartID',
    '_station'
]

# Ensure df contains these columns
df = df[position_columns]

# Calculate the differences
df['Diff_X_0'] = df['ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_X']
df['Diff_X_2'] = df['ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[2]_X']
df['Diff_Y_0'] = df['ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_Y']
df['Diff_Y_2'] = df['ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[2]_Y']

# Remove noise (99 values)
df = df.replace(99, pd.NA).dropna()

# Apply KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(df[['Diff_X_0', 'Diff_X_2', 'Diff_Y_0', 'Diff_Y_2']])
df['Cluster'] = kmeans.labels_

# Separate data by clusters
cluster_0 = df[df['Cluster'] == 0]
cluster_1 = df[df['Cluster'] == 1]

# Summary statistics for each cluster
summary_stats_0 = cluster_0.describe()
summary_stats_1 = cluster_1.describe()

print("Summary Statistics for Cluster 0:")
print(summary_stats_0)
print("\nSummary Statistics for Cluster 1:")
print(summary_stats_1)

# Visualize distributions of differences
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
sns.histplot(cluster_0['Diff_X_0'], kde=True, bins=30)
plt.title('Cluster 0 - Distribution of Differences in X Position (0)')

plt.subplot(2, 2, 2)
sns.histplot(cluster_1['Diff_X_0'], kde=True, bins=30)
plt.title('Cluster 1 - Distribution of Differences in X Position (0)')

plt.subplot(2, 2, 3)
sns.histplot(cluster_0['Diff_X_2'], kde=True, bins=30)
plt.title('Cluster 0 - Distribution of Differences in X Position (2)')

plt.subplot(2, 2, 4)
sns.histplot(cluster_1['Diff_X_2'], kde=True, bins=30)
plt.title('Cluster 1 - Distribution of Differences in X Position (2)')

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
sns.histplot(cluster_0['Diff_Y_0'], kde=True, bins=30)
plt.title('Cluster 0 - Distribution of Differences in Y Position (0)')

plt.subplot(2, 2, 2)
sns.histplot(cluster_1['Diff_Y_0'], kde=True, bins=30)
plt.title('Cluster 1 - Distribution of Differences in Y Position (0)')

plt.subplot(2, 2, 3)
sns.histplot(cluster_0['Diff_Y_2'], kde=True, bins=30)
plt.title('Cluster 0 - Distribution of Differences in Y Position (2)')

plt.subplot(2, 2, 4)
sns.histplot(cluster_1['Diff_Y_2'], kde=True, bins=30)
plt.title('Cluster 1 - Distribution of Differences in Y Position (2)')

plt.tight_layout()
plt.show()

# Temporal analysis for each cluster
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(cluster_0['ProcessResults_Info_TimeStamp'], cluster_0['Diff_X_0'], label='Diff_X_0', alpha=0.7)
plt.plot(cluster_0['ProcessResults_Info_TimeStamp'], cluster_0['Diff_X_2'], label='Diff_X_2', alpha=0.7)
plt.plot(cluster_0['ProcessResults_Info_TimeStamp'], cluster_0['Diff_Y_0'], label='Diff_Y_0', alpha=0.7)
plt.plot(cluster_0['ProcessResults_Info_TimeStamp'], cluster_0['Diff_Y_2'], label='Diff_Y_2', alpha=0.7)
plt.title('Cluster 0 - Temporal Analysis of Position Differences')
plt.xlabel('Timestamp')
plt.ylabel('Difference')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(cluster_1['ProcessResults_Info_TimeStamp'], cluster_1['Diff_X_0'], label='Diff_X_0', alpha=0.7)
plt.plot(cluster_1['ProcessResults_Info_TimeStamp'], cluster_1['Diff_X_2'], label='Diff_X_2', alpha=0.7)
plt.plot(cluster_1['ProcessResults_Info_TimeStamp'], cluster_1['Diff_Y_0'], label='Diff_Y_0', alpha=0.7)
plt.plot(cluster_1['ProcessResults_Info_TimeStamp'], cluster_1['Diff_Y_2'], label='Diff_Y_2', alpha=0.7)
plt.title('Cluster 1 - Temporal Analysis of Position Differences')
plt.xlabel('Timestamp')
plt.ylabel('Difference')
plt.legend()

plt.tight_layout()
plt.show()

# Additional analysis to identify potential root causes
additional_columns = ['ProcessResults_GeneralData_TotalProcessingTime', 'ProcessResults_Info_CarrierID', 'ProcessResults_Info_ECartID', '_station']

cluster_0_additional = cluster_0[additional_columns + ['Diff_X_0', 'Diff_X_2', 'Diff_Y_0', 'Diff_Y_2']]
cluster_1_additional = cluster_1[additional_columns + ['Diff_X_0', 'Diff_X_2', 'Diff_Y_0', 'Diff_Y_2']]

# Summary statistics for additional factors
print("Cluster 0 - Additional Factors")
print(cluster_0_additional.describe())

print("\nCluster 1 - Additional Factors")
print(cluster_1_additional.describe())

# Visualize the distribution of additional factors
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
sns.countplot(x='ProcessResults_Info_CarrierID', data=cluster_0_additional)
plt.title('Cluster 0 - Distribution of Carrier ID')

plt.subplot(2, 2, 2)
sns.countplot(x='ProcessResults_Info_CarrierID', data=cluster_1_additional)
plt.title('Cluster 1 - Distribution of Carrier ID')

plt.subplot(2, 2, 3)
sns.countplot(x='ProcessResults_Info_ECartID', data=cluster_0_additional)
plt.title('Cluster 0 - Distribution of ECart ID')

plt.subplot(2, 2, 4)
sns.countplot(x='ProcessResults_Info_ECartID', data=cluster_1_additional)
plt.title('Cluster 1 - Distribution of ECart ID')

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.histplot(cluster_0['ProcessResults_GeneralData_TotalProcessingTime'], kde=True, bins=30)
plt.title('Cluster 0 - Distribution of Total Processing Time')

plt.subplot(1, 2, 2)
sns.histplot(cluster_1['ProcessResults_GeneralData_TotalProcessingTime'], kde=True, bins=30)
plt.title('Cluster 1 - Distribution of Total Processing Time')

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.histplot(cluster_0['_station'], kde=True, bins=30)
plt.title('Cluster 0 - Distribution of Station')

plt.subplot(1, 2, 2)
sns.histplot(cluster_1['_station'], kde=True, bins=30)
plt.title('Cluster 1 - Distribution of Station')

plt.tight_layout()
plt.show()

#%% 1. OK AND NOK BY CONNECTOR



import matplotlib.pyplot as plt
import seaborn as sns

# Define the connector position columns
connector_positions = [
    'ProcessResults_Placing_Positions_Positions[1]_Result',
    'ProcessResults_Placing_Positions_Positions[2]_Result',
    'ProcessResults_Placing_Positions_Positions[3]_Result',
    'ProcessResults_Placing_Positions_Positions[4]_Result',
    'ProcessResults_Placing_Positions_Positions[5]_Result',
    'ProcessResults_Placing_Positions_Positions[6]_Result',
    'ProcessResults_Placing_Positions_Positions[7]_Result',
    'ProcessResults_Placing_Positions_Positions[8]_Result',
    'ProcessResults_Placing_Positions_Positions[9]_Result',
    'ProcessResults_Placing_Positions_Positions[10]_Result',
    'ProcessResults_Placing_Positions_Positions[11]_Result'
]

# Map the Result values to OK and NOK labels
st160_filter['ProcessResults_Placing_Result_Label'] = st160_filter['ProcessResults_Placing_Result'].map({1: 'OK', 2: 'NOK'})

# Plotting the distribution per connector position with specified colors and labels
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 16), constrained_layout=True)
axes = axes.flatten()

for i, position in enumerate(connector_positions):
    sns.histplot(data=st160_filter, x=position, hue='ProcessResults_Placing_Result_Label', multiple='stack', bins=30, ax=axes[i], palette={'OK': 'blue', 'NOK': 'red'})
    axes[i].set_title(f'Distribution for {position}')
    axes[i].set_xlabel('Result')
    axes[i].set_ylabel('Frequency')

# Hide any extra subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.show()

#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



# Ensure _station is treated as a categorical variable
st160_filter['_station'] = st160_filter['_station'].astype('category')

# Map the Result values to OK and NOK labels
st160_filter['ProcessResults_Placing_Result_Label'] = st160_filter['Result'].map({1: 'OK', 2: 'NOK'})

# Define the conditions for assigning robots
conditions = [
    st160_filter[['ProcessResults_Placing_Positions_Positions[1]_Result',
                  'ProcessResults_Placing_Positions_Positions[2]_Result',
                  'ProcessResults_Screwing_30432_Screws_Screws[1]_Result',
                  'ProcessResults_Screwing_30432_Screws_Screws[2]_Result']].notna().any(axis=1),
    st160_filter[['ProcessResults_Placing_Positions_Positions[3]_Result',
                  'ProcessResults_Placing_Positions_Positions[7]_Result',
                  'ProcessResults_Placing_Positions_Positions[8]_Result',
                  'ProcessResults_Placing_Positions_Positions[4]_Result',
                  'ProcessResults_Placing_Positions_Positions[6]_Result',
                  'ProcessResults_Placing_Positions_Positions[9]_Result',
                  'ProcessResults_Placing_Positions_Positions[10]_Result',
                  'ProcessResults_Placing_Positions_Positions[5]_Result',
                  'ProcessResults_Placing_Positions_Positions[11]_Result',
                  'ProcessResults_Placing_Positions_Positions[12]_Result']].notna().any(axis=1)
]

# Define the corresponding robot labels
choices = ['R10', 'R20']


# Verify the data types and unique values
print(st160_filter.dtypes)
print(st160_filter['_station'].unique())

# Plotting the distribution by station
plt.figure(figsize=(14, 7))
sns.countplot(data=st160_filter, x='_station', hue='ProcessResults_Placing_Result_Label', palette={'OK': 'blue', 'NOK': 'red'})
plt.title('OK/NOK Distribution by Station')
plt.xlabel('Station')
plt.ylabel('Count')
plt.legend(title='Result')
plt.show()

# Plotting the distribution by robot
plt.figure(figsize=(14, 7))
sns.countplot(data=st160_filter, x='Robot', hue='ProcessResults_Placing_Result_Label', palette={'OK': 'blue', 'NOK': 'red'})
plt.title('OK/NOK Distribution by Robot')
plt.xlabel('Robot')
plt.ylabel('Count')
plt.legend(title='Result')
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Remove duplicate columns with the same name
st160_filter = st160_filter.loc[:, ~st160_filter.columns.duplicated()]

# Ensure _station is treated as a categorical variable
st160_filter['_station'] = st160_filter['_station'].astype('category')

# Map the Result values to OK and NOK labels
st160_filter['ProcessResults_Placing_Result_Label'] = st160_filter['Result'].map({1: 'OK', 2: 'NOK'})

# Define the conditions for R10 and R20
conditions_r10 = st160_filter[['ProcessResults_Placing_Positions_Positions[1]_Result',
                               'ProcessResults_Placing_Positions_Positions[2]_Result',
                               'ProcessResults_Screwing_30432_Screws_Screws[1]_Result',
                               'ProcessResults_Screwing_30432_Screws_Screws[2]_Result']].notna().any(axis=1)

conditions_r20 = st160_filter[['ProcessResults_Placing_Positions_Positions[3]_Result',
                               'ProcessResults_Placing_Positions_Positions[7]_Result',
                               'ProcessResults_Placing_Positions_Positions[8]_Result',
                               'ProcessResults_Placing_Positions_Positions[4]_Result',
                               'ProcessResults_Placing_Positions_Positions[6]_Result',
                               'ProcessResults_Placing_Positions_Positions[9]_Result',
                               'ProcessResults_Placing_Positions_Positions[10]_Result',
                               'ProcessResults_Placing_Positions_Positions[5]_Result',
                               'ProcessResults_Placing_Positions_Positions[11]_Result',
                               'ProcessResults_Placing_Positions_Positions[12]_Result']].notna().any(axis=1)

# Assign robot labels
st160_filter['Robot'] = np.where(conditions_r10, 'R10', np.where(conditions_r20, 'R20', 'Unknown'))

# Extract rows that should be assigned to R20
r20_results = st160_filter[conditions_r20]

# Display some sample data for R20
print(r20_results.head())

# Verify the robot assignment
print(st160_filter['Robot'].value_counts())

# Plotting the distribution by station
plt.figure(figsize=(14, 7))
sns.countplot(data=st160_filter, x='_station', hue='ProcessResults_Placing_Result_Label', palette={'OK': 'blue', 'NOK': 'red'})
plt.title('OK/NOK Distribution by Station')
plt.xlabel('Station')
plt.ylabel('Count')
plt.legend(title='Result')
plt.show()

# Plotting the distribution by robot
plt.figure(figsize=(14, 7))
sns.countplot(data=st160_filter, x='Robot', hue='ProcessResults_Placing_Result_Label', palette={'OK': 'blue', 'NOK': 'red'})
plt.title('OK/NOK Distribution by Robot')
plt.xlabel('Robot')
plt.ylabel('Count')
plt.legend(title='Result')
plt.show()

#%% X e Y segundo analisis

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

# Load the data
st160_filter = pd.read_excel("st160_filter.xlsx")

# Make a copy of the DataFrame for the analysis
st160_filter_8 = st160_filter.copy()

# List of indices to create new columns for
indices = [2, 1, 12, 11, 7, 8, 9, 10, 6, 5, 4, 3]

# Create new columns for differences
for i in indices:
    st160_filter_8[f'Placing_{i}_Position_X_Diff'] = st160_filter_8[f'ProcessResults_PositionSearch2D_Search_Search_Placing[{i}]_Position[2]_X'] - st160_filter_8[f'ProcessResults_PositionSearch2D_Search_Search_Placing[{i}]_Position[0]_X']
    st160_filter_8[f'Placing_{i}_Position_Y_Diff'] = st160_filter_8[f'ProcessResults_PositionSearch2D_Search_Search_Placing[{i}]_Position[2]_Y'] - st160_filter_8[f'ProcessResults_PositionSearch2D_Search_Search_Placing[{i}]_Position[0]_Y']

# Verify the new columns
new_columns = [col for col in st160_filter_8.columns if 'Diff' in col]
print("New columns created:", new_columns)

# STEP 2
# Calculate descriptive statistics for the difference columns
descriptive_stats = st160_filter_8[new_columns].describe()
print("Descriptive Statistics:\n", descriptive_stats)

# Separate OK and NOK results for comparison
ok_results = st160_filter_8[st160_filter_8['ProcessResults_Placing_Result'] == 1]
nok_results = st160_filter_8[st160_filter_8['ProcessResults_Placing_Result'] == 2]

# Calculate descriptive statistics for OK results
ok_descriptive_stats = ok_results[new_columns].describe()
print("Descriptive Statistics for OK Results:\n", ok_descriptive_stats)

# Calculate descriptive statistics for NOK results
nok_descriptive_stats = nok_results[new_columns].describe()
print("Descriptive Statistics for NOK Results:\n", nok_descriptive_stats)



#%% ESTRUCTURA

'''
  General columns
 'ProcessResults_GeneralData_TotalProcessingTime',
 'ProcessResults_Info_CarrierID',
 'ProcessResults_Info_CarrierRoundtrips',
 'ProcessResults_Info_ECartID',
 'ProcessResults_Info_TimeStamp',
 'ProductID',
 'Result',
 '_data[0]_result',
 '_station',
 '__ts_time',
 '_station',
 'ProcessResults_ProcessTimes_ProcessTimes[0]_Duration',
 'ProcessResults_ProcessTimes_ProcessTimes[1]_Duration',
 'ProcessResults_ProcessTimes_ProcessTimes[2]_Duration',
 'ProcessResults_ProcessTimes_ProcessTimes[3]_Duration',
 'ProcessResults_ProcessTimes_ProcessTimes[4]_Duration',
 'ProcessResults_ProcessTimes_ProcessTimes[5]_Duration',
 'ProcessResults_ProcessTimes_ProcessTimes[6]_Duration',
 'ProcessResults_ProcessTimes_ProcessTimes[7]_Duration',
 'ProcessResults_ProcessTimes_ProcessTimes[8]_Duration',
 'ProcessResults_ProcessTimes_ProcessTimes[9]_Duration',
 'ProcessResults_ProcessTimes_ProcessTimes[10]_Duration',
 'ProcessResults_ProcessTimes_ProcessTimes[11]_Duration',
 'ProcessResults_ProcessTimes_ProcessTimes[12]_Duration',
 
 
  PLACING
 'ProcessResults_Placing_Positions_Positions[1]_Result',
 'ProcessResults_Placing_Positions_Positions[2]_Result',
 'ProcessResults_Placing_Positions_Positions[3]_Result',
 'ProcessResults_Placing_Positions_Positions[7]_Result',
 'ProcessResults_Placing_Positions_Positions[8]_Result',
 'ProcessResults_Placing_Positions_Positions[4]_Result',
 'ProcessResults_Placing_Positions_Positions[6]_Result',
 'ProcessResults_Placing_Positions_Positions[9]_Result',
 'ProcessResults_Placing_Positions_Positions[10]_Result',
 'ProcessResults_Placing_Positions_Positions[5]_Result',
 'ProcessResults_Placing_Positions_Positions[11]_Result',
 'ProcessResults_Placing_Positions_Positions[12]_Result',
 'ProcessResults_Placing_Positions_Positions[1]_PositionIndex',
 'ProcessResults_Placing_Positions_Positions[2]_PositionIndex',
 'ProcessResults_Placing_Positions_Positions[3]_PositionIndex',
 'ProcessResults_Placing_Positions_Positions[7]_PositionIndex',
 'ProcessResults_Placing_Positions_Positions[8]_PositionIndex',
 'ProcessResults_Placing_Positions_Positions[4]_PositionIndex',
 'ProcessResults_Placing_Positions_Positions[6]_PositionIndex',
 'ProcessResults_Placing_Positions_Positions[9]_PositionIndex',
 'ProcessResults_Placing_Positions_Positions[10]_PositionIndex',
 'ProcessResults_Placing_Positions_Positions[5]_PositionIndex',
 'ProcessResults_Placing_Positions_Positions[11]_PositionIndex',
 'ProcessResults_Placing_Positions_Positions[12]_PositionIndex',
 'ProcessResults_Placing_Positions_Positions[1]_TimeStamp',
 'ProcessResults_Placing_Positions_Positions[2]_TimeStamp',
 'ProcessResults_Placing_Positions_Positions[3]_TimeStamp',
 'ProcessResults_Placing_Positions_Positions[7]_TimeStamp',
 'ProcessResults_Placing_Positions_Positions[8]_TimeStamp',
 'ProcessResults_Placing_Positions_Positions[4]_TimeStamp',
 'ProcessResults_Placing_Positions_Positions[6]_TimeStamp',
 'ProcessResults_Placing_Positions_Positions[9]_TimeStamp',
 'ProcessResults_Placing_Positions_Positions[10]_TimeStamp',
 'ProcessResults_Placing_Positions_Positions[5]_TimeStamp',
 'ProcessResults_Placing_Positions_Positions[11]_TimeStamp',
 'ProcessResults_Placing_Positions_Positions[12]_TimeStamp',
 'ProcessResults_Placing_Result',
 
  CAMARA 
 'ProcessResults_PositionSearch2D_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_PositionIndex',
 'ProcessResults_PositionSearch2D_Search_Search_Screw[2]_PositionIndex',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[1]_PositionIndex',
 'ProcessResults_PositionSearch2D_Search_Search_Screw[1]_PositionIndex',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_PositionIndex',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[1]_PositionIndex',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[7]_PositionIndex',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[8]_PositionIndex',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[9]_PositionIndex',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[0]_PositionIndex',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[6]_PositionIndex',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[5]_PositionIndex',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[4]_PositionIndex',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[3]_PositionIndex',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Screw[2]_Position[0]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[1]_Position[0]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Screw[1]_Position[0]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[1]_Position[0]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[7]_Position[0]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[8]_Position[0]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[9]_Position[0]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[0]_Position[0]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[6]_Position[0]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[5]_Position[0]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[4]_Position[0]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[3]_Position[0]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Screw[2]_Position[0]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[1]_Position[0]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Screw[1]_Position[0]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[1]_Position[0]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[7]_Position[0]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[8]_Position[0]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[9]_Position[0]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[0]_Position[0]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[6]_Position[0]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[5]_Position[0]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[4]_Position[0]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[3]_Position[0]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Screw[2]_Position[0]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[1]_Position[0]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Screw[1]_Position[0]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[0]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[1]_Position[0]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[7]_Position[0]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[8]_Position[0]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[9]_Position[0]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[0]_Position[0]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[6]_Position[0]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[5]_Position[0]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[4]_Position[0]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[3]_Position[0]_Result',
 'ProcessResults_PositionSearch2D_Search_Search[0]_Position[2]_X',
 'ProcessResults_PositionSearch2D_Search_Search[1]_Position[2]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[1]_Position[2]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Screw[1]_Position[2]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[2]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[1]_Position[2]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[7]_Position[2]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[8]_Position[2]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[9]_Position[2]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[0]_Position[2]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[6]_Position[2]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[5]_Position[2]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[4]_Position[2]_X',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[3]_Position[2]_X',
 'ProcessResults_PositionSearch2D_Search_Search[0]_Position[2]_Y',
 'ProcessResults_PositionSearch2D_Search_Search[1]_Position[2]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[1]_Position[2]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Screw[1]_Position[2]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[2]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[1]_Position[2]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[7]_Position[2]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[8]_Position[2]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[9]_Position[2]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[0]_Position[2]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[6]_Position[2]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[5]_Position[2]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[4]_Position[2]_Y',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[3]_Position[2]_Y',
 'ProcessResults_PositionSearch2D_Search_Search[0]_Position[2]_Result',
 'ProcessResults_PositionSearch2D_Search_Search[1]_Position[2]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[1]_Position[2]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Screw[1]_Position[2]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[2]_Position[2]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[1]_Position[2]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[7]_Position[2]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[8]_Position[2]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[9]_Position[2]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[0]_Position[2]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[6]_Position[2]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[5]_Position[2]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[4]_Position[2]_Result',
 'ProcessResults_PositionSearch2D_Search_Search_Placing[3]_Position[2]_Result',
 
 SCREW
 'ProcessResults_Screwing_30432_Result',
 'ProcessResults_Screwing_30432_Screws_Screws[1]_Result',
 'ProcessResults_Screwing_30432_Screws_Screws[2]_Result',
 'ProcessResults_Screwing_30432_Screws_Screws[1]_Angle_NV',
 'ProcessResults_Screwing_30432_Screws_Screws[2]_Angle_NV',
 'ProcessResults_Screwing_30432_Screws_Screws[1]_Angle_LT',
 'ProcessResults_Screwing_30432_Screws_Screws[2]_Angle_LT',
 'ProcessResults_Screwing_30432_Screws_Screws[1]_Angle_UT',
 'ProcessResults_Screwing_30432_Screws_Screws[2]_Angle_UT',
 'ProcessResults_Screwing_30432_Screws_Screws[1]_Angle_AV',
 'ProcessResults_Screwing_30432_Screws_Screws[2]_Angle_AV',
 'ProcessResults_Screwing_30432_Screws_Screws[1]_Torque_NV',
 'ProcessResults_Screwing_30432_Screws_Screws[2]_Torque_NV',
 'ProcessResults_Screwing_30432_Screws_Screws[1]_Torque_LT',
 'ProcessResults_Screwing_30432_Screws_Screws[2]_Torque_LT',
 'ProcessResults_Screwing_30432_Screws_Screws[1]_Torque_AV',
 'ProcessResults_Screwing_30432_Screws_Screws[2]_Torque_AV',
 'ProcessResults_Screwing_30432_Screws_Screws[1]_Timestamp',
 'ProcessResults_Screwing_30432_Screws_Screws[2]_Timestamp',
 'ProcessResults_Screwing_30432_Screws_Screws[1]_Torque_UT',
 'ProcessResults_Screwing_30432_Screws_Screws[2]_Torque_UT',
 'ProcessResults_Screwing_30432_Screws_Screws[1]_PositionIndex',
 'ProcessResults_Screwing_30432_Screws_Screws[2]_PositionIndex',
 'ProcessResults_Screwing_30432_Screws_Screws[1]_Stages[0]_Angle_AV',
 'ProcessResults_Screwing_30432_Screws_Screws[2]_Stages[0]_Angle_AV',
 'ProcessResults_Screwing_30432_Screws_Screws[1]_Stages[0]_Torque_AV',
 'ProcessResults_Screwing_30432_Screws_Screws[2]_Stages[0]_Torque_AV',
 'ProcessResults_Screwing_30432_Screws_Screws[1]_Stages[1]_Angle_AV',
 'ProcessResults_Screwing_30432_Screws_Screws[2]_Stages[1]_Angle_AV',
 'ProcessResults_Screwing_30432_Screws_Screws[1]_Stages[1]_Torque_AV',
 'ProcessResults_Screwing_30432_Screws_Screws[2]_Stages[1]_Torque_AV']


 IMPORTANT
 'ProcessResults_GeneralData_TotalProcessingTime', # TOTAL TIME
 'ProcessResults_Info_CarrierRoundtrips', # CHECK IT WITH TIME AND OK AND NOK from 'Result',
 'ProcessResults_Info_ECartID', # CHECK IT WITH TIME AND OK AND NOK from 'Result',
 '_station', # Which station performs what ST160.1 and ST160.2
 # Not here but easy to calculate R10 takes care of:
 'ProcessResults_Placing_Positions_Positions[1]_Result',
 'ProcessResults_Placing_Positions_Positions[2]_Result',
 'ProcessResults_Screwing_30432_Screws_Screws[1]_Result',
 'ProcessResults_Screwing_30432_Screws_Screws[2]_Result',
 The rest of them R20:
     'ProcessResults_Placing_Positions_Positions[3]_Result',
     'ProcessResults_Placing_Positions_Positions[7]_Result',
     'ProcessResults_Placing_Positions_Positions[8]_Result',
     'ProcessResults_Placing_Positions_Positions[4]_Result',
     'ProcessResults_Placing_Positions_Positions[6]_Result',
     'ProcessResults_Placing_Positions_Positions[9]_Result',
     'ProcessResults_Placing_Positions_Positions[10]_Result',
     'ProcessResults_Placing_Positions_Positions[5]_Result',
     'ProcessResults_Placing_Positions_Positions[11]_Result',
     'ProcessResults_Placing_Positions_Positions[12]_Result',
 
 IN PLACING [12 POSITIONS]
 'ProcessResults_Placing_Result', # TOTAL RESULTS
 
'''
