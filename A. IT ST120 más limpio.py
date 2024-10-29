# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 02:07:55 2024

@author: guill
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.impute import SimpleImputer
import numpy as np


st120 = pd.read_excel("ST0120_20062024.xlsx")


st120.dtypes

#%% CLEAN COLUMNS NAMES

# Function to clean column names
def clean_column_name_v3(col):
    if 'ProcessResults' in col:
        prefix = 'ProcessResults_'
        col = col.replace("['data'][0]['norm']['ProcessResults']['", "").replace("']", "").split("']['")
    else:
        prefix = ''
        col = col.replace("['data'][0]['norm']['", "").replace("']", "").split("']['")
    
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
    
    clean_name = prefix + "_".join([cleaned_parts[i] for i in range(len(cleaned_parts)) if i == 0 or cleaned_parts[i] != cleaned_parts[i - 1]])
    
    return clean_name

# Clean column names
st120.columns = [clean_column_name_v3(col) for col in st120.columns]

#%% DROP NAN, borramos todas menos una, porque está sin valores, realmente deberiamos borrarla pero si lo hago me jode luego el codigo asiq esto es más comodo, luego ya la borramos.

st120 = st120.loc[:, (st120.columns == 'ProcessResults_ProcessTimes_ProcessTimes[5]_StartTime') | (st120.notna().any())]

st120_columns = st120.columns.tolist()

#%% CONVERT TO DATETIME

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
identified_datetime_columns = identify_datetime_columns(st120)

# Convert these identified columns to datetime
st120 = convert_to_datetime(st120, identified_datetime_columns)

# Remove timezone information from these datetime columns
st120 = remove_timezone(st120, identified_datetime_columns)

#%% CHANGE THE POSITION INDEX IN SCREW, they are named from [0] to [47] it should be from [1] to [48] otherwise the PositionIndex column does not make sense. 

import re

def increment_all_screw_indices(col_name):
    pattern = r'(ProcessResults_Screwing_30432_Screws_Screws\[(\d+)\])'
    matches = re.findall(pattern, col_name)
    
    for match in matches:
        old_index_str = match[1]
        new_index_str = str(int(old_index_str) + 1)
        old_pattern = f"[{old_index_str}]"
        new_pattern = f"[{new_index_str}]"
        col_name = col_name.replace(old_pattern, new_pattern, 1)
    
    return col_name

# Apply the renaming function to all relevant columns in st120
st120.columns = [increment_all_screw_indices(col) for col in st120.columns]



#%%
st120_sample = st120.head(600)
st120_sample.to_excel("st120.1.xlsx", index=False)


#%% ROBOTS, Which robot place each battery, in each Module?

'''
Robot 1, R10. ---> Module 1,2,3
Robot 2, R20. ---> Module 4,5,6
Robot 1, R10. ---> Module 7,8,9
Robot 2, R20. ---> Module 10,11,12

'''

#%% BATTERY

# EN SCREW, 4 tornillos son 1 bateria. ABAJO UNA FUNCIÓN PARA CREAR COLUMNAS CON LOS EL VALOR DE CADA BATERIA Y EL NOMBRE DEL TORNILLO COMO COLUMNA (USAR SI QUIERES, PERO LUEGO BORRA LAS COLUMNAS).

# 1,2,3,4 --> 1
# 5,6,7,8 --> 2
# 9,10,11,12 --> 3
# 13,14,15,16 --> 4
# 17,18,19,20 --> 5
# 21,22,23,24 --> 6
# 25,26,27,28 --> 7
# 29,30,31,32 --> 8
# 33,34,35,36 --> 9
# 37,38,39,40 --> 10
# 41,42,43,44 --> 11
# 45,46,47,48 --> 12

'''
# Function to create the new column names based on battery and screw correlation
def create_and_populate_battery_screw_columns(df):
    for battery_index in range(12):
        screw_indices = range(battery_index * 4, battery_index * 4 + 4)
        battery_number = battery_index + 1
        for i, screw_index in enumerate(screw_indices):
            new_col_name = f"ProcessResults_PlacingBatModule_{battery_number}_Screwing_{screw_index + 1}_BatteryNumber"
            df[new_col_name] = battery_number
    return df

# Apply the function to create and populate new columns
NOMBRE DEL DATAFRAME = create_and_populate_battery_screw_columns(NOMBRE DEL DATAFRAME)
'''

#%% TIME VALUES

'''
EN POSITION:

ProcessTimes_StartTime y EndTime --> Preguntar porqué vienen 6 columnas.
Asumo, no sé si erroneamente que position se hace de la siguiente manera, aunque empieza antes de screw,
al hacer el screw con unos tiempos muy cercanos cada par de módulos, asumo que hace lo mismo en position (mirar luego tiempos de screw):
   
R10 Posiciona los Módulos 3 y 2 --> 1er StartTime 
R20 Posiciona los Módulos 12 y 11 --> 2und StartTime
R10 Posiciona los Módulos 1 y 9 --> 3er StartTime
R20 Posiciona los Módulos 10 y 6 --> 4to StartTime
R10 Posiciona los Módulos 7 y 8 --> 5to StartTime * 
R20 Posiciona los Módulos 4 y 5 --> 6to StartTime


*La quinta columna de StartTime viene vacia, solo tenemos EndTime.
Dado que a no ser que hagamos TimeSeries, para los demás modelos de ML tenemos que dejar valores numéricos.
Restemos StarTime y EndTime
Hay que decir que hacer con el quinto, al no tener StartTime.

Aunque para que los robots no se choquen para mí tiene más sentido:
    
R10 Posiciona los Módulos 1 y 7 --> 1er StartTime 
R20 Posiciona los Módulos 4 y 10 --> 4to StartTime
R10 Posiciona los Módulos 2 y 8 --> 2und StartTime
R20 Posiciona los Módulos 5 y 11 --> 5to StartTime *
R10 Posiciona los Módulos 3 y 9 --> 3ero StartTime  
R20 Posiciona los Módulos 6 y 12 --> 6to StartTime

EN SCREWING:

TENEMOS TIMESTAMP.
CADA 4 Tornillos cambia el Timestamp, ya que muestra el minuto de cuando se puso cada tornillo.
O borramos el TIMESTAMP para el modelo de ML.
O restamos los tiempos, ya sea entre baterias, o en los valores de la propia columna (aunque en esta última opción, estos valores no serían representativos,
                                                                                      no es el tiempo en que tarda en poner el screw entre cada bateria ya que, 
                                                                                      la bateria proviene de la previa estación y este tiempo lo desconocemos)
1,2,3,4 --> 1 -->      R10 --> 18/03/2024  10:23:37 -- 18/03/2024  11:31:50
5,6,7,8 --> 2 -->      R10 --> 18/03/2024  10:07:07 -- 18/03/2024  11:29:10
9,10,11,12 --> 3 -->   R10 --> 18/03/2024  10:06:10 -- 18/03/2024  11:20:20
13,14,15,16 --> 4 -->  R20 --> 18/03/2024  10:36:12 -- 18/03/2024  11:34:02
17,18,19,20 --> 5 -->  R20 --> 18/03/2024  10:42:21 -- 18/03/2024  11:35:50
21,22,23,24 --> 6 -->  R20 --> 18/03/2024  10:35:16 -- 18/03/2024  11:33:08
25,26,27,28 --> 7 -->  R10 --> 18/03/2024  10:39:55 -- 18/03/2024  11:33:49
29,30,31,32 --> 8 -->  R10 --> 18/03/2024  10:40:47 -- 18/03/2024  11:35:08
33,34,35,36 --> 9 -->  R10 --> 18/03/2024  10:27:47 -- 18/03/2024  11:32:51
37,38,39,40 --> 10 --> R20 --> 18/03/2024  10:25:26 -- 18/03/2024  11:32:12
41,42,43,44 --> 11 --> R20 --> 18/03/2024  10:23:03 -- 18/03/2024  11:31:18
45,46,47,48 --> 12 --> R20 --> 18/03/2024  09:51:49 -- 18/03/2024  11:28:37



1,2,3,4 --> 1 -->      R10 
5,6,7,8 --> 2 -->      R10 
9,10,11,12 --> 3 -->   R10 
13,14,15,16 --> 4 -->  R20 
17,18,19,20 --> 5 -->  R20 
21,22,23,24 --> 6 -->  R20 
25,26,27,28 --> 7 -->  R10 
29,30,31,32 --> 8 -->  R10 
33,34,35,36 --> 9 -->  R10 
37,38,39,40 --> 10 --> R20 
41,42,43,44 --> 11 --> R20 
45,46,47,48 --> 12 --> R20 

Por lo que parece el ROBOT 10, empieza a hacer screw con los modulos 3, 2 ,1 y posteriormente va a los módulos 9 ,7 y 8.
Y el ROBOT 20 empieza por los modulos 10, 11 y 12 y posteriormente va a los módulos 6,5,7.

'''

# PARA LA PRIMERA PARTE PLACING TIMESTAMP:

# Drop the problematic columns with NaN values
columns_to_drop = ['ProcessResults_ProcessTimes_ProcessTimes[5]_StartTime', 'ProcessResults_ProcessTimes_ProcessTimes[5]_EndTime'] # Aunque tengamos EndTime al no tener StartTime, el resultado saldría negativo por lo que he decido quitarla, up to you si seguirlo.
st120 = st120.drop(columns=[col for col in columns_to_drop if col in st120.columns])

# Create duration columns, restamos el StartTime - EndTime 
for i in range(1, 7):
    start_col = f'ProcessResults_ProcessTimes_ProcessTimes[{i}]_StartTime'
    end_col = f'ProcessResults_ProcessTimes_ProcessTimes[{i}]_EndTime'
    if start_col in st120.columns and end_col in st120.columns:
        # Convert to datetime
        st120[start_col] = pd.to_datetime(st120[start_col], errors='coerce')
        st120[end_col] = pd.to_datetime(st120[end_col], errors='coerce')
        
        # Check for any conversion issues
        print(f'Conversion check for column {start_col}:\n', st120[start_col].head())
        print(f'Conversion check for column {end_col}:\n', st120[end_col].head())
        
        # Calculate duration in seconds
        st120[f'ProcessResults_ProcessTimes_ProcessTimes[{i}]_Duration'] = (st120[end_col] - st120[start_col]).dt.total_seconds()

# Drop original StartTime and EndTime columns only for ProcessTimes
st120 = st120.drop(columns=[f'ProcessResults_ProcessTimes_ProcessTimes[{i}]_StartTime' for i in range(1, 7) if f'ProcessResults_ProcessTimes_ProcessTimes[{i}]_StartTime' in st120.columns] + 
                      [f'ProcessResults_ProcessTimes_ProcessTimes[{i}]_EndTime' for i in range(1, 7) if f'ProcessResults_ProcessTimes_ProcessTimes[{i}]_EndTime' in st120.columns])

st120_columns = st120.columns.tolist()



#%% CHARGE


# Las columnas son muy similares pero tienen ligeras diferencias, así que las he mantenido por el momento.
# De acuerdo con chat GPT.

'''
Battery Charge:

ProcessResults_Screwing_30432_Screws_Screws[11]_Charge: This column might represent the electrical charge or battery level associated with the 11th screw operation. It could indicate the power consumption or the remaining battery life of the tool performing the screwing operation.
ProcessResults_MaterialConsumption_RegularConsumption_RegularConsumption[9]_Charge: Similarly, this column could represent the charge or energy consumption related to the 9th material consumption process. It might indicate how much electrical charge was used during the material consumption process.
Load/Stress Charge:

In some contexts, "charge" could refer to the load or stress applied during a process. This could be relevant in the context of manufacturing or assembly operations where specific parts are subjected to stress or load tests.
'''

#%% COUNTERHOLDING RESULTS - SUJECIÓN

# ProcessResults_PlacingBatModule_Positions_Positions[1]_Counterholding_Result INTERESENTE VER SI HAY ALGÚN PATRÓN (no parece pero estaría bien mirar la correlación entre Counterholding_Result y el resultado de cada posición).


#%% SPEED AND FORCE

# Speed en m/s, están vacios la mayoría. El único que hay es constante. ProcessResults_PlacingBatModule_Positions_Positions[10]_SpeedToTheEnd_NV. De la posición 1 hasta la 12. con Valor de 10.
# Force in Newtons


#%% SCREWING Y ANGLE UMBRALES *IMP AV QUE SALE SOLO ES LO MISMO QUE STAGE 2 Y ES EL QUE HAY QUE COMPARAR CON UT Y LT.

'''
Stage 2 tanto de Angle como de Torque es lo mismo que la columna total de Torque y Angle 
NV = Nominal Value, LT = Lower Tolerance, UT = Upper Tolerance --> esto 3 son constantes tanto en Torque como Angle.
AV = Actual Value, el valor real. 
Angle_NV = 180, Angle_LT = 175, Angle_UT = 185.
Torque_NV = 16, Torque_LT = 17,6 , Torque_UT = 40.

LT = lOWER TOLERANCE
UP = UPPER TOLERANCE
NV= NET VALUE
AV = ACTUAL VALUE
'''

#%% CLEANING

general_columns = [
    'ProcessResults_GeneralData_TotalProcessingTime', 'ProcessResults_Info_CarrierID',
    'ProcessResults_Info_CarrierRoundtrips', 'ProcessResults_Info_ECartID', 'ProductID',
    '_data[0]_result', '_station', '__ts_time', '_data[0]_ts',
]

placing_columns = [
    'ProcessResults_PlacingBatModule_Result',
] + [f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_Force_AV' for i in range(1, 13)] + \
    [f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_PositionIndex' for i in range(1, 13)] + \
    [f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_HoldingTime_NV' for i in range(1, 2)] + \
    [f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_SpeedToTheEnd_NV' for i in range(1, 2)] + \
    [f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_Counterholding_Result' for i in range(1, 13)] + \
    [f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_EndPositonModul_Result' for i in range(1, 13)] + \
    [f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_PositionPinsDuringProcess_NV' for i in range(1, 2)] + \
    [f'ProcessResults_ProcessTimes_ProcessTimes[{i}]_Duration' for i in range(1, 7)]
    
    
screwing_columns = [
    'ProcessResults_Screwing_30432_Result'
] + [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result' for i in range(1, 49)] + \
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_NV' for i in range(1)] + \
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV' for i in range(1, 49)] + \
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_LT' for i in range(1, 2)] + \
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_UT' for i in range(1 ,2)] + \
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Timestamp' for i in range(1, 49)] + \
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_NV' for i in range(1, 2)] + \
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV' for i in range(1, 49)] + \
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_LT' for i in range(1, 2)] + \
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_UT' for i in range(1, 2)] + \
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_PositionIndex' for i in range(1, 49)] + \
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Stages[0]_Angle_AV' for i in range(1, 49)] + \
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Stages[0]_Torque_AV' for i in range(1, 49)] + \
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Stages[1]_Angle_AV' for i in range(1, 49)] + \
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Stages[1]_Torque_AV' for i in range(1, 49)] 
    
material_consumption_columns = [
    'ProcessResults_MaterialConsumption_RegularConsumption_RegularConsumption[1]_DeliveryNumber'
]

'''
IMPORTANTE
    QUITAMOS:
        
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Stages[2]_Angle_AV' for i in range(1, 49)] 
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Stages[2]_Torque_AV' for i in range(1, 49)]
    ES EXACTAMENTE LO MISMO QUE 
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV' for i in range(1, 49)] + \
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV' for i in range(1, 49)] + \
DEJAMOS ESTA DOS ÚLTIMAS PORQUE SI LAS PONES UNA AL LADO DE LA OTRA SON IGUALES, PERO LAS SEGUNDAS SON MÁS COMPLETAS, PROVADLO SI QUEREIS.
 
TAMBIÉN VOY A QUITAR CHARGE, no sé si debo pero...no veo su utilidad de momento, la dejo aquí por si quereís recuperala
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Charge' for i in range(1, 49)] + \

'''
# ProcessResults_MaterialConsumption_RegularConsumption_RegularConsumption[9]_Charge

# Combine all columns to keep
columns_to_keep = general_columns + placing_columns + screwing_columns + material_consumption_columns
columns_to_keep = [col for col in columns_to_keep if col in st120.columns]
# Filter the combined_data DataFrame to keep only the specified columns
st120_filter = st120[columns_to_keep]

# st120_filter.to_excel("st120.fin1.xlsx", index=False)
st120_filter.dtypes

#%%
'''
st120_filter

La podemos utilizar para hacer los analisis previos a ML.
Columnas IMP:
    ProductID
    Result (resultados totales de todo) 2 NOK y 1 
    _station --> Te lo divide por ST120.1, ST120.2 y ST120.3 para poder comparar.
	ProcessResults_PlacingBatModule_Result --> Resultados totales de Placing
    ProcessResults_PlacingBatModule_Positions_Positions[i]_EndPositonModul_Result --> Resultados por cada módulo individual (bateria)
	ProcessResults_Screwing_30432_Result --> Resultado Total de Screwing
	ProcessResults_Screwing_30432_Screws_Screws[i]_Result --> Resultado por cada módulo
    PositionIndex, ellos analizan las cosas en base la PositionIndex, es una gilipollez pero...... es como ellos lo tienen, el esquema está en A.IT SUMMARY.


    
PLACING: 
Buscar la correlación de ProcessResults_PlacingBatModule_Result con:
En un incio debería tener relación directa con las columnas: ProcessResults_PlacingBatModule_Positions_Positions[i]_EndPositonModul_Result Pero estas tienen un 1, es decir están bien, por tanto el resultado debería ser que todo está correcto.
En cambio aparece un relación directa con  ProcessResults_Screwing_30432_Result, es decir, los resultados del Screw totales.
Sin embargo esta relación con el Screw deja de aparecer tras el día 8 de marzo o abril

A su vez cada uno de los modulos ProcessResults_PlacingBatModule_Positions_Positions[i]_EndPositonModul_Result, para las 12 posiciones del módulo.
Parece que tiene relación con la Fuerza, en cada uno de los modulos, y posiblemente con ProcessResults_ProcessTimes_ProcessTimes[{i}]_Duration', con ProcessResults_PlacingBatModule_Positions_Positions[{i}]_Counterholding_Result' y quizás 'ProcessResults_MaterialConsumption_RegularConsumption_RegularConsumption[{i}]_Charge'
Pero son suposiciones... Habría que mirarlo.

SCREW:
    
En screwing en cambio parace muy claro que si en Angle o Torque el AV no está en los umbrales UT o LT, o se aproximan mucho a ellos, el screw sale incorrecto, por tanto deberíamos mostrar un gráfico con los resultados en base a los umbrales.
Podemos hacer el analisis por tiempo, por single screw, o podemos y debereríamos hacerlo por batería.

ABAJO st120_ML para ML (no hace falta correr el codigo de en medio para llegar a st120_ML)

'''
#%%


#%% GROUPING # NO HACER SI NO ES NECESARIO FALTARÍAN LOS DATOS DE GENERAL COLUMNS

st120_columns = st120.columns.tolist()

# Divide data into groups in case you would like to do an analysis

placing_batmodule_columns = [col for col in st120_filter.columns if 'PlacingBatModule' in col]
screwing_data_columns = [col for col in st120_filter.columns if 'Screwing' in col]

placing_batmodule = st120_filter[placing_batmodule_columns]
screwing_data = st120_filter[screwing_data_columns]
screwing_data = screwing_data.select_dtypes(include=['number'])

correlation_matrix_build = placing_batmodule.select_dtypes(include='number').corr()

plt.figure(figsize=(15, 15))
plt.title('Correlation Matrix for Numerical placing_batmodule')
sns.heatmap(correlation_matrix_build, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.show()

#%% Export it si quieres
screwing_data_sample = screwing_data.head(1000)
screwing_data.to_excel("screwing_data.xlsx", index=False)

#%% ANALISIS DE SCREW

import matplotlib.pyplot as plt
import seaborn as sns


# Extract columns related to screw results
screw_result_columns = [col for col in screwing_data.columns if '_Result' in col]

# Filter for NOK results
nok_results = screwing_data[screwing_data[screw_result_columns] == 2]

# Count NOK occurrences per screw
nok_counts = nok_results[screw_result_columns].apply(pd.Series.value_counts).fillna(0).loc[2]

# Plotting the NOK occurrences
plt.figure(figsize=(12, 6))
sns.barplot(x=nok_counts.index, y=nok_counts.values)
plt.title('NOK Occurrences by Screw')
plt.xlabel('Screw')
plt.ylabel('Number of NOK Occurrences')
plt.xticks(rotation=90)
plt.show()

#%% PARA ML


st120_ML = st120_filter.copy()

# VALORES NUMERICOS:
st120_ML = st120_ML.select_dtypes(include=['number'])
st120_ML.drop(columns=['ProcessResults_Screwing_30432_Result'], inplace=True) # QUITAMOS los resultados totales de Screw, ya que depende de todas las columnas de screw previas, no así por ejemplo position que depende de más factores.
# Podríamos quitar también o la columna de Results o la columna de Results - Position. La cuestión es que aunque tienen una alta correlación, results depende también de screw, al quitar ya screw quitas gran parte de la correlación.



# ********** HAY COMO 3 ROW DE PRODUCT IT CASI VACIAS y con nan values, EL PROBLEMA ES QUE DECIDIESEMOS BORRARLOS Y LUEGO EN LAS ST120, sigue abajo
# Fueran útiles y tuvieran datos, no haríamos merge perfecto y por ende o las borramos también y perdemos los datos o las dejamos como ruido...
# Y de nuevo si en el anális previo las columnas de charge no afectan o no tienen correlación con las demás variables, podemos eliminarlas.
# Los valores de Charge hay que ver si son imporantes en el análisis previo, si no hay que borrarlo.


st120_ML.to_excel("st120_ML.xlsx", index=False)
st120_ML_columns = st120_ML.columns.tolist()


#%% AMBULANCE 

'''
3 OPCIONES, preguntar a FILIPPO. ABAJO ESTÁN SIN NAN y AGREGANDO ProductID para el merge, y exportación en excel.

1) CAMBIAMOS LOS VALORES
2) AGREGAMOS COLUMNAS
3) NO HACEMOS NADA y Utilizamos st120_ML directamente.


'''

# OPCIÓN 1 CAMBIAR EL VALOR DIRECTAMENTE EN LAS COLUMNAS

ambu_ML = st120_ML.copy()

angle_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV' for i in range(1, 49)]
torque_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV' for i in range(1, 49)]

# Replace angle values outside the 175-185 range with 180
for col in angle_columns:
    ambu_ML[col] = ambu_ML[col].apply(lambda x: 180 if pd.isna(x) or x < 175 or x > 185 else x)

# Replace torque values outside the 17.6-40 range with 30
for col in torque_columns:
    ambu_ML[col] = ambu_ML[col].apply(lambda x: 32 if pd.isna(x) or x < 17.6 or x > 40 else x)


# OPCIÓN 2 CREAR COLUMNAS NUEVAS 

ambu_ML_2 = st120_ML.copy()

# Define the columns for angles and torque
angle_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV' for i in range(1, 49)]
torque_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV' for i in range(1, 49)]

# Create new columns with the `_ambu` suffix for angles
for col in angle_columns:
    ambu_ML_2[f'{col}_ambu'] = ambu_ML_2[col].apply(lambda x: 180 if pd.isna(x) or x < 175 or x > 185 else x)

# Create new columns with the `_ambu` suffix for torque
for col in torque_columns:
    ambu_ML_2[f'{col}_ambu'] = ambu_ML_2[col].apply(lambda x: 30 if pd.isna(x) or x < 17.6 or x > 40 else x)


nan_counts = ambu_ML_2.isna().sum()


# LAS COLUMNAS POR SEPARADO

angle_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV_ambu' for i in range(1, 49)]
torque_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV_ambu' for i in range(1, 49)]

selected_columns = angle_columns + torque_columns

# Create a new dataframe with only the selected columns
ambu_ML_filtered = ambu_ML_2[selected_columns]



#%% FILL NAN OR NULL VALUES y EXPORTA PARA LUEGO

# DEPENDE DE LA DF QUE ESCOJAS, PON UNA U OTRA

# ambu_ML_2

ambu_ML_2 = ambu_ML_2.apply(lambda x: x.fillna(x.mean()) if x.dtype in ['float64', 'int64'] else x)

# Verificar si no hay nan values.
nan_counts_after = ambu_ML_2.isna().sum()
print(nan_counts_after[nan_counts_after > 0])

# AGREGAMOS PRODUCT ID para el merge posterior:
if 'ProductID' not in ambu_ML_2.columns and 'ProductID' in st120.columns:
    ambu_ML_2['ProductID'] = st120['ProductID']

# ProductID con muchos ceros:
    # st120 y st160 no concuerdan en rows, es decir hay algunos ProductID que de base se han perdido, por perder 5 más con muchos ceros espero que no pase nada, si no revertimos este paso.

# Index 14	ProductID 300055578-001P010092439000003
# Index 74	ProductID 300055578-001P010103200000123
# Index 3605	ProductID 300055578-001P010095997000033
# Index 3604	ProductID 300055578-001P010095569000065
# Index 3246 	ProductID 300055578-001P010116019000109


product_ids_to_remove = [
    "300055578-001P010092439000003",
    "300055578-001P010103200000123",
    "300055578-001P010095997000033",
    "300055578-001P010095569000065",
    "300055578-001P010116019000109"

]

ambu_ML_2 = ambu_ML_2[~ambu_ML_2['ProductID'].isin(product_ids_to_remove)]

ambu_ML_2.to_excel("ambu_ML_2.xlsx", index=False)

# ambu_ML

ambu_ML = ambu_ML.apply(lambda x: x.fillna(x.mean()) if x.dtype in ['float64', 'int64'] else x)

# Verificar si no hay nan values.
nan_counts_after = ambu_ML.isna().sum()
print(nan_counts_after[nan_counts_after > 0])

# AGREGAMOS PRODUCT ID para el merge posterior:
if 'ProductID' not in ambu_ML.columns and 'ProductID' in st120.columns:
    ambu_ML['ProductID'] = st120['ProductID']
    
# ProductID con muchos ceros:
    # st120 y st160 no concuerdan en rows, es decir hay algunos ProductID que de base se han perdido, por perder 5 más con muchos ceros espero que no pase nada, si no revertimos este paso.

# Index 14	ProductID 300055578-001P010092439000003
# Index 74	ProductID 300055578-001P010103200000123
# Index 3605	ProductID 300055578-001P010095997000033
# Index 3604	ProductID 300055578-001P010095569000065
# Index 3246 	ProductID 300055578-001P010116019000109

product_ids_to_remove = [
    "300055578-001P010092439000003",
    "300055578-001P010103200000123",
    "300055578-001P010095997000033",
    "300055578-001P010095569000065",
    "300055578-001P010116019000109"
]

ambu_ML = ambu_ML[~ambu_ML['ProductID'].isin(product_ids_to_remove)]

ambu_ML.to_excel("ambu_ML.xlsx", index=False)

# st120_ML

st120_ML = st120_ML.apply(lambda x: x.fillna(x.mean()) if x.dtype in ['float64', 'int64'] else x)

# Verificar si no hay nan values.
nan_counts_after = st120_ML.isna().sum()
print(nan_counts_after[nan_counts_after > 0])

# AGREGAMOS PRODUCT ID para el merge posterior:
if 'ProductID' not in st120_ML.columns and 'ProductID' in st120.columns:
    st120_ML['ProductID'] = st120['ProductID']
    
# ProductID con muchos ceros:
    # st120 y st160 no concuerdan en rows, es decir hay algunos ProductID que de base se han perdido, por perder 4 más con muchos ceros espero que no pase nada, si no revertimos este paso.

# Index 14	ProductID 300055578-001P010092439000003
# Index 74	ProductID 300055578-001P010103200000123
# Index 3605	ProductID 300055578-001P010095997000033
# Index 3604	ProductID 300055578-001P010095569000065
# Index 3246 	ProductID 300055578-001P010116019000109

product_ids_to_remove = [
    "300055578-001P010092439000003",
    "300055578-001P010103200000123",
    "300055578-001P010095997000033",
    "300055578-001P010095569000065",
    "300055578-001P010116019000109"
]

st120_ML = st120_ML[~st120_ML['ProductID'].isin(product_ids_to_remove)]

st120_ML.to_excel("st120_ML.xlsx", index=False)

st120_ML_columns = st120_ML.columns.to_list()




