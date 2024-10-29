# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 19:51:57 2024

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
1,2,3,4 --> 1 --> R10 --> 18/03/2024  10:23:37 -- 18/03/2024  11:31:50
5,6,7,8 --> 2 --> R10 --> 18/03/2024  10:07:07 -- 18/03/2024  11:29:10
9,10,11,12 --> 3 --> R10 --> 18/03/2024  10:06:10 -- 18/03/2024  11:20:20
13,14,15,16 --> 4 --> R20 --> 18/03/2024  10:36:12 -- 18/03/2024  11:34:02
17,18,19,20 --> 5 --> R20 --> 18/03/2024  10:42:21 -- 18/03/2024  11:35:50
21,22,23,24 --> 6 --> R20 --> 18/03/2024  10:35:16 -- 18/03/2024  11:33:08
25,26,27,28 --> 7 --> R10 --> 18/03/2024  10:39:55 -- 18/03/2024  11:33:49
29,30,31,32 --> 8 --> R10 --> 18/03/2024  10:40:47 -- 18/03/2024  11:35:08
33,34,35,36 --> 9 --> R10 --> 18/03/2024  10:27:47 -- 18/03/2024  11:32:51
37,38,39,40 --> 10 --> R20 --> 18/03/2024  10:25:26 -- 18/03/2024  11:32:12
41,42,43,44 --> 11 --> R20 --> 18/03/2024  10:23:03 -- 18/03/2024  11:31:18
45,46,47,48 --> 12 --> R20 --> 18/03/2024  9:51:49 -- 18/03/2024  11:28:37


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

#%% Otras DATETIME COLUMNS

# 1) Restar las dos columnas de tiempo total de estación, crear una nueva elimninando las previas.

st120['Total_Duration_of_St120'] = (st120['_data[0]_ts'] - st120['_data[0]_tsUtc']).dt.total_seconds()

# Obtenemos 2 valores en segundos, para la nueva columna Total_Durantion_of_St120 los procesos duran 3600 o 7200 segundos.

# Drop the original timestamp columns
st120 = st120.drop(columns=['_data[0]_ts', '_data[0]_tsUtc'])

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


# Define columns to keep based on the provided criteria
general_columns = [
    'ProcessResults_GeneralData_TotalProcessingTime', 'ProcessResults_Info_CarrierID',
    'ProcessResults_Info_CarrierRoundtrips', 'ProcessResults_Info_ECartID', 'ProductID',
    '_data[0]_result', 'Total_Duration_of_St120', '_station', '__ts_time'
]

placing_columns = [
    'ProcessResults_PlacingBatModule_Result'
] + [f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_Force_AV' for i in range(1, 13)] + \
    [f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_PositionIndex' for i in range(1, 13)] + \
    [f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_HoldingTime_NV' for i in range(1, 2)] + \
    [f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_SpeedToTheEnd_NV' for i in range(1, 2)] + \
    [f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_Counterholding_Result' for i in range(1, 13)] + \
    [f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_EndPositonModul_Result' for i in range(1, 13)] + \
    [f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_PositionPinsDuringProcess_NV' for i in range(1, 2)] + \
    [f'ProcessResults_MaterialConsumption_RegularConsumption_RegularConsumption[{i}]_Charge' for i in range(1, 13)] + \
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
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Charge' for i in range(1, 49)] + \
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Stages[0]_Angle_AV' for i in range(1, 49)] + \
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Stages[0]_Torque_AV' for i in range(1, 49)] + \
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Stages[1]_Angle_AV' for i in range(1, 49)] + \
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Stages[1]_Torque_AV' for i in range(1, 49)] 

'''
HE quitado 'Result' en general_columns
IMPORTANTE
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Stages[2]_Angle_AV' for i in range(1, 49)] 
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Stages[2]_Torque_AV' for i in range(1, 49)]
    ES EXACTAMENTE LO MISMO QUE 
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV' for i in range(1, 49)] + \
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV' for i in range(1, 49)] + \
DEJAMOS ESTA DOS ÚLTIMAS PORQUE SI LAS PONES UNA AL LADO DE LA OTRA SON IGUALES, PERO LAS SEGUNDAS SON MÁS COMPLETAS, PROVADLO SI QUEREIS.
 
TAMBIÉN VOY A QUITAR CHARGE, no sé si debo pero...no veo su utilidad de momento, la dejo aquí por si quereís recuperala


'''
# ProcessResults_MaterialConsumption_RegularConsumption_RegularConsumption[9]_Charge

# Combine all columns to keep
columns_to_keep = general_columns + placing_columns + screwing_columns
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
screwing_data_sample.to_excel("screwing_data_sample.xlsx", index=False)

st120_filter.to_excel("st120_filter.xlsx", index=False)


#%% PARA ML


st120_ML = st120_filter.copy()

# VALORES NUMERICOS:
st120_ML = st120_ML.select_dtypes(include=['number'])



#%%

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import shap
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import shap
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight


# Target column
target = 'ProcessResults_Screwing_30432_Result'

# Check the unique values to ensure proper mapping
print("Unique values in target after mapping:", st120_ML[target].unique())

# Split the data into features and target
X = st120_ML.drop(columns=[target], errors='ignore')
y = st120_ML[target]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Calculate sample weights
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Train the XGBoost model with sample weights
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train, sample_weight=sample_weights)

# Compute SHAP values
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Convert SHAP values to a DataFrame to include feature names
shap_values_df = pd.DataFrame(shap_values.values, columns=X.columns)

# Visualize SHAP values using waterfall plot for a single instance (e.g., the first instance)
shap.waterfall_plot(shap_values[0])

# Save the plot to a file
plt.savefig("/mnt/data/shap_waterfall_plot.png")
plt.show()

#%%
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import shap
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight

# Function to train XGBoost and explain with SHAP
def train_and_explain(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Calculate sample weights
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # Train the XGBoost model with sample weights
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Compute SHAP values
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    return shap_values, X.columns

# Function to create additional SHAP plots
def create_shap_plots(shap_values, dataset_name, feature_names):
    # Ensure the SHAP values contain the correct feature names
    shap_values.feature_names = feature_names
    
    # Waterfall plot for the first prediction
    shap.waterfall_plot(shap_values[0])
    plt.savefig(f'{dataset_name}_waterfall_plot.png')
    plt.clf()

    # Force plot for the first prediction
    shap.force_plot(shap_values[0])
    plt.savefig(f'{dataset_name}_force_plot.png')
    plt.clf()

    # Beeswarm plot to summarize the effects of all features
    shap.plots.beeswarm(shap_values)
    plt.savefig(f'{dataset_name}_beeswarm_plot.png')
    plt.clf()

    # Bar plot of the mean absolute SHAP values for each feature
    shap.plots.bar(shap_values)
    plt.savefig(f'{dataset_name}_bar_plot.png')
    plt.clf()

# Train and explain for st120_ML
shap_values_st120, feature_names_st120 = train_and_explain(st120_ML, 'ProcessResults_Screwing_30432_Result')

# Create SHAP plots for st120_ML
create_shap_plots(shap_values_st120, "st120_ML", feature_names_st120)

#%%

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import shap
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight

# Function to train XGBoost and explain with SHAP
def train_and_explain(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Calculate sample weights
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # Train the XGBoost model with sample weights
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Compute SHAP values
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    return shap_values, X.columns

# Function to create additional SHAP plots
def create_shap_plots(shap_values, dataset_name, feature_names):
    # Ensure the SHAP values contain the correct feature names
    shap_values.feature_names = feature_names

    # Set the max display value
    max_display = 30
    
    # Waterfall plot for the first prediction (displays top features for one instance)
    shap.waterfall_plot(shap_values[0], max_display=max_display)
    plt.savefig(f'{dataset_name}_waterfall_plot.png')
    plt.clf()

    # Force plot for the first prediction
    shap.force_plot(shap_values[0])
    plt.savefig(f'{dataset_name}_force_plot.png')
    plt.clf()

    # Beeswarm plot to summarize the effects of all features
    shap.plots.beeswarm(shap_values, max_display=max_display)  # Display more features
    plt.savefig(f'{dataset_name}_beeswarm_plot.png')
    plt.clf()

    # Bar plot of the mean absolute SHAP values for each feature
    shap.plots.bar(shap_values, max_display=max_display)  # Display more features
    plt.savefig(f'{dataset_name}_bar_plot.png')
    plt.clf()

# Train and explain for st120_ML
shap_values_st120, feature_names_st120 = train_and_explain(st120_ML, 'ProcessResults_Screwing_30432_Result')

# Create SHAP plots for st120_ML
create_shap_plots(shap_values_st120, "st120_ML", feature_names_st120)

#%% SCREW ANGLE AND TORQUE

# Define tolerance limits
ANGLE_LT = 175
ANGLE_UT = 185
TORQUE_LT = 17.6
TORQUE_UT = 40

# Extract relevant columns
result_col = 'ProcessResults_Screwing_30432_Result'
angle_av_cols = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV' for i in range(1, 49)]
torque_av_cols = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV' for i in range(1, 49)]
screw_result_cols = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result' for i in range(1, 49)]

# Check if AV values are within tolerance limits
screwing_data['Angle_AV_OK'] = screwing_data[angle_av_cols].apply(lambda row: row.between(ANGLE_LT, ANGLE_UT).all(), axis=1)
screwing_data['Torque_AV_OK'] = screwing_data[torque_av_cols].apply(lambda row: row.between(TORQUE_LT, TORQUE_UT).all(), axis=1)

# Combine the results
screwing_data['All_AV_OK'] = screwing_data['Angle_AV_OK'] & screwing_data['Torque_AV_OK']

# Check the relationship between the AV_OK and the NOK results
nok_due_to_av = screwing_data[screwing_data[result_col] == 2].groupby('All_AV_OK').size()

nok_due_to_av

'''
All_AV_OK
False    1688
True       57
dtype: int64
'''
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style
sns.set(style="whitegrid")

# Create a figure and a set of subplots
fig, ax = plt.subplots(figsize=(10, 6))

# Bar plot showing the relationship between AV_OK and NOK results
nok_counts = nok_due_to_av.reset_index()
nok_counts.columns = ['All_AV_OK', 'Count']
nok_counts['All_AV_OK'] = nok_counts['All_AV_OK'].map({True: 'Within Tolerance', False: 'Outside Tolerance'})

sns.barplot(data=nok_counts, x='All_AV_OK', y='Count', palette='viridis', ax=ax)

# Add labels and title
ax.set_title('NOK Results Based on Tolerance Levels of Angle and Torque AV', fontsize=16)
ax.set_xlabel('Tolerance Status', fontsize=14)
ax.set_ylabel('Number of NOK Results', fontsize=14)

# Show the count values on top of the bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define tolerance limits
ANGLE_LT = 175
ANGLE_UT = 185
TORQUE_LT = 17.6
TORQUE_UT = 40

# Extract relevant columns
result_col = 'ProcessResults_Screwing_30432_Result'
angle_av_cols = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV' for i in range(1, 49)]
torque_av_cols = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV' for i in range(1, 49)]

# Separate the data into NOK results for angles and torque
angle_nok = screwing_data[screwing_data[result_col] == 2].copy()
torque_nok = screwing_data[screwing_data[result_col] == 2].copy()

# Prepare data for visualization
angle_nok_melted = angle_nok.melt(value_vars=angle_av_cols, var_name='Screw', value_name='Angle_AV')
torque_nok_melted = torque_nok.melt(value_vars=torque_av_cols, var_name='Screw', value_name='Torque_AV')

# Add a column to indicate whether the value is within tolerance
angle_nok_melted['Status'] = angle_nok_melted['Angle_AV'].between(ANGLE_LT, ANGLE_UT).map({True: 'OK', False: 'NOK'})
torque_nok_melted['Status'] = torque_nok_melted['Torque_AV'].between(TORQUE_LT, TORQUE_UT).map({True: 'OK', False: 'NOK'})

# Define custom colors
colors = {"OK": "#00F1FF", "NOK": "#FF0400"}

# Create plot for Angle AV
fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(data=angle_nok_melted, x='Screw', y='Angle_AV', hue='Status', palette=colors, s=40, edgecolor='black', ax=ax)
ax.axhline(ANGLE_LT, color='#00F1FF', linestyle='--', linewidth=2, label='Lower Tolerance')
ax.axhline(ANGLE_UT, color='#FF0400', linestyle='--', linewidth=2, label='Upper Tolerance')
ax.set_title('Distribution of Angle OK and NOK Values, ST 120', fontsize=16)
ax.set_ylabel('Angle AV', fontsize=14)
ax.set_xlabel('Screw', fontsize=14)
ax.set_ylim(160, 190)  # Zoom in on the relevant range
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Create plot for Torque AV
fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(data=torque_nok_melted, x='Screw', y='Torque_AV', hue='Status', palette=colors, s=40, edgecolor='black', ax=ax)
ax.axhline(TORQUE_LT, color='#00F1FF', linestyle='--', linewidth=2, label='Lower Tolerance')
ax.axhline(TORQUE_UT, color='#FF0400', linestyle='--', linewidth=2, label='Upper Tolerance')
ax.set_title('Distribution of Torque OK and NOK Values, ST 120', fontsize=16)
ax.set_ylabel('Torque AV', fontsize=14)
ax.set_xlabel('Screw', fontsize=14)
ax.set_ylim(0, 45)  # Zoom in on the relevant range
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#%% POSITION no hacer

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the placing data
file_path = '/mnt/data/screwing_data_st120.xlsx'  # Adjust the path to your file
st120_filter = pd.read_excel(file_path)

# Define arbitrary tolerance limits for force (since actual thresholds are not provided)
FORCE_LT = 1400
FORCE_UT = 1600

# Define placing_batmodule_columns
placing_batmodule_columns = [
    ] + [f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_Force_AV' for i in range(1, 13)] + \
    [f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_PositionIndex' for i in range(1, 13)] + \
    [f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_EndPositonModul_Result' for i in range(1, 13)]

# Filter the relevant columns
placing_batmodule = st120_filter[placing_batmodule_columns]

# Prepare the data for visualization
data_list = []
for i in range(1, 13):
    force_col = f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_Force_AV'
    position_col = f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_PositionIndex'
    result_col = f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_EndPositonModul_Result'
    temp_df = placing_batmodule[[force_col, position_col, result_col]].copy()
    temp_df.columns = ['Force_AV', 'PositionIndex', 'EndPositonModul_Result']
    data_list.append(temp_df)

combined_data = pd.concat(data_list, ignore_index=True)

# Add a column to indicate the actual status based on EndPositonModul_Result (1 means OK, 0 means NOK)
combined_data['Status'] = combined_data['EndPositonModul_Result'].map({1: 'OK', 0: 'NOK'})

# Define custom colors
colors = {"OK": "#00F1FF", "NOK": "#FF0400"}

# Create plot for Force AV
fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(data=combined_data, x='PositionIndex', y='Force_AV', hue='Status', palette=colors, s=40, edgecolor='black', ax=ax)
ax.axhline(FORCE_LT, color='#00F1FF', linestyle='--', linewidth=2, label='Lower Tolerance')
ax.axhline(FORCE_UT, color='#FF0400', linestyle='--', linewidth=2, label='Upper Tolerance')
ax.set_title('Distribution of Force OK and NOK Values', fontsize=16)
ax.set_ylabel('Force AV', fontsize=14)
ax.set_xlabel('Position Index', fontsize=14)
ax.set_ylim(1200, 1700)  # Adjust the y-axis range as needed
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#%%


# Extract relevant columns for all 48 screws
angle_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV' for i in range(1, 49)]
torque_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV' for i in range(1, 49)]
result_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result' for i in range(1, 49)]

# Extract the relevant data
angle_av_data = screwing_data[angle_av_columns]
torque_av_data = screwing_data[torque_av_columns]
result_data = screwing_data[result_columns]

# Define tolerance limits
angle_nv, angle_lt, angle_ut = 180, 175, 185
torque_nv, torque_lt, torque_ut = 16, 17.6, 40

# Function to check if AV values are within the tolerance limits
def check_within_tolerance(angle_av, torque_av):
    angle_ok = angle_lt <= angle_av <= angle_ut
    torque_ok = torque_lt <= torque_av <= torque_ut
    return angle_ok and torque_ok

# Apply the function and compare with the results
results_check = []
for i in range(1, 49):
    angle_av = angle_av_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV']
    torque_av = torque_av_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV']
    actual_result = result_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result']
    
    # Check if within tolerance for all rows
    within_tolerance = angle_av.apply(lambda x: angle_lt <= x <= angle_ut) & torque_av.apply(lambda x: torque_lt <= x <= torque_ut)
    predicted_result = within_tolerance.apply(lambda x: 1 if x else 2)  # 1 for OK, 2 for NOK
    
    # Compare actual result with predicted result
    comparison = actual_result == predicted_result
    results_check.append(comparison)

# Combine results into a single DataFrame
results_check_df = pd.concat(results_check, axis=1)
results_check_df.columns = [f'Screw_{i}_Result_Check' for i in range(1, 49)]

# Display the results
print(results_check_df.head())



#%% REAL PLOT 

import matplotlib.pyplot as plt

# Extracting angle and torque AV values along with results
angle_av_data = screwing_data[[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV' for i in range(1, 49)]]
torque_av_data = screwing_data[[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV' for i in range(1, 49)]]
results_data = screwing_data[[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result' for i in range(1, 49)]]

# Flattening the data
angle_av_flat = angle_av_data.values.flatten()
torque_av_flat = torque_av_data.values.flatten()
results_flat = results_data.values.flatten()

# Generating x-axis values for screws
screws = list(range(1, 49)) * len(screwing_data)

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
plt.title('Distribution of Angle OK and NOK Values, ST 120')
plt.ylim(160, 190)  # Zoom in on the relevant range

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
plt.title('Distribution of Torque OK and NOK Values, ST 120')

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
import seaborn as sns
import matplotlib.pyplot as plt

# Extract relevant columns for all 48 screws
angle_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV' for i in range(1, 49)]
torque_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV' for i in range(1, 49)]
result_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result' for i in range(1, 49)]
position_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_PositionIndex' for i in range(1, 49)]

# Extract the relevant data
angle_av_data = screwing_data[angle_av_columns]
torque_av_data = screwing_data[torque_av_columns]
result_data = screwing_data[result_columns]
position_data = screwing_data[position_columns]

# Initialize lists to hold discrepancies data
angle_list = []
torque_list = []
position_list = []

# Loop through each screw and check for zero values
for i in range(1, 49):
    angle_av = angle_av_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV']
    torque_av = torque_av_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV']
    actual_result = result_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result']
    position_index = position_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_PositionIndex']
    
    # Identify rows where AV is 0 and result is OK (1)
    zero_mask = ((angle_av == 0) | (torque_av == 0)) & (actual_result == 1)
    
    angle_list.extend(angle_av[zero_mask].values)
    torque_list.extend(torque_av[zero_mask].values)
    position_list.extend(position_index[zero_mask].values)

# Create a DataFrame for zero value discrepancies
zero_value_df = pd.DataFrame({
    'Angle_AV': angle_list,
    'Torque_AV': torque_list,
    'PositionIndex': position_list
})

# Display the zero value discrepancies
print("Zero value discrepancies:")
print(zero_value_df)

# Calculate the correlation between torque AV and angle AV for the zero value discrepancies
correlation = zero_value_df[['Angle_AV', 'Torque_AV']].corr()

# Display the correlation
print("Correlation between Torque AV and Angle AV for zero value discrepancies:")
print(correlation)

# Plotting the correlation heatmap for clarity
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation between Torque AV and Angle AV for Zero Value Discrepancies')
plt.show()

# Visualize the zero value discrepancies using a scatter plot with zoomed-in view
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Torque_AV', y='Angle_AV', data=zero_value_df, hue='PositionIndex', palette='viridis')
plt.xlim(-0.1, 0.1)  # Zoom in on the relevant range for Torque AV
plt.ylim(-10, 200)   # Zoom in on the relevant range for Angle AV
plt.title('Scatter Plot of Torque AV vs. Angle AV for Zero Value Discrepancies')
plt.xlabel('Torque AV')
plt.ylabel('Angle AV')
plt.legend(title='Position Index')
plt.show()




#%% TIMESTAMP ANALYIS FOR SCREW, ERRORS WITH 0 -NO MUY UTIL NO CORRER

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Extract relevant columns for all 48 screws
angle_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV' for i in range(1, 49)]
torque_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV' for i in range(1, 49)]
result_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result' for i in range(1, 49)]
position_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_PositionIndex' for i in range(1, 49)]
timestamp_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Timestamp' for i in range(1, 49)]

# Extract the relevant data
angle_av_data = screwing_data[angle_av_columns]
torque_av_data = screwing_data[torque_av_columns]
result_data = screwing_data[result_columns]
position_data = screwing_data[position_columns]
timestamp_data = screwing_data[timestamp_columns]

# Initialize lists to hold discrepancies data
angle_list = []
torque_list = []
position_list = []
timestamp_list = []

# Loop through each screw and check for zero values
for i in range(1, 49):
    angle_av = angle_av_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV']
    torque_av = torque_av_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV']
    actual_result = result_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result']
    position_index = position_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_PositionIndex']
    timestamp = timestamp_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Timestamp']
    
    # Identify rows where AV is 0 and result is OK (1)
    zero_mask = ((angle_av == 0) | (torque_av == 0)) & (actual_result == 1)
    
    angle_list.extend(angle_av[zero_mask].values)
    torque_list.extend(torque_av[zero_mask].values)
    position_list.extend(position_index[zero_mask].values)
    timestamp_list.extend(timestamp[zero_mask].values)

# Create a DataFrame for zero value discrepancies
zero_value_df = pd.DataFrame({
    'Angle_AV': angle_list,
    'Torque_AV': torque_list,
    'PositionIndex': position_list,
    'Timestamp': timestamp_list
})

# Display the zero value discrepancies
print("Zero value discrepancies:")
print(zero_value_df)

# Analyze the counts and distribution
print("\nCounts of zero value discrepancies by position index:")
print(zero_value_df['PositionIndex'].value_counts())

print("\nCounts of zero value discrepancies by timestamp:")
print(zero_value_df['Timestamp'].value_counts())

# Visualize the counts by position index
plt.figure(figsize=(10, 6))
sns.countplot(y='PositionIndex', data=zero_value_df, palette='viridis')
plt.title('Counts of Zero Value Discrepancies by Position Index')
plt.xlabel('Count')
plt.ylabel('Position Index')
plt.show()

# Visualize the counts by timestamp
plt.figure(figsize=(10, 6))
sns.countplot(y='Timestamp', data=zero_value_df, palette='viridis')
plt.title('Counts of Zero Value Discrepancies by Timestamp')
plt.xlabel('Count')
plt.ylabel('Timestamp')
plt.show()

#%%

# Define tolerance limits again for reference
angle_nv, angle_lt, angle_ut = 180, 175, 185
torque_nv, torque_lt, torque_ut = 16, 17.6, 40

# Extract relevant columns for all 48 screws
angle_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV' for i in range(1, 49)]
torque_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV' for i in range(1, 49)]
result_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result' for i in range(1, 49)]

# Extract the relevant data
angle_av_data = screwing_data[angle_av_columns]
torque_av_data = screwing_data[torque_av_columns]
result_data = screwing_data[result_columns]

# Initialize a list to hold discrepancies
discrepancies = []

# Loop through each screw and check for discrepancies
for i in range(1, 49):
    angle_av = angle_av_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV']
    torque_av = torque_av_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV']
    actual_result = result_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result']
    
    # Identify rows where AV is outside thresholds but result is OK (1)
    angle_discrepancies = (angle_av < angle_lt) | (angle_av > angle_ut)
    torque_discrepancies = (torque_av < torque_lt) | (torque_av > torque_ut)
    
    discrepancies_mask = (angle_discrepancies | torque_discrepancies) & (actual_result == 1)
    
    discrepancies.append(screwing_data[discrepancies_mask])

# Combine all discrepancies into a single DataFrame
discrepancies_df = pd.concat(discrepancies)

# Display the discrepancies
print(discrepancies_df)


#%% DISCREPANCIES LIST

import pandas as pd

# Define tolerance limits
angle_nv, angle_lt, angle_ut = 180, 175, 185
torque_nv, torque_lt, torque_ut = 16, 17.6, 40

# Extract relevant columns for all 48 screws
angle_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV' for i in range(1, 49)]
torque_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV' for i in range(1, 49)]
result_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result' for i in range(1, 49)]
position_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_PositionIndex' for i in range(1, 49)]

# Extract the relevant data
angle_av_data = screwing_data[angle_av_columns]
torque_av_data = screwing_data[torque_av_columns]
result_data = screwing_data[result_columns]
position_data = screwing_data[position_columns]

# Initialize a list to hold discrepancies
discrepancies = []

# Loop through each screw and check for discrepancies
for i in range(1, 49):
    angle_av = angle_av_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV']
    torque_av = torque_av_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV']
    actual_result = result_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result']
    position_index = position_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_PositionIndex']
    
    # Identify rows where AV is outside thresholds but result is OK (1)
    angle_discrepancies = (angle_av < angle_lt) | (angle_av > angle_ut)
    torque_discrepancies = (torque_av < torque_lt) | (torque_av > torque_ut)
    
    discrepancies_mask = (angle_discrepancies | torque_discrepancies) & (actual_result == 1)
    
    discrepancies_data = screwing_data[discrepancies_mask]
    discrepancies_data['PositionIndex'] = position_index[discrepancies_mask]
    
    discrepancies.append(discrepancies_data)

# Combine all discrepancies into a single DataFrame
discrepancies_df = pd.concat(discrepancies)

# Display the discrepancies
print(discrepancies_df)

# Calculate the correlation between torque AV and angle AV for the discrepancies
correlation = discrepancies_df[torque_av_columns + angle_av_columns].corr()

# Display the correlation
print("Correlation between Torque AV and Angle AV for discrepancies:")
print(correlation)


#%% Discrepancias PLOT BUENO

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Define tolerance limits
angle_nv, angle_lt, angle_ut = 180, 175, 185
torque_nv, torque_lt, torque_ut = 16, 17.6, 40

# Extract relevant columns for all 48 screws
angle_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV' for i in range(1, 49)]
torque_av_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV' for i in range(1, 49)]
result_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result' for i in range(1, 49)]
position_columns = [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_PositionIndex' for i in range(1, 49)]

# Extract the relevant data
angle_av_data = screwing_data[angle_av_columns]
torque_av_data = screwing_data[torque_av_columns]
result_data = screwing_data[result_columns]
position_data = screwing_data[position_columns]

# Initialize lists to hold discrepancies data
angle_list = []
torque_list = []
position_list = []
result_list = []
discrepancy_type_list = []

# Loop through each screw and check for discrepancies
for i in range(1, 49):
    angle_av = angle_av_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Angle_AV']
    torque_av = torque_av_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Torque_AV']
    actual_result = result_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Result']
    position_index = position_data[f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_PositionIndex']
    
    # Identify rows where result is NOK (2)
    nok_mask = actual_result == 2
    
    # Check if torque is within thresholds and angle is outside
    within_torque_threshold = (torque_av >= torque_lt) & (torque_av <= torque_ut)
    outside_angle_threshold = (angle_av < angle_lt) | (angle_av > angle_ut)
    
    # Check if angle is within thresholds and torque is outside
    within_angle_threshold = (angle_av >= angle_lt) & (angle_av <= angle_ut)
    outside_torque_threshold = (torque_av < torque_lt) | (torque_av > torque_ut)
    
    # Filter for NOK results within these conditions
    nok_within_torque = nok_mask & within_torque_threshold & outside_angle_threshold
    nok_within_angle = nok_mask & within_angle_threshold & outside_torque_threshold
    
    angle_list.extend(angle_av[nok_within_torque | nok_within_angle].values)
    torque_list.extend(torque_av[nok_within_torque | nok_within_angle].values)
    position_list.extend(position_index[nok_within_torque | nok_within_angle].values)
    result_list.extend(actual_result[nok_within_torque | nok_within_angle].values)
    
    discrepancy_type_list.extend(['Torque within, Angle outside'] * nok_within_torque.sum())
    discrepancy_type_list.extend(['Angle within, Torque outside'] * nok_within_angle.sum())

# Create a DataFrame for these discrepancies
discrepancy_df = pd.DataFrame({
    'Angle_AV': angle_list,
    'Torque_AV': torque_list,
    'PositionIndex': position_list,
    'Result': result_list,
    'DiscrepancyType': discrepancy_type_list
})

# Display the discrepancies
print("NOK discrepancies where one AV is within threshold and the other is outside threshold:")
print(discrepancy_df)

# Analyze the counts of discrepancy types
discrepancy_counts = discrepancy_df['DiscrepancyType'].value_counts()

print("\nCounts of discrepancy types:")
print(discrepancy_counts)

# Visualize the counts by discrepancy type
plt.figure(figsize=(10, 6))
sns.countplot(y='DiscrepancyType', data=discrepancy_df, palette='viridis')
plt.title('Counts of NOK Discrepancies by Type')
plt.xlabel('Count')
plt.ylabel('Discrepancy Type')
plt.show()

# Visualize the discrepancies with a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Torque_AV', y='Angle_AV', hue='DiscrepancyType', data=discrepancy_df, palette='viridis')
plt.axvline(x=torque_lt, color='cyan', linestyle='--', label='Torque Lower Tolerance')
plt.axvline(x=torque_ut, color='red', linestyle='--', label='Torque Upper Tolerance')
plt.axhline(y=angle_lt, color='cyan', linestyle='--', label='Angle Lower Tolerance')
plt.axhline(y=angle_ut, color='red', linestyle='--', label='Angle Upper Tolerance')
plt.title('Scatter Plot of NOK Discrepancies (Torque AV vs. Angle AV)')
plt.xlabel('Torque AV')
plt.ylabel('Angle AV')
plt.legend(title='Discrepancy Type')
plt.show()

#%% Total Duration VS SCREW RESULTS  no hacer


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Extract the relevant columns
duration_column = 'Total_Duration_of_St120'
screw_result_column = 'ProcessResults_Screwing_30432_Result'

# Extract the relevant data
duration_data = st120_filter[duration_column]
screw_result_data = screwing_data[screw_result_column]

# Flatten the screw results data for easier processing
screw_result_flat = screw_result_data.apply(pd.Series).stack().astype(int).reset_index(drop=True)

# Repeat the duration data to match the flattened screw result data
duration_repeated = duration_data.repeat(48).reset_index(drop=True)

# Create a DataFrame for the screw results and the total duration
screw_duration_df = pd.DataFrame({
    'Screw_Result': screw_result_flat,
    'Total_Duration_of_St120': duration_repeated
})

# Calculate NOK counts for different durations
nok_counts = screw_duration_df[screw_duration_df['Screw_Result'] == 2].groupby('Total_Duration_of_St120').size()
total_counts = screw_duration_df.groupby('Total_Duration_of_St120').size()

# Calculate NOK ratio
nok_ratio = (nok_counts / total_counts).fillna(0)

# Create a DataFrame to hold duration and NOK ratio
comparison_df = pd.DataFrame({
    'Total_Duration_of_St120': nok_ratio.index,
    'NOK_Ratio': nok_ratio.values
})

# Display the comparison DataFrame
print("Comparison of Total Duration and NOK Ratio:")
print(comparison_df)

# Visualize the comparison using a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Total_Duration_of_St120', y='NOK_Ratio', data=comparison_df)
plt.title('Comparison of Total Duration of St120 and NOK Ratio')
plt.xlabel('Total Duration of St120')
plt.ylabel('NOK Ratio')
plt.show()

'''
   Total_Duration_of_St120  NOK_Ratio
0                 3599.997   0.027778
1                 3599.998   0.000000
2                 3599.999   0.021277
3                 3600.000   0.024185
4                 7199.996   0.000000
5                 7199.997   0.000000
6                 7199.998   0.000000
7                 7199.999   0.000000
8                 7200.000   0.000000

'''


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming comparison_df is already created and contains the relevant data
plt.figure(figsize=(12, 6))
sns.lineplot(x='Total_Duration_of_St120', y='NOK_Ratio', data=comparison_df, marker='o')
plt.title('Comparison of Total Duration of St120 and NOK Ratio')
plt.xlabel('Total Duration of St120')
plt.ylabel('NOK Ratio')
plt.xticks(rotation=45)
plt.show()

# Assuming comparison_df is already created and contains the relevant data
plt.figure(figsize=(12, 6))
sns.barplot(x='Total_Duration_of_St120', y='NOK_Ratio', data=comparison_df, color='#00F1FF')
plt.title('Comparison of Total Duration of St120 and NOK Ratio for Screw')
plt.xlabel('Total Duration of St120')
plt.ylabel('NOK Ratio')
plt.xticks(rotation=45)
plt.show()