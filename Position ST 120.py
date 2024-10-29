# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 00:52:28 2024

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
#%% SHAP SIN FILTROS

if 'ProcessResults_PlacingBatModule_Result' in st120.columns:
    process_results = st120['ProcessResults_PlacingBatModule_Result']
    print(process_results)
else:
    print("The column 'ProcessResults_PlacingBatModule_Result' does not exist in the DataFrame.")

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import shap
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight

# Select only numeric columns
st120_numeric = st120.select_dtypes(include=['number'])

# Map target values from [1, 2] to [0, 1]
if 'ProcessResults_PlacingBatModule_Result' in st120_numeric.columns:
    st120_numeric['ProcessResults_PlacingBatModule_Result'] = st120_numeric['ProcessResults_PlacingBatModule_Result'].map({1: 0, 2: 1})

# Debug: Check unique values after mapping
print("Unique values in 'ProcessResults_PlacingBatModule_Result' after mapping:", st120_numeric['ProcessResults_PlacingBatModule_Result'].unique())

# Function to train XGBoost and explain with SHAP
def train_and_explain(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Debug: Check unique values in y
    print("Unique values in target variable y:", y.unique())

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Debug: Check unique values in y_train and y_test
    print("Unique values in y_train:", y_train.unique())
    print("Unique values in y_test:", y_test.unique())

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

# Ensure that 'ProcessResults_PlacingBatModule_Result' is in the numeric DataFrame
if 'ProcessResults_PlacingBatModule_Result' in st120_numeric.columns:
    # Train and explain for st120_numeric
    shap_values_st120, feature_names_st120 = train_and_explain(st120_numeric, 'ProcessResults_PlacingBatModule_Result')

    # Create SHAP plots for st120_numeric
    create_shap_plots(shap_values_st120, "st120_numeric", feature_names_st120)
else:
    print("The target column 'ProcessResults_PlacingBatModule_Result' is not in the numeric DataFrame.")


#%% CLEANING


# Define columns to keep based on the provided criteria
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
He quitado:
    'TotalDuration_FirstStart_LastEnd' # SEGUNDA OPCIÓN DE TIEMPO NO HA SALIDO BIEN.
    [f'ProcessResults_MaterialConsumption_RegularConsumption_RegularConsumption[{i}]_Charge' for i in range(1, 13)] + \
    [f'ProcessResults_Screwing_30432_Screws_Screws[{i}]_Charge' for i in range(1, 49)] + \
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
screwing_data_sample.to_excel("screwing_data_sample.xlsx", index=False)

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



#%% SHAP FOR POSITION

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import shap
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight

# Map target values from [1, 2] to [0, 1]
if 'ProcessResults_PlacingBatModule_Result' in st120_ML.columns:
    st120_ML['ProcessResults_PlacingBatModule_Result'] = st120_ML['ProcessResults_PlacingBatModule_Result'].map({1: 0, 2: 1})

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
shap_values_st120, feature_names_st120 = train_and_explain(st120_ML, 'ProcessResults_PlacingBatModule_Result')

# Create SHAP plots for st120_ML
create_shap_plots(shap_values_st120, "st120_ML", feature_names_st120)

#%%


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

#%% FORCE CORRELATION


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Extracting Force_AV and EndPositonModul_Result columns for each module
force_columns = [f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_Force_AV' for i in range(1, 13)]
end_position_columns = [f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_EndPositonModul_Result' for i in range(1, 13)]

force_data = placing_batmodule[force_columns]
end_position_data = placing_batmodule[end_position_columns]

# Flatten data and add module and result columns
flattened_data = pd.DataFrame()

for i in range(1, 13):
    force_col = f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_Force_AV'
    end_pos_col = f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_EndPositonModul_Result'
    
    module_data = pd.DataFrame({
        'Force_AV': force_data[force_col],
        'EndPositonModul_Result': end_position_data[end_pos_col],
        'Module': i
    })
    flattened_data = pd.concat([flattened_data, module_data], ignore_index=True)

# Define OK (1) and NOK (2) results
flattened_data['Result'] = flattened_data['EndPositonModul_Result'].apply(lambda x: 'OK' if x == 1 else 'NOK')

# Visualize the data using a scatter plot
plt.figure(figsize=(15, 7))
sns.scatterplot(x='Module', y='Force_AV', hue='Result', data=flattened_data, palette={'OK': 'cyan', 'NOK': 'red'}, edgecolor='black')
plt.title('Relationship between Force_AV and EndPositonModul_Result for All Modules for ST 120')
plt.xlabel('Module')
plt.ylabel('Force_AV')
plt.legend(title='Result')
plt.show()

#%% CARRIER ROUND TRIPS



# SECOND OPTION

import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'st120_filter'
time_and_roundtrips_new = st120_filter[['_data[0]_ts', 'ProcessResults_Info_CarrierRoundtrips', 'ProcessResults_PlacingBatModule_Result']]

# Convert the '_data[0]_ts' column to datetime format
time_and_roundtrips_new['_data[0]_ts'] = pd.to_datetime(time_and_roundtrips_new['_data[0]_ts'])

# Plotting time on x-axis and carrier roundtrips on y-axis with OK and NOK values distinctly
plt.figure(figsize=(14, 7))

# Plot OK values
plt.scatter(time_and_roundtrips_new[time_and_roundtrips_new['ProcessResults_PlacingBatModule_Result'] == 1]['_data[0]_ts'],
            time_and_roundtrips_new[time_and_roundtrips_new['ProcessResults_PlacingBatModule_Result'] == 1]['ProcessResults_Info_CarrierRoundtrips'],
            color='blue', label='OK', alpha=0.6)

# Plot NOK values
plt.scatter(time_and_roundtrips_new[time_and_roundtrips_new['ProcessResults_PlacingBatModule_Result'] == 2]['_data[0]_ts'],
            time_and_roundtrips_new[time_and_roundtrips_new['ProcessResults_PlacingBatModule_Result'] == 2]['ProcessResults_Info_CarrierRoundtrips'],
            color='red', label='NOK', alpha=0.6)

# Add axis labels
plt.xlabel('Time')
plt.ylabel('Number of Carrier Roundtrips')
plt.title('Carrier Roundtrips Over Time for OK (Blue) and NOK (Red) Results')
plt.legend()
plt.grid(axis='both')

# Save and display the plot
plt.savefig('comparison_carrier_roundtrips_over_time_new.png')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'st120_filter'
time_and_roundtrips_new = st120_filter[['_data[0]_ts', 'ProcessResults_Info_CarrierRoundtrips', 'ProcessResults_PlacingBatModule_Result']]

# Convert the '_data[0]_ts' column to datetime format
time_and_roundtrips_new['_data[0]_ts'] = pd.to_datetime(time_and_roundtrips_new['_data[0]_ts'])

# Define colors
color_ok = 'blue'
color_nok = 'red'

# Plotting time on x-axis and carrier roundtrips on y-axis with OK and NOK values distinctly
plt.figure(figsize=(14, 7))

# Plot OK values with outlines
plt.scatter(time_and_roundtrips_new[time_and_roundtrips_new['ProcessResults_PlacingBatModule_Result'] == 1]['_data[0]_ts'],
            time_and_roundtrips_new[time_and_roundtrips_new['ProcessResults_PlacingBatModule_Result'] == 1]['ProcessResults_Info_CarrierRoundtrips'],
            color=color_ok, edgecolor='black', label='OK', alpha=0.6)

# Plot NOK values with outlines
plt.scatter(time_and_roundtrips_new[time_and_roundtrips_new['ProcessResults_PlacingBatModule_Result'] == 2]['_data[0]_ts'],
            time_and_roundtrips_new[time_and_roundtrips_new['ProcessResults_PlacingBatModule_Result'] == 2]['ProcessResults_Info_CarrierRoundtrips'],
            color=color_nok, edgecolor='black', label='NOK', alpha=0.6)

# Add axis labels
plt.xlabel('Time')
plt.ylabel('Number of Carrier Roundtrips')
plt.title('Carrier Roundtrips Over Time for OK and NOK Results st 120')
plt.legend()
plt.grid(axis='both')

# Save and display the plot
plt.savefig('comparison_carrier_roundtrips_over_time_default_colors.png')
plt.show()

#%% Material 

import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'st120_filter'
# Select relevant columns
time_and_material_consumption = st120_filter[['_data[0]_ts', 'ProcessResults_MaterialConsumption_RegularConsumption_RegularConsumption[1]_DeliveryNumber', 'ProcessResults_PlacingBatModule_Result']]

# Convert the '_data[0]_ts' column to datetime format
time_and_material_consumption['_data[0]_ts'] = pd.to_datetime(time_and_material_consumption['_data[0]_ts'])

# Define colors for OK and NOK
color_ok = 'blue'
color_nok = 'red'

# Plotting time on x-axis and material consumption delivery number on y-axis with OK and NOK values distinctly
plt.figure(figsize=(14, 7))

# Plot OK values with outlines
plt.scatter(time_and_material_consumption[time_and_material_consumption['ProcessResults_PlacingBatModule_Result'] == 0]['_data[0]_ts'],
            time_and_material_consumption[time_and_material_consumption['ProcessResults_PlacingBatModule_Result'] == 0]['ProcessResults_MaterialConsumption_RegularConsumption_RegularConsumption[1]_DeliveryNumber'],
            color=color_ok, edgecolor='black', label='OK', alpha=0.6)

# Plot NOK values with outlines
plt.scatter(time_and_material_consumption[time_and_material_consumption['ProcessResults_PlacingBatModule_Result'] == 1]['_data[0]_ts'],
            time_and_material_consumption[time_and_material_consumption['ProcessResults_PlacingBatModule_Result'] == 1]['ProcessResults_MaterialConsumption_RegularConsumption_RegularConsumption[1]_DeliveryNumber'],
            color=color_nok, edgecolor='black', label='NOK', alpha=0.6)

# Add axis labels
plt.xlabel('Time')
plt.ylabel('Material Consumption Delivery Number')
plt.title('Material Consumption Delivery Number Over Time for OK (Blue) and NOK (Red) Results')
plt.legend()
plt.grid(axis='both')

# Save and display the plot
plt.savefig('material_consumption_delivery_over_time.png')
plt.show()



# OTRO

import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'st120_filter'
# Select relevant columns
time_and_material_consumption = st120_filter[['_data[0]_ts', 'ProcessResults_MaterialConsumption_RegularConsumption_RegularConsumption[1]_DeliveryNumber', 'ProcessResults_PlacingBatModule_Result']]

# Convert the '_data[0]_ts' column to datetime format
time_and_material_consumption['_data[0]_ts'] = pd.to_datetime(time_and_material_consumption['_data[0]_ts'])

# Check the unique values in 'ProcessResults_PlacingBatModule_Result'
unique_values = time_and_material_consumption['ProcessResults_PlacingBatModule_Result'].unique()
print("Unique values in 'ProcessResults_PlacingBatModule_Result':", unique_values)

# Create a mapping for result types, assuming 1 is OK and 2 is NOK
result_mapping = {1: 'OK', 2: 'NOK'}

# Apply mapping to 'ProcessResults_PlacingBatModule_Result' column
time_and_material_consumption['Result_Type'] = time_and_material_consumption['ProcessResults_PlacingBatModule_Result'].map(result_mapping)

# Check for any unmapped values (NaN) after mapping
unmapped_values = time_and_material_consumption[time_and_material_consumption['Result_Type'].isna()]
print("Unmapped values:\n", unmapped_values[['ProcessResults_PlacingBatModule_Result', 'Result_Type']])

# Calculate mean delivery numbers for OK and NOK results
mean_delivery_numbers = time_and_material_consumption.groupby('Result_Type')['ProcessResults_MaterialConsumption_RegularConsumption_RegularConsumption[1]_DeliveryNumber'].mean()

# Create a bar plot
plt.figure(figsize=(10, 6))
mean_delivery_numbers.plot(kind='bar', color=['blue', 'red'], edgecolor='black')

# Add labels and title
plt.xlabel('Result Type')
plt.ylabel('Average Material Consumption Delivery Number')
plt.title('Average Material Consumption Delivery Number by Result Type')
plt.grid(axis='y')

# Save and display the plot
plt.savefig('average_material_consumption_delivery_number.png')
plt.show()

#%% COUNTERHOLDING - Not much relation.


import seaborn as sns
import matplotlib.pyplot as plt

# Extract the relevant columns for correlation analysis
counterholding_cols = [f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_Counterholding_Result' for i in range(1, 13)]
endpositionmodul_cols = [f'ProcessResults_PlacingBatModule_Positions_Positions[{i}]_EndPositonModul_Result' for i in range(1, 13)]

# Create a DataFrame with the selected columns
correlation_data = st120_filter[counterholding_cols + endpositionmodul_cols]

# Calculate the correlation matrix
correlation_matrix = correlation_data.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Counterholding and End Position Module Results')
plt.show()
