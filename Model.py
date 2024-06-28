## IMPORTACIÓN DE LIBRERÍAS

import pandas as pd # Pandas
import matplotlib.pyplot as plt # Matplorlib
import matplotlib.colors as mcolors # Matplorlib
import matplotlib.dates as mdates
from dateutil.relativedelta import relativedelta # Dates
import seaborn as sns #Seaborn
import numpy as np #Numpy
from matplotlib.colors import LinearSegmentedColormap #Paleta de colores
import pycaret
from pycaret.regression import *
from prophet import Prophet
from pycaret.regression import save_model
from pycaret.regression import load_model
# Desactivamos Warnings
#import warnings
#warnings.filterwarnings('ignore')



df_completo = pd.read_csv("/mnt/c6c29b92-c396-4913-b7c7-b2456280d17a/Software y Programación/Ciencia de Datos/Datathon/df_completo.csv")

reg = setup(data = df_completo, target = 'Cant_vendidas', session_id=123,
            ignore_features = ['Fecha_venta', 'Año', 'Status', 'ID_tienda', 'Mes', 'Facturacion'],
            categorical_features=['ID_producto', 'ID_proveedor', 'Categoria', 'Clasif_proveedor', 'Dia_sem', 'Promocion'],  # nos aseguramos que se trate como categórica
            normalize = True, transformation = True,
            transform_target = True, fold=5)  # Usando validación cruzada de 5 pliegues)

best_model = compare_models()

#Afinación del Modelo. Se evaluan  automaticamente los Hipermarametros a utilizar.
tuned_model = tune_model(best_model)

predictions = predict_model(tuned_model, data=df_completo)

save_model(tuned_model, 'modelo_final')