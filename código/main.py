# %%

# Importación de módulos y funciones necesarias
from src import exploracion as exploracion
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Definición de las rutas de los archivos CSV
ruta_continente = "../data/Coffee_Qlty_By_Continent.csv"
ruta_país = "../data/Coffee_Qlty_By_Country.csv"
ruta_calidad = "../data/Coffee_Qlty.csv"
ruta_consumo = "../data/CoffeeConsumption.csv"

# Carga de DataFrames desde archivos CSV
df_quality_continent, df_quality_country = exploracion.cargar_dataframes(ruta_continente, ruta_país)
df_quality, df_consumption = exploracion.cargar_dataframes(ruta_calidad, ruta_consumo)

# Configuración de la visualización
exploracion.configurar_visualizacion()

# Conversión del índice en una columna para cada DataFrame
dataframes = [df_consumption, df_quality, df_quality_continent, df_quality_country]
for df in dataframes: 
    df.reset_index(inplace=True, drop=False)

# Exploración de los DataFrames por una columna específica
exploracion.exploracion_dataframe(df_consumption,"country")
exploracion.exploracion_dataframe(df_quality,"Species")
exploracion.exploracion_dataframe(df_quality_continent,"Country.of.Origin")
exploracion.exploracion_dataframe(df_quality_country,"Country.of.Origin")

# Transformación de los nombres de las columnas para cada DataFrame
dataframes = [df_consumption, df_quality, df_quality_continent, df_quality_country]
for df in dataframes: 
    exploracion.transformar_nombres_columnas(df)

# Renombrado de columnas en un DataFrame específico
new_name = {"country_of_origin":"continent_of_origin"}
df_quality_continent.rename(columns=new_name, inplace=True)

# Imputación de valores nulos en una columna específica utilizando la mediana como estrategia
exploracion.imputar_nulos_mediana(df_consumption, "percapitacons2016", estrategia='median')
exploracion.imputar_nulos_iterative(df_consumption, "totcons2019")

# Eliminación de una fila específica en un DataFrame
df_quality.drop(1197, inplace=True)

# Imputación de valores nulos en columnas específicas utilizando la moda como estrategia
lista_moda = ["variety", "color", "processing_method"]
exploracion.imputar_nulos_moda(df_quality, lista_moda)
exploracion.imputar_nulos_mediana(df_quality, "harvest_year", estrategia='median')

# Aplicación de una transformación a una columna específica
df_quality["harvest_year"] = df_quality["harvest_year"].apply(exploracion.transformar_int)

# Guardado de DataFrames en archivos CSV
exploracion.guardar_csv(df_consumption, df_quality)

# Carga de DataFrames desde archivos CSV para comprobar que estan los cambios aplicados a los CSV modificados 
ruta_df1 = "../data/calidad_cafe.csv"
ruta_df2 = "../data/consumo_cafe.csv"
df_calidad, df_consumo = exploracion.cargar_dataframes(ruta_df1, ruta_df2)

# %%

