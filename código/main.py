#%%
from src import exploracion as exploracion
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns

ruta_continente = "../data/Coffee_Qlty_By_Continent.csv"
ruta_país = "../data/Coffee_Qlty_By_Country.csv"
ruta_calidad = "../data/Coffee_Qlty.csv"
ruta_consumo = "../data/CoffeeConsumption.csv"

df_quality_continent, df_quality_country = exploracion.cargar_dataframes(ruta_continente, ruta_país)
df_quality, df_consumption = exploracion.cargar_dataframes(ruta_calidad, ruta_consumo)

# Configurar la visualización
exploracion.configurar_visualizacion()


dataframes = [df_consumption, df_quality, df_quality_continent, df_quality_country]
for df in dataframes: 
    # Convertir el índice en una columna
    df.reset_index(inplace=True, drop=False)


exploracion.exploracion_dataframe(df_consumption,"country")

exploracion.exploracion_dataframe(df_quality,"Species")
 
exploracion.exploracion_dataframe(df_quality_continent,"Country.of.Origin")

exploracion.exploracion_dataframe(df_quality_country,"Country.of.Origin")


dataframes = [df_consumption, df_quality, df_quality_continent, df_quality_country]
for df in dataframes: 
    exploracion.transformar_nombres_columnas(df)

new_name = {"country_of_origin":"continent_of_origin"}
df_quality_continent.rename(columns=new_name, inplace=True)

exploracion.imputar_valores_nulos(df_consumption, "percapitacons2016", estrategia='median')

exploracion.imputar_nulos(df_consumption, "totcons2019")

# Ahora, para eliminar la fila 1197
df_quality.drop(1197, inplace=True)

df_quality.head(10)

lista_moda = ["variety", "color", "processing_method"]

exploracion.imputar_nulos_moda(df_quality, lista_moda)

exploracion.imputar_valores_nulos(df_quality, "harvest_year", estrategia='median')

# Aplicar una transformación a la columna "harvest year"
df_quality["harvest_year"] = df_quality["harvest_year"].apply(exploracion.transformar_int)


exploracion.guardar_csv(df_consumption, df_quality)
# %%

ruta_df1 = "../data/calidad_cafe.csv"
ruta_df2 = "../data/consumo_cafe.csv"

df_calidad, df_consumo = exploracion.cargar_dataframes(ruta_df1, ruta_df2)
# %%

