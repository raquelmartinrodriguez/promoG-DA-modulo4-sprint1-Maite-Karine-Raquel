#%%
from src import exploracion as exploracion
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
#%%

ruta_continente = "../data/Coffee_Qlty_By_Continent.csv"
ruta_país = "../data/Coffee_Qlty_By_Country.csv"
ruta_calidad = "../data/Coffee_Qlty.csv"
ruta_consumo = "../data/CoffeeConsumption.csv"

df_quality_continent, df_quality_country = exploracion.cargar_dataframes(ruta_continente, ruta_país)
df_quality, df_consumption = exploracion.cargar_dataframes(ruta_calidad, ruta_consumo)

# %%
df_quality_continent.head(5)
# %%
df_quality_country.head(5)
# %%
df_quality.head(5)
# %%
df_consumption.head(5)
# %%
# Configurar la visualización
exploracion.configurar_visualizacion()

# %%
dataframes = [df_consumption, df_quality, df_quality_continent, df_quality_country]

for df in dataframes: 
    exploracion.exploracion_dataframe(df)