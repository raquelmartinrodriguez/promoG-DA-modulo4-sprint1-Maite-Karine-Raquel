
#%%
# exploracion.py
import pandas as pd
import numpy as np 
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

def cargar_dataframes(ruta_csv1, ruta_csv2):
    """Carga los dataframes desde archivos CSV.

    Args:
    - ruta_flight (str): La ruta al archivo CSV de Customer Flight Activity.
    - ruta_loyalty (str): La ruta al archivo CSV de Customer Loyalty History.

    Returns:
    - df_flight (DataFrame): El dataframe cargado desde el archivo CSV de Customer Flight Activity.
    - df_loyalty (DataFrame): El dataframe cargado desde el archivo CSV de Customer Loyalty History.
    """
    df1= pd.read_csv(ruta_csv1, index_col=0)
    df2 = pd.read_csv(ruta_csv2, index_col=0)
    return df1, df2

def combinar_dataframes(df1, df2):
    """Combina los dataframes df_flight y df_loyalty utilizando la función merge.

    Args:
    - df_flight (DataFrame): El dataframe de Customer Flight Activity.
    - df_loyalty (DataFrame): El dataframe de Customer Loyalty History.

    Returns:
    - df(DataFrame): El dataframe resultante de la combinación de df_flight y df_loyalty.
    """
    df_merge = pd.merge(df1, df2, left_index=True, right_index=True)
    return df_merge


def configurar_visualizacion():
    """Configura la visualización de pandas para mostrar todas las columnas y formato de los números."""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', '{:.2f}'.format)


def exploracion_dataframe(dataframe,columna_control):
    """Realiza un análisis exploratorio básico de un DataFrame.

    Args:
    - df_merge(DataFrame): El DataFrame que se va a explorar.
    - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.
    """
    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ..................... \n")
    
    # generamos un DataFrame para los valores nulos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0])
    
    print("\n ..................... \n")
    print(f"Los tipos de las columnas son:")
    display(pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"]))
    
    
    print("\n ..................... \n")
    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = dataframe.select_dtypes(include = "O")
    
    for col in dataframe_categoricas.columns:
        print(f"La columna {col.upper()} tiene las siguientes valore únicos:")
        display(pd.DataFrame(dataframe[col].value_counts()).head())    
    
    # Sacamos los principales estadísticos del dataframe 
    
    for categoria in dataframe[columna_control].unique():
        
        dataframe_filtrado = dataframe[dataframe[columna_control] == categoria]
    
        print("\n ..................... \n")
        print(f"Los principales estadísticos de las columnas categóricas para el {categoria.upper()} son: ")
        display(dataframe_filtrado.describe(include = "O").T)
        
        print("\n ..................... \n")
        print(f"Los principales estadísticos de las columnas numéricas para el {categoria.upper()} son: ")
        display(dataframe_filtrado.describe().T)
        
    return dataframe


def imputar_valores_nulos(dataframe, columna, estrategia='median'):
    """Imputa valores nulos en una columna del DataFrame utilizando SimpleImputer.

    Args:
    - dataframe (DataFrame): El DataFrame en el que se imputarán los valores nulos.
    - columna (str): El nombre de la columna en la que se imputarán los valores nulos.
    - estrategia (str, optional): La estrategia a utilizar para la imputación. Por defecto es 'median'.
    """
    
    imputer = SimpleImputer(strategy=estrategia)
    columna_imputada = imputer.fit_transform(dataframe[[columna]])
    dataframe[columna] = columna_imputada

def imputar_nulos_moda(dataframe, columnas_moda):
    """Imputa valores nulos en una columna del DataFrame utilizando SimpleImputer.

    Args:
    - dataframe (DataFrame): El DataFrame en el que se imputarán los valores nulos.
    - columna (str): El nombre de la columna en la que se imputarán los valores nulos.
    - estrategia (str, optional): La estrategia a utilizar para la imputación. Por defecto es 'median'.
    
    Returns:
    - dataframe (DataFrame): El DataFrame actualizado después de la imputación.
    """
    # iteramos por la lista creada en el paso anterior:
    for columna in columnas_moda:
        
        # calculamos la moda para la columna por la que estamos iterando
        moda = dataframe[columna].mode()[0]    
        
        # utilizando el método fillna reemplazamos los valores nulos por la moda calculada en el paso anterior. 
        dataframe[columna] = dataframe[columna].fillna(moda)
        
    # por último chequeamos si se han eliminado los nulos en las columnas de "marital" y "loan"
    print("Después del reemplazo usando 'fillna' quedan los siguientes nulos")

    dataframe[columnas_moda].isnull().sum()
        
    return dataframe

def imputar_nulos(dataframe, columna): 
    imputer_iterative = IterativeImputer(max_iter=20, random_state=42)

    # Ajustamos y transformamos los datos
    imputer_iterative_imputado = imputer_iterative.fit_transform(dataframe[[columna]])  # Pasamos la columna como DataFrame
    return dataframe

def eliminar_columnas(dataframe, columnas):
    """Elimina columnas del DataFrame.

    Args:
    - dataframe (DataFrame): El DataFrame del que se eliminarán las columnas.
    - columnas (list): La lista de nombres de las columnas que se eliminarán.
    """
    dataframe.drop(columns=columnas, inplace=True)

def transformar_nombres_columnas(dataframe):
    """Transforma los nombres de las columnas del DataFrame.

    Args:
    - dataframe (DataFrame): El DataFrame cuyos nombres de columnas se transformarán.
    """
    nuevas_columnas = [col.replace(".", '_').lower() for col in dataframe.columns]
    dataframe.columns = nuevas_columnas
    
    return dataframe

def transformar_int(valor):
    """Transforma el valor de la columna 'Salario'.

    Args:
    - valor: El valor que se transformará.

    Returns:
    - int: El valor transformado.
    """
    if pd.notnull(valor):
        valor = str(valor)
        valor = valor.split('.')[0]  # Obtiene la parte entera antes del punto
        return int(valor)
    else:
        return np.nan


def analisis_frecuencia_cancelaciones(dataframe, columnas):
    """Realiza un análisis de frecuencia de cancelaciones según meses y años.

    Args:
    - dataframe (DataFrame): El DataFrame en el que se realizará el análisis.
    - columnas (list): La lista de nombres de las columnas a analizar.
    """
    for columnas_analizar in columnas:
        for columna in columnas_analizar:
            frecuencia_valores = dataframe[columna].value_counts()
            print(f"La frecuencia de cancelaciones para la columna '{columna}' es:\n {frecuencia_valores}")

def guardar_csv(dataframe1, dataframe2):
    """
    Guarda un DataFrame en un archivo CSV.

    Args:
    - dataframe: El DataFrame que se guardará en CSV.

    Returns:
    - None
    """
    lista_df = [dataframe1, dataframe2]
    
    for df in lista_df: 
        ruta_archivo = input("Inserte la ruta completa junto con el nombre del archivo CSV para guardar el DataFrame: ")
        df.to_csv(ruta_archivo, index=False)  # Usamos index=False para evitar que se guarde el índice del DataFrame

# %%
