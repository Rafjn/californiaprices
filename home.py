import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd

from joblib import load
from notebooks.src.config import DADOS_GEO_MEDIAN, DADOS_LIMPOS, MODELO_FINAL

#Carregamento de dados
@st.cache_data
def carregar_dados_limpos():
    return pd.read_parquet(DADOS_LIMPOS)

@st.cache_data
def carregar_dados_geo():
    return pd.read_parquet(DADOS_GEO_MEDIAN)

@st.cache_resource
def carregar_modelo():
    return load(MODELO_FINAL)

#Variáveis para os dados
df = carregar_dados_limpos()
gdf_geo = carregar_dados_geo()
modelo = carregar_modelo()

#Título
st.title('Previsão preços de Imóveis')

#Dados de input

counties = list(gdf_geo['name'].sort_values())

selecionar_condado = st.selectbox('Condados', counties)

longitude = gdf_geo.query('name == @selecionar_condado')['longitude'].values
latitude = gdf_geo.query('name == @selecionar_condado')['latitude'].values

housing_median_age = st.number_input('Idade do Imóvel', value= 10, min_value=df['housing_median_age'].min(), max_value = df['housing_median_age'].max() )

total_rooms = gdf_geo.query('name == @selecionar_condado')['total_rooms'].values
total_bedrooms = gdf_geo.query('name == @selecionar_condado')['total_bedrooms'].values
population = gdf_geo.query('name == @selecionar_condado')['population'].values
households = gdf_geo.query('name == @selecionar_condado')['households'].values

median_income = st.slider('Renda média (milhares de US$)', 5.0, 100.0, 45.0, 5.0)

ocean_proximity = gdf_geo.query('name == @selecionar_condado')['ocean_proximity'].values

bins_income = [0, 1.5, 3, 4.5, 6, np.inf]
median_income_cat = np.digitize(median_income/10, bins_income)

rooms_per_households = gdf_geo.query('name == @selecionar_condado')['rooms_per_households'].values
bedrooms_per_room = gdf_geo.query('name == @selecionar_condado')['bedrooms_per_room'].values
population_per_househoulds = gdf_geo.query('name == @selecionar_condado')['population_per_househoulds'].values

#Recebendo os dados do modelo
entrada_modelo = {
    'longitude': longitude,
    'latitude': latitude,
    'housing_median_age': housing_median_age,
    'total_rooms': total_rooms,
    'total_bedrooms': total_bedrooms,
    'population': population,
    'households': households,
    'median_income': median_income / 10,
    'ocean_proximity': ocean_proximity,
    'median_income_cat': median_income_cat,
    'rooms_per_households': rooms_per_households,
    'bedrooms_per_room': bedrooms_per_room,
    'population_per_househoulds': population_per_househoulds,
}

#Retorno do modelo e botão
df_entrada_modelo = pd.DataFrame(entrada_modelo, index=[0])

botao_previsao = st.button('Prever preço')

if botao_previsao:
    preco = modelo.predict(df_entrada_modelo)
    st.write(f'Preço previsto: U$ {preco[0][0]:.2f}')



