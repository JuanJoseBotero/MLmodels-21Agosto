import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# ---------------------------
# Función para crear dataset sintético
# ---------------------------
def generar_dataset(n_muestras=100, n_columnas=4):
    deportes = ["Fútbol", "Baloncesto", "Tenis", "Natación", "Ciclismo"]
    paises = ["Colombia", "EE.UU.", "España", "Brasil", "Argentina"]
    genero = ["Masculino", "Femenino"]

    data = {
        "Deporte": np.random.choice(deportes, n_muestras),
        "País": np.random.choice(paises, n_muestras),
        "Género": np.random.choice(genero, n_muestras),
        "Edad": np.random.randint(15, 40, n_muestras),
        "Altura_cm": np.random.randint(150, 210, n_muestras),
        "Peso_kg": np.random.randint(50, 100, n_muestras),
        "Puntaje": np.random.randint(0, 100, n_muestras)
    }

    df = pd.DataFrame(data)

    # Limitar columnas a máximo 6
    return df.iloc[:, :n_columnas]

# ---------------------------
# Configuración Streamlit
# ---------------------------
st.set_page_config(page_title="EDA Deportivo", layout="wide")

st.title("📊 Análisis Exploratorio de Datos (EDA) - Deportes")

# Selección de parámetros
n_muestras = st.slider("Número de muestras", 50, 500, 200)
n_columnas = st.slider("Número de columnas", 2, 6, 4)

# Generar dataset
df = generar_dataset(n_muestras, n_columnas)

st.subheader("📋 Vista previa de los datos")
st.dataframe(df.head())

# Selección de columnas a analizar
columnas = st.multiselect("Selecciona columnas para analizar:", df.columns, default=df.columns.tolist())

# Selección de tipo de gráfica
opciones_graficas = ["Histograma", "Gráfico de Barras", "Dispersión", "Tendencia", "Pastel"]
grafica = st.selectbox("Selecciona el tipo de gráfica:", opciones_graficas)

# ---------------------------
# Visualizaciones
# ---------------------------
if grafica == "Histograma":
    col_num = st.selectbox("Selecciona columna numérica:", df.select_dtypes(include=np.number).columns)
    fig, ax = plt.subplots()
    sns.histplot(df[col_num], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

elif grafica == "Gráfico de Barras":
    col_cat = st.selectbox("Selecciona columna categórica:", df.select_dtypes(exclude=np.number).columns)
    fig, ax = plt.subplots()
    df[col_cat].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

elif grafica == "Dispersión":
    col_x = st.selectbox("Eje X:", df.select_dtypes(include=np.number).columns)
    col_y = st.selectbox("Eje Y:", df.select_dtypes(include=np.number).columns)
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[col_x], y=df[col_y], hue=df["Deporte"] if "Deporte" in df.columns else None, ax=ax)
    st.pyplot(fig)

elif grafica == "Tendencia":
    col_num = st.selectbox("Selecciona columna numérica:", df.select_dtypes(include=np.number).columns)
    fig, ax = plt.subplots()
    ax.plot(df.index, df[col_num], marker="o")
    ax.set_title(f"Tendencia de {col_num}")
    st.pyplot(fig)

elif grafica == "Pastel":
    col_cat = st.selectbox("Selecciona columna categórica:", df.select_dtypes(exclude=np.number).columns)
    fig, ax = plt.subplots()
    df[col_cat].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

# ---------------------------
# Mostrar tabla completa
# ---------------------------
if st.checkbox("Mostrar tabla completa"):
    st.dataframe(df)
