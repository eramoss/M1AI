import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

CAMINHO_MODELO = "melhor_modelo.keras"
CAMINHO_SCALER = "scaler_final.pkl"
CAMINHO_ENCODER = "label_encoder.pkl"


@st.cache_resource
def carregar_modelo():
    modelo = tf.keras.models.load_model(CAMINHO_MODELO)
    scaler = joblib.load(CAMINHO_SCALER)
    encoder = joblib.load(CAMINHO_ENCODER)
    return modelo, scaler, encoder


modelo, scaler, encoder = carregar_modelo()

st.title("🔮 Classificador com Rede Neural")

st.write("Insira os valores das features:")


feature_names = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRatio",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "Roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]

inputs = []


for i, feature in enumerate(feature_names):
    val = st.number_input(feature, value=0.0)
    inputs.append(val)

entrada = np.array(inputs).reshape(1, -1)

if st.button("Prever"):
    entrada_scaled = scaler.transform(entrada)

    probs = modelo.predict(entrada_scaled)[0]
    classe_idx = np.argmax(probs)
    classe_nome = encoder.inverse_transform([classe_idx])[0]

    st.subheader(f"Classe prevista: {classe_nome}")

    fig, ax = plt.subplots()
    ax.bar(encoder.classes_, probs)
    ax.set_ylabel("Probabilidade")
    ax.set_xlabel("Classe")
    ax.set_title("Distribuição de Probabilidades")

    st.pyplot(fig)
