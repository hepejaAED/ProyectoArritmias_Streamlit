import json
import numpy as np
import streamlit as st
from src.data_loader import load_data
from src.model import load_model
from src.utils import load_threshold, save_threshold

MODEL_PATH = "models/best_model.pkl"
THRESHOLD_PATH = "models/threshold.json"

model = load_model(MODEL_PATH)

# ==============================
# CARGAR THRESHOLD
# ==============================
st.sidebar.header("Threshold")

# cargar valor actual del json
threshold = load_threshold()

# slider sincronizado
new_threshold = st.sidebar.slider(
    "Threshold de clasificación",
    0.0, 1.0,
    value=threshold,
    step=0.01
)

# si cambia → guardar
if new_threshold != threshold:
    save_threshold(new_threshold)
    threshold = new_threshold






st.title("Predictor")

st.write(f"Threshold actual: **{threshold:.2f}**")

# ==============================
# FORMULARIO
# ==============================
df = load_data("data/Arritmias.csv")
X_cols = df.drop(columns=["AV", "PACIENTES"]).columns

inputs = {}

with st.form("predict_form"):

    for col in X_cols:

        if col.upper() == "SEXO":
            sexo = st.selectbox(
                "SEXO",
                options=["Hombre", "Mujer"]
            )
            inputs[col] = 1 if sexo == "Hombre" else 2

        elif col.upper() == "EDAD":
            inputs[col] = st.number_input(
                "EDAD",
                value=int(df[col].iloc[0]),
                step=1,
                format="%d"
            )

        else:
            inputs[col] = st.number_input(
                col,
                value=float(df[col].iloc[0])
            )

    submit = st.form_submit_button("Predecir")

# ==============================
# PREDICCIÓN
# ==============================
if submit:
    X_input = np.array([list(inputs.values())])

    prob = model.predict_proba(X_input)[:, 1][0]
    pred = int(prob >= threshold)

    st.subheader("Resultado")
    st.write(f"Probabilidad: **{prob:.3f}**")
    if pred == 1:
        st.write("Predicción: **AV presente**")
    else:
        st.write("Predicción: **AV ausente**")
