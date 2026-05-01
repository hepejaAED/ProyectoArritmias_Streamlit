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


# ==============================
# FORMULARIO
# ==============================
df = load_data("data/Arritmias.csv")
X_cols = df.drop(columns=["AV", "PACIENTES"]).columns

inputs = {}

st.markdown("### Datos del paciente")

tab1, tab2 = st.tabs(["✍️ Manual", "📁 CSV"])
with tab1:

    st.caption(f"Predefinido hay datos de ejemplo")
    with st.form("predict_form"):

        cols = st.columns(3) 

        i = 0

        for col in X_cols:

            current_col = cols[i % 3]

            with current_col:

                col_clean = col.upper()

                # ================= SEXO =================
                if col_clean == "SEXO":
                    sexo = st.selectbox(
                        "SEXO",
                        options=["Hombre", "Mujer"],
                        key="sexo"
                    )
                    inputs[col] = 1 if sexo == "Hombre" else 2

                # ================= EDAD =================
                elif col_clean == "EDAD":
                    inputs[col] = st.number_input(
                        "EDAD",
                        value=int(df[col].iloc[0]),
                        step=1,
                        format="%d",
                        key=col
                    )

                # ================= RESTO =================
                else:
                    inputs[col] = st.number_input(
                        col,
                        value=float(df[col].iloc[0]),
                        key=col
                    )

            i += 1

        submit = st.form_submit_button(
            "🔍 Predecir",
            use_container_width=True
        )
with tab2:

    uploaded_file = st.file_uploader(
        "Sube un CSV con las mismas columnas del modelo",
        type=["csv"]
    )

    if uploaded_file is not None:
        import pandas as pd
        df_input = pd.read_csv(uploaded_file, decimal=",", sep=";")

        st.write("Vista previa:")
        st.dataframe(df_input.head())

        if st.button("📊 Predecir CSV"):

            X_input = df_input[X_cols]

            prob = model.predict_proba(X_input)[:, 1]
            pred = (prob >= threshold).astype(int)

            df_input["probabilidad"] = prob
            df_input["prediccion"] = pred

            st.success("Predicción completada")
            st.dataframe(df_input)

            st.download_button(
                "📥 Descargar resultados",
                data=df_input.to_csv(index=False, decimal=",", sep=";"),
                file_name="predicciones_resultado.csv",
                mime="text/csv"
            )
# ==============================
# PREDICCIÓN
# ==============================
if submit:
    X_input = np.array([list(inputs.values())])

    prob = model.predict_proba(X_input)[:, 1][0]
    pred = int(prob >= threshold)

    st.markdown("## Resultado del paciente")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Probabilidad de AV",
            value=f"{prob:.3f}"
        )

    with col2:
        st.metric(
            label="Clasificación",
            value="AV presente" if pred == 1 else "AV ausente"
        )


    # Interpretación clínica simple

    if prob >= threshold:
        st.error("Riesgo ALTO")
    else:
        st.success("Riesgo BAJO")

    st.caption(f"Threshold actual: {threshold:.2f}")
