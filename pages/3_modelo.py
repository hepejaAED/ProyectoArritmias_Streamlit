import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

from src.data_loader import load_data, get_marker_cols
from src.model import load_model

st.set_page_config(page_title="Modelo", layout="wide")

st.title("🤖 Análisis del modelo entrenado")
st.markdown("---")


# ==============================
# CARGAR DATOS
# ==============================
@st.cache_data
def load_dataset():
    return load_data("data/Arritmias.csv")

df = load_dataset()

marker_cols = get_marker_cols(df)

X = df.drop(columns=["AV","PACIENTES"])
y = df["AV"]


# ==============================
# CARGAR MODELO
# ==============================
MODEL_PATH = "models/best_model.pkl"

model = load_model(MODEL_PATH)


# ==============================
# THRESHOLD CONTROL
# ==============================
st.sidebar.header("⚙️ Configuración")

threshold = st.sidebar.slider(
    "Threshold de clasificación",
    0.0, 1.0, 0.5, 0.01
)


# ==============================
# PREDICCIÓN
# ==============================
y_prob = model.predict_proba(X)[:, 1]
y_pred = (y_prob >= threshold).astype(int)


# ==============================
# MÉTRICAS
# ==============================
auc = roc_auc_score(y, y_prob)
cm = confusion_matrix(y, y_pred)

st.header("📊 Resultados del modelo")

col1, col2, col3 = st.columns(3)

tn, fp, fn, tp = cm.ravel()

with col1:
    st.metric("AUC", f"{auc:.3f}")

with col2:
    st.metric("Sensibilidad", f"{tp/(tp+fn):.3f}" if (tp+fn)>0 else 0)

with col3:
    st.metric("Especificidad", f"{tn/(tn+fp):.3f}" if (tn+fp)>0 else 0)


# ==============================
# ROC CURVE
# ==============================
st.subheader("📈 Curva ROC")

fpr, tpr, _ = roc_curve(y, y_prob)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
ax.plot([0,1],[0,1],"--")

ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.legend()

st.pyplot(fig)


# ==============================
# MATRIZ DE CONFUSIÓN
# ==============================
st.subheader("📊 Matriz de confusión (threshold ajustado)")

st.write(pd.DataFrame(
    cm,
    index=["Real 0", "Real 1"],
    columns=["Pred 0", "Pred 1"]
))


# ==============================
# SHAP (opcional pero potente)
# ==============================
st.subheader("🔍 Interpretabilidad (SHAP)")

try:
    import shap

    explainer = shap.Explainer(model.named_steps["model"])
    X_scaled = model.named_steps["scaler"].transform(X)

    shap_values = explainer(X_scaled)

    st.write("Top features (resumen)")

    shap_summary = pd.DataFrame(
        np.abs(shap_values.values).mean(axis=0),
        index=marker_cols,
        columns=["importance"]
    ).sort_values("importance", ascending=False)

    st.dataframe(shap_summary)

except Exception as e:
    st.warning(f"SHAP no disponible: {e}")