import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from sklearn.metrics import (
    roc_curve,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay
)
from src.utils import load_threshold, save_threshold

from src.data_loader import load_data, get_marker_cols
from src.model import load_model

st.set_page_config(page_title="Modelo", layout="wide")

st.title("Análisis del modelo entrenado")
st.markdown("---")

st.markdown(
    "Modelo de clasificación (logistic regression) para predecir arritmias ventriculares (AV)."
)

# ==============================
# CARGAR DATOS
# ==============================
@st.cache_data
def load_dataset():
    return load_data("data/Arritmias.csv")

df = load_dataset()

marker_cols = get_marker_cols(df)

X = df.drop(columns=["AV", "PACIENTES"])
y = df["AV"]

# ==============================
# MODELO
# ==============================
MODEL_PATH = "models/best_model.pkl"
model = load_model(MODEL_PATH)

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("Threshold")

threshold = st.sidebar.slider(
    "Threshold de clasificación",
    0.0, 1.0,
    value=load_threshold(),
    step=0.01
)

save_threshold(threshold)
# ==============================
# PREDICCIÓN
# ==============================
y_prob = model.predict_proba(X)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

# ==============================
# MÉTRICAS
# ==============================
auc = roc_auc_score(y, y_prob)
acc = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)

cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()

sens = tp / (tp + fn) if (tp + fn) > 0 else 0
spec = tn / (tn + fp) if (tn + fp) > 0 else 0

st.header("Resultados del modelo")

c1, c2, c3, c4,c5,c6 = st.columns(6)
with c1:
    st.metric("AUC", f"{auc:.3f}")
with c2:
    st.metric("Threshold", f"{threshold:.2f}")
with c3:
    st.metric("F1-score", f"{f1:.3f}")
with c4:
    st.metric("Accuracy", f"{acc:.3f}")
with c5:
    st.metric("Sensibilidad", f"{sens:.3f}")
with c6:
    st.metric("Especificidad", f"{spec:.3f}")

st.markdown("---")

c1, c2 = st.columns(2)
# ==============================
# ROC
# ==============================
with c1:
    st.subheader("Curva ROC")

    fpr, tpr, thresholds = roc_curve(y, y_prob)

    idx = np.argmin(np.abs(thresholds - threshold))
    fpr_point, tpr_point = fpr[idx], tpr[idx]

    fig, ax = plt.subplots()

    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", linewidth=1)
    ax.scatter(fpr_point, tpr_point, color="red", s=80)

    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

# ==============================
# MATRIZ DE CONFUSIÓN
# ==============================
with c2:
    st.subheader("Matriz de confusión")

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No AV", "AV"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    st.pyplot(fig)
    plt.close(fig)

# ==============================
# SHAP
# ==============================
st.subheader("Interpretabilidad (SHAP)")

import shap

model.fit(X, y)
X_transformed = model.named_steps["scaler"].transform(X)

if hasattr(model.named_steps["model"], "coef_"):
    explainer = shap.LinearExplainer(model.named_steps["model"], X_transformed)
else:
    explainer = shap.TreeExplainer(model.named_steps["model"])

shap_values = explainer(X_transformed)

fig = plt.figure()
shap.summary_plot(shap_values, X, feature_names=X.columns, show=False)

st.pyplot(fig)
plt.close(fig)