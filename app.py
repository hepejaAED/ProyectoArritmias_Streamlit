import streamlit as st
import pandas as pd
from src.data_loader import (
    load_data,
    get_marker_cols,
    get_group_summary,
    get_descriptive_stats,
    mann_whitney_test,
    split_by_group
)

st.set_page_config(
    page_title="Arritmias - Inicio",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Análisis de Arritmias Ventriculares")
st.markdown("---")

# Cargar datos
@st.cache_data # Para recargar la página de manera innecesaria
def load_dataset():
    return load_data("data/Arritmias.csv")

try:
    df = load_dataset()
except FileNotFoundError:
    st.error("El archivo data/Arritmias.csv no se encontró")
    st.stop()

# Definir colores
COLOR_AV0 = '#4C72B0'
COLOR_AV1 = '#DD8452'
LABEL_AV0 = 'AV = 0 (sin arritmia)'
LABEL_AV1 = 'AV = 1 (con arritmia)'

# Base de datos

# ===== TABLA COMPLETA =====
if st.checkbox("Ver tabla completa de datos", value=True):
    st.subheader("Datos completos")
    st.dataframe(df, use_container_width=True)


# ===== RESUMEN GENERAL =====
st.header("Resumen de la Base de Datos")

summary = get_group_summary(df)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("**Total de pacientes**", summary['Total'])

with col2:
    st.metric(
        "**Sin arritmia**",
        summary['Sin arritmia (AV=0)'],
        delta=f"{summary['Sin arritmia (AV=0)']/summary['Total']*100:.1f}%"
    )

with col3:
    st.metric(
        "**Con arritmia**",
        summary['Con arritmia (AV=1)'],
        delta=f"{summary['Con arritmia (AV=1)']/summary['Total']*100:.1f}%"
    )


st.markdown("---")

# ===== ESTADÍSTICOS DESCRIPTIVOS =====
st.header("📋 Estadísticos Descriptivos por Grupo")

marker_cols = get_marker_cols(df)

col1, col2 = st.columns([1, 3])

with col1:
    selected_marker = st.selectbox(
        "Selecciona un marcador:",
        marker_cols,
        help="Visualiza estadísticos para un marcador específico"
    )

with col2:
    st.empty()

# Mostrar tabla de estadísticos
desc_stats = get_descriptive_stats(df, marker_cols)
marker_stats = desc_stats.loc[selected_marker]

st.dataframe(
    marker_stats,
    use_container_width=True,
    column_config={
        'Media': st.column_config.NumberColumn(format="%.3f"),
        'Desv. Típica': st.column_config.NumberColumn(format="%.3f"),
    }
)

st.markdown("---")

# ===== TEST MANN-WHITNEY =====
st.header("🔬 Test Mann-Whitney U por Marcador")

mw_table, p_values = mann_whitney_test(df, marker_cols)

st.dataframe(
    mw_table,
    use_container_width=True,
    column_config={
        'p-valor': st.column_config.NumberColumn(format="%.5f"),
    }
)

st.markdown(
    """
    **Interpretación:**
    - `***` → p < 0.001 (muy significativo)
    - `**` → p < 0.01 (altamente significativo)
    - `*` → p < 0.05 (significativo)
    - `ns` → no significativo
    """
)

st.markdown("---")



# ===== INFORMACIÓN =====
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        ### 📚 Sobre este análisis
        
        Este proyecto analiza marcadores cardíacos post-infarto para predecir
        la presencia de arritmias ventriculares (AV).
        
        **Variables principales:**
        - `LV MASS (g)`: Masa del ventrículo izquierdo
        - `LVEF`: Fracción de eyección ventricular
        - `BZ + CORE`: Zona infartada (borde + núcleo)
        - `EDAD`: Edad del paciente
        """
    )

with col2:
    st.info(
        f"""
        **Estadísticas rápidas:**
        
        - Pacientes: {summary['Total']}
        - Marcadores: {len(marker_cols)}
        - Significativos: {sum(1 for p in p_values.values() if p < 0.05)}
        """
    )