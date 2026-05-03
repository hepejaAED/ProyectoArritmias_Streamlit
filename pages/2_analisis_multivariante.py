import streamlit as st
import numpy as np

from src.data_loader import (
    load_data,
    get_marker_cols,
    mann_whitney_test,
    split_by_group,
    get_top_markers,
)
from src.analysis import (
    compute_correlation_matrices,
    top_correlation_differences,
    compute_mahalanobis_matrix,
    get_top4_mahal_pairs,
    SHORT_NAMES,
)
from src.visualizations import (
    plot_correlation_heatmap,
    plot_pairplot,
    plot_mahalanobis_heatmap,
    plot_mahalanobis_scatters,
    plot_scatter_kde,
    COLOR_AV0, COLOR_AV1,
)

# ─── Configuración de página ───────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis Multivariante",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔗 Análisis Multivariante")
st.markdown("Relaciones entre marcadores y separación conjunta entre grupos AV.")
st.markdown("---")


# ─── Carga de datos ────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    return load_data("data/Arritmias.csv")


try:
    df = load_dataset()
except FileNotFoundError:
    st.error("No se encontró el archivo `data/Arritmias.csv`.")
    st.stop()

marker_cols        = get_marker_cols(df)
df0, df1           = split_by_group(df)
_, p_values        = mann_whitney_test(df, marker_cols)
top5               = get_top_markers(p_values, n=5)


# ══════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 – Heatmap de correlaciones
# ══════════════════════════════════════════════════════════════════════════
st.header("1 · Matrices de Correlación de Pearson por Grupo")
st.markdown(
    "Correlación entre marcadores dentro de cada grupo AV. "
    "Solo se muestra el **triángulo inferior** para evitar redundancia."
)

corr0, corr1 = compute_correlation_matrices(df, marker_cols, short_names=SHORT_NAMES)

col_h1, col_h2 = st.columns(2)
with col_h1:
    fig_corr0 = plot_correlation_heatmap(
        corr0,
        title=f"AV = 0 · Sin arritmia (n={len(df0)})",
        title_color=COLOR_AV0,
    )
    st.plotly_chart(fig_corr0, use_container_width=True)

with col_h2:
    fig_corr1 = plot_correlation_heatmap(
        corr1,
        title=f"AV = 1 · Con arritmia (n={len(df1)})",
        title_color=COLOR_AV1,
    )
    st.plotly_chart(fig_corr1, use_container_width=True)

# Top diferencias de correlación
with st.expander("📋 Top 5 pares con mayor diferencia de correlación entre grupos"):
    top_diffs = top_correlation_differences(corr0, corr1, top_n=5)
    st.dataframe(
        top_diffs,
        use_container_width=True,
        column_config={"|Δr|": st.column_config.NumberColumn(format="%.3f")},
    )
    st.caption(
        "Se mide como |r_AV1 − r_AV0|. "
        "Los pares con Δr más alto son los que cambian más de estructura entre grupos."
    )

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 – Pairplot (scatter matrix) de los top N marcadores
# ══════════════════════════════════════════════════════════════════════════
st.header("2 · Scatter Matrix de Marcadores más Discriminantes")
st.markdown(
    "Scatter matrix de los N marcadores con menor p-valor (más discriminantes). "
    "Ayuda a detectar pares con buena separación visual entre grupos."
)

col_n, _ = st.columns([1, 3])
with col_n:
    n_top = st.slider(
        "Número de marcadores top:",
        min_value=3, max_value=len(marker_cols), value=5, step=1,
    )

top_n_markers = get_top_markers(p_values, n=n_top)
fig_pair      = plot_pairplot(df, top_n_markers)
st.plotly_chart(fig_pair, use_container_width=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 – Distancia de Mahalanobis
# ══════════════════════════════════════════════════════════════════════════
st.header("3 · Separación Bivariante (Distancia de Mahalanobis)")
st.markdown(
    "Cuantifica la separación entre los **centroides** de AV=0 y AV=1 para cada par "
    "de marcadores, teniendo en cuenta la covarianza de los datos."
)

D, short_labels, pair_scores = compute_mahalanobis_matrix(df, top5)
top4_pairs = get_top4_mahal_pairs(pair_scores)

col_m1, col_m2 = st.columns([1, 2])

with col_m1:
    st.subheader("Matriz de distancias")
    fig_mheat = plot_mahalanobis_heatmap(D, short_labels)
    st.plotly_chart(fig_mheat, use_container_width=True)

with col_m2:
    st.subheader("Top 4 pares con mayor separación")
    fig_mscatter = plot_mahalanobis_scatters(df, top4_pairs)
    st.plotly_chart(fig_mscatter, use_container_width=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 – Scatter + Contornos KDE interactivo
# ══════════════════════════════════════════════════════════════════════════
st.header("4 · Scatter + Contornos de Densidad")
st.markdown(
    "Exploración libre de cualquier par de marcadores. "
    "Las elipses representan contornos de densidad al **68 % y 95 %** para cada grupo."
)

col_x, col_y = st.columns(2)
with col_x:
    x_col = st.selectbox("Eje X:", marker_cols, index=marker_cols.index("LVEF") if "LVEF" in marker_cols else 0)
with col_y:
    default_y = next((i for i, c in enumerate(marker_cols) if "LV MASS" in c), 0)
    y_col = st.selectbox("Eje Y:", marker_cols, index=default_y)

fig_kde = plot_scatter_kde(df, x_col=x_col, y_col=y_col)
st.plotly_chart(fig_kde, use_container_width=True)