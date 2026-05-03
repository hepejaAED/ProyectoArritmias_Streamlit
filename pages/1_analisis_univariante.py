import streamlit as st
import numpy as np

from src.data_loader import (
    load_data,
    get_marker_cols,
    get_descriptive_stats,
    mann_whitney_test,
    split_by_group,
    get_top_markers,
    get_significant_markers,
    normalize_minmax,
)
from src.analysis import (
    get_radar_data,
    get_demographic_groups,
    SHORT_NAMES,
)
from src.visualizations import (
    plot_violin_box_strip,
    plot_radar,
    plot_edad_sexo,
)

# ─── Configuración de página ───────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis Univariante",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Análisis Univariante")
st.markdown("Distribución individual de cada marcador cardíaco por grupo AV.")
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
mw_table, p_values = mann_whitney_test(df, marker_cols)
sig_markers        = get_significant_markers(p_values)
top5               = get_top_markers(p_values, n=5)


# ══════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 – Violin + Box + Strip
# ══════════════════════════════════════════════════════════════════════════
st.header("1 · Distribución por marcador (Violin + Box + Strip)")
st.markdown(
    "Selecciona un marcador para ver su distribución completa en ambos grupos. "
    "El p-valor corresponde al **test Mann-Whitney U** (dos colas)."
)

col_sel, col_info = st.columns([1, 2])

with col_sel:
    selected_marker = st.selectbox(
        "Marcador:",
        marker_cols,
        help="Cambia el marcador para actualizar la gráfica.",
    )

with col_info:
    p = p_values[selected_marker]

    def nivel(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    sig_map = {
        "***": ("🔴 Muy significativo",      "p < 0.001"),
        "**":  ("🟠 Altamente significativo", "p < 0.01"),
        "*":   ("🟡 Significativo",           "p < 0.05"),
        "ns":  ("⚪ No significativo",        "p ≥ 0.05"),
    }
    sig = nivel(p)
    label, _ = sig_map[sig]

    c1, c2, c3 = st.columns(3)
    c1.metric("p-valor", f"{p:.5f}")
    c2.metric("Significancia", sig)
    c3.metric("Nivel", label)

fig_violin = plot_violin_box_strip(df, selected_marker, p)
st.plotly_chart(fig_violin, use_container_width=True)

with st.expander("📋 Ver tabla completa Mann-Whitney por marcador"):
    st.dataframe(
        mw_table,
        use_container_width=True,
        column_config={"p-valor": st.column_config.NumberColumn(format="%.5f")},
    )
    st.markdown(
        "`***` p<0.001 · `**` p<0.01 · `*` p<0.05 · `ns` no significativo"
    )

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 – Estadísticos descriptivos
# ══════════════════════════════════════════════════════════════════════════
st.header("2 · Estadísticos Descriptivos por Grupo")

desc_stats = get_descriptive_stats(df, marker_cols)

col_sel2, _ = st.columns([1, 2])
with col_sel2:
    sel_desc = st.selectbox(
        "Marcador (descriptivos):",
        marker_cols,
        key="desc_marker",
    )

st.dataframe(
    desc_stats.loc[sel_desc],
    use_container_width=True,
    column_config={
        "Media":        st.column_config.NumberColumn(format="%.3f"),
        "Desv. Típica": st.column_config.NumberColumn(format="%.3f"),
    },
)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 – Variables demográficas (Edad y Sexo)
# ══════════════════════════════════════════════════════════════════════════
st.header("3 · Variables Demográficas: Edad y Sexo")
st.markdown(
    "Distribución de edad separada por sexo y estado de arritmia. "
    "Ten en cuenta el **gran desbalance** en el grupo de mujeres (n muy pequeño)."
)

demo_groups = get_demographic_groups(df)
fig_demo    = plot_edad_sexo(demo_groups)
st.plotly_chart(fig_demo, use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    st.info(
        f"**Hombres** · AV=0: {len(demo_groups['hombres_av0'])} pacientes · "
        f"AV=1: {len(demo_groups['hombres_av1'])} pacientes"
    )
with col_b:
    st.warning(
        f"**Mujeres** · AV=0: {len(demo_groups['mujeres_av0'])} pacientes · "
        f"AV=1: {len(demo_groups['mujeres_av1'])} pacientes  "
        "— muestra insuficiente para extraer conclusiones."
    )

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 – Radar chart (perfil medio normalizado)
# ══════════════════════════════════════════════════════════════════════════
st.header("4 · Perfil Medio Normalizado (Radar Chart)")
st.markdown(
    "Cada eje representa un marcador normalizado a escala 0–100 %. "
    "Permite comparar de un vistazo el **perfil medio** de cada grupo."
)

mean0, mean1, labels, angles, vals0, vals1 = get_radar_data(
    df, marker_cols, short_names=SHORT_NAMES
)
fig_radar = plot_radar(angles, vals0, vals1, labels)
st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 – Resumen de significancia
# ══════════════════════════════════════════════════════════════════════════
st.header("5 · Resumen de Significancia")

col1, col2, col3 = st.columns(3)
col1.metric("Total de marcadores",              len(marker_cols))
col2.metric("Significativos (p < 0.05)",        len(sig_markers))
col3.metric("No significativos",                len(marker_cols) - len(sig_markers))

c1, c2 = st.columns(2)
with c1:
    st.success("**Significativos (p < 0.05)**\n\n" + "\n".join(f"- {m}" for m in sig_markers))
with c2:
    ns = [m for m in marker_cols if m not in sig_markers]
    st.error("**No significativos**\n\n" + "\n".join(f"- {m}" for m in ns))