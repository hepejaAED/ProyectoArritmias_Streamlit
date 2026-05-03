"""
visualizations.py – Gráficas reutilizables con Plotly para la app de arritmias.
Todas las funciones devuelven figuras de Plotly (go.Figure) listas para
ser renderizadas con st.plotly_chart().
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .analysis import (
    COLOR_AV0, COLOR_AV1, LABEL_AV0, LABEL_AV1, PALETTE,
    SHORT_NAMES, normalize_minmax,
)


# ──────────────────────────────────────────────────────────
# Figura 1 – Violin + Box + Strip por marcador
# ──────────────────────────────────────────────────────────

def plot_violin_box_strip(
    df: pd.DataFrame,
    marker: str,
    p_value: float,
) -> go.Figure:
    """
    Violin + Box + Strip (jitter) para un único marcador, separado por grupo AV.

    Parameters
    ----------
    df      : DataFrame con columnas 'AV' y el marcador indicado.
    marker  : nombre de la columna a representar.
    p_value : p-valor del test Mann-Whitney para ese marcador.
    """
    df0 = df[df['AV'] == 0][marker].dropna()
    df1 = df[df['AV'] == 1][marker].dropna()

    sig = _nivel_significancia(p_value)
    color_sig = '#c0392b' if p_value < 0.05 else '#555555'

    fig = go.Figure()

    for vals, color, label, x_name in [
        (df0, COLOR_AV0, LABEL_AV0, f'AV=0 (n={len(df0)})'),
        (df1, COLOR_AV1, LABEL_AV1, f'AV=1 (n={len(df1)})'),
    ]:
        # Violin
        fig.add_trace(go.Violin(
            y=vals,
            name=label,
            legendgroup=label,
            box_visible=True,
            meanline_visible=False,
            fillcolor=color,
            opacity=0.45,
            line_color=color,
            x0=x_name,
            showlegend=True,
            points=False,
        ))
        # Strip (jitter)
        fig.add_trace(go.Box(
            y=vals,
            name=label,
            legendgroup=label,
            x0=x_name,
            boxpoints='all',
            jitter=0.35,
            pointpos=0,
            marker=dict(color=color, size=5, opacity=0.8,
                        line=dict(color='white', width=0.5)),
            fillcolor='rgba(0,0,0,0)',
            line_color='rgba(0,0,0,0)',
            showlegend=False,
        ))

    # Anotación p-valor
    y_max = max(df0.max(), df1.max())
    fig.add_annotation(
        text=f'p={p_value:.4f} {sig}',
        xref='paper', yref='y',
        x=0.5, y=y_max * 1.08,
        showarrow=False,
        font=dict(size=13, color=color_sig,
                  family='Arial Black' if p_value < 0.05 else 'Arial'),
    )

    fig.update_layout(
        title=dict(text=marker, font=dict(size=14, family='Arial Black')),
        yaxis_title=marker,
        violinmode='group',
        showlegend=True,
        height=420,
        margin=dict(t=60, b=40, l=50, r=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(orientation='h', y=-0.15),
    )
    return fig


def plot_violin_grid(
    df: pd.DataFrame,
    marker_cols,
    p_values: dict,
) -> list[go.Figure]:
    """Devuelve una lista de figuras (una por marcador) para mostrar en grid."""
    return [
        plot_violin_box_strip(df, col, p_values[col])
        for col in marker_cols
    ]


# ──────────────────────────────────────────────────────────
# Figura 2 – Heatmap de correlaciones
# ──────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    corr: pd.DataFrame,
    title: str,
    title_color: str,
) -> go.Figure:
    """
    Heatmap de correlación de Pearson (triángulo inferior).

    Parameters
    ----------
    corr        : DataFrame de correlaciones (cuadrado).
    title       : título de la gráfica.
    title_color : color hex del título.
    """
    n = len(corr)
    # Máscara: NaN en triángulo superior
    z = corr.values.copy().astype(float)
    for i in range(n):
        for j in range(i + 1, n):
            z[i, j] = np.nan

    labels = list(corr.columns)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        zmin=-1, zmax=1,
        colorscale='RdBu',
        reversescale=True,
        text=np.where(np.isnan(z), '', np.round(z, 2).astype(str)),
        texttemplate='%{text}',
        textfont=dict(size=9),
        showscale=True,
        colorbar=dict(title='Pearson r', len=0.75),
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=13, color=title_color, family='Arial Black'),
        ),
        xaxis=dict(tickangle=-40, tickfont=dict(size=9)),
        yaxis=dict(autorange='reversed', tickfont=dict(size=9)),
        height=480,
        margin=dict(t=60, b=80, l=80, r=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    return fig


# ──────────────────────────────────────────────────────────
# Figura 3 – Pairplot de los top 5 marcadores
# ──────────────────────────────────────────────────────────

def plot_pairplot(
    df: pd.DataFrame,
    top5_markers: list[str],
) -> go.Figure:
    """
    Scatter matrix (pairplot) de los 5 marcadores más discriminantes.
    La diagonal muestra histogramas por grupo.
    """
    df_plot = df[top5_markers + ['AV']].copy()
    df_plot['Grupo'] = df_plot['AV'].map({0: LABEL_AV0, 1: LABEL_AV1})

    fig = px.scatter_matrix(
        df_plot,
        dimensions=top5_markers,
        color='Grupo',
        color_discrete_map={LABEL_AV0: COLOR_AV0, LABEL_AV1: COLOR_AV1},
        opacity=0.6,
        title='Top 5 marcadores más discriminantes (Scatter Matrix)',
    )
    fig.update_traces(marker=dict(size=5), diagonal_visible=False)
    fig.update_layout(
        height=650,
        margin=dict(t=60, b=40, l=40, r=40),
    )
    return fig


# ──────────────────────────────────────────────────────────
# Figura 3b – Mahalanobis heatmap
# ──────────────────────────────────────────────────────────

def plot_mahalanobis_heatmap(
    D: np.ndarray,
    short_labels: list[str],
) -> go.Figure:
    """Heatmap de la matriz de distancias de Mahalanobis (triángulo inferior)."""
    n = len(short_labels)
    z = D.copy().astype(float)
    for i in range(n):
        for j in range(i + 1, n):
            z[i, j] = np.nan

    fig = go.Figure(go.Heatmap(
        z=z,
        x=short_labels,
        y=short_labels,
        colorscale='YlOrRd',
        text=np.where(np.isnan(z), '', np.round(z, 2).astype(str)),
        texttemplate='%{text}',
        textfont=dict(size=9),
        colorbar=dict(title='Distancia<br>Mahalanobis', len=0.75),
    ))

    fig.update_layout(
        title='Matriz de separación bivariante (Distancia de Mahalanobis)',
        xaxis=dict(tickangle=-40, tickfont=dict(size=9)),
        yaxis=dict(autorange='reversed', tickfont=dict(size=9)),
        height=400,
        margin=dict(t=60, b=80, l=80, r=20),
        paper_bgcolor='white',
    )
    return fig


def plot_mahalanobis_scatters(
    df: pd.DataFrame,
    top4_pairs: list[tuple],
) -> go.Figure:
    """
    4 scatter plots (2×2) de los pares con mayor distancia de Mahalanobis,
    con elipses de confianza al 95 %.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"{a.split('(')[0].strip()} vs {b.split('(')[0].strip()}  (d={d:.2f})"
            for (a, b), d in top4_pairs
        ],
    )

    show_legend = True
    for idx, ((col_a, col_b), _) in enumerate(top4_pairs):
        row, col = divmod(idx, 2)
        row += 1; col += 1

        for av, color, label in [(0, COLOR_AV0, LABEL_AV0), (1, COLOR_AV1, LABEL_AV1)]:
            sub = df[df['AV'] == av][[col_a, col_b]].dropna().values
            fig.add_trace(go.Scatter(
                x=sub[:, 0], y=sub[:, 1],
                mode='markers',
                marker=dict(color=color, size=6, opacity=0.55,
                            line=dict(color='white', width=0.3)),
                name=label,
                legendgroup=label,
                showlegend=show_legend,
            ), row=row, col=col)

            # Elipse de confianza al 95 % (analítica)
            if len(sub) >= 3:
                mu = sub.mean(axis=0)
                cov = np.cov(sub, rowvar=False)
                ellipse_x, ellipse_y = _confidence_ellipse(mu, cov, n_std=1.96)
                fig.add_trace(go.Scatter(
                    x=ellipse_x, y=ellipse_y,
                    mode='lines',
                    line=dict(color=color, width=1.8, dash='dash'),
                    showlegend=False,
                    legendgroup=label,
                ), row=row, col=col)

            # Centroide
            if len(sub) >= 1:
                mu = sub.mean(axis=0)
                fig.add_trace(go.Scatter(
                    x=[mu[0]], y=[mu[1]],
                    mode='markers',
                    marker=dict(symbol='x', size=12, color=color,
                                line=dict(width=2.5, color=color)),
                    showlegend=False,
                ), row=row, col=col)

        # Etiquetas de ejes
        short_a = col_a.split('(')[0].strip()
        short_b = col_b.split('(')[0].strip()
        fig.update_xaxes(title_text=short_a, row=row, col=col)
        fig.update_yaxes(title_text=short_b, row=row, col=col)

        show_legend = False  # Solo primera vez

    fig.update_layout(
        title='Top 4 pares con mayor distancia de Mahalanobis entre grupos AV',
        height=700,
        margin=dict(t=80, b=40, l=60, r=20),
        paper_bgcolor='white',
    )
    return fig


# ──────────────────────────────────────────────────────────
# Figura 4 – Variables demográficas (Edad y Sexo)
# ──────────────────────────────────────────────────────────

def plot_edad_sexo(demographic_groups: dict) -> go.Figure:
    """
    Histogramas de edad separados por sexo y estado de arritmia.

    Parameters
    ----------
    demographic_groups : salida de analysis.get_demographic_groups()
    """
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Hombres', 'Mujeres'],
                        shared_yaxes=True)

    n_bins = 8

    for col_idx, (key_av0, key_av1, sexo) in enumerate([
        ('hombres_av0', 'hombres_av1', 'Hombres'),
        ('mujeres_av0', 'mujeres_av1', 'Mujeres'),
    ], start=1):
        for vals, color, label, show in [
            (demographic_groups[key_av0], COLOR_AV0, LABEL_AV0, col_idx == 1),
            (demographic_groups[key_av1], COLOR_AV1, LABEL_AV1, col_idx == 1),
        ]:
            fig.add_trace(go.Histogram(
                x=vals,
                nbinsx=n_bins,
                name=label,
                legendgroup=label,
                marker_color=color,
                opacity=0.7,
                showlegend=show,
            ), row=1, col=col_idx)

    fig.update_layout(
        title='Distribución de edad por sexo y estado de arritmia',
        barmode='overlay',
        xaxis_title='Edad (años)',
        yaxis_title='Frecuencia',
        height=420,
        margin=dict(t=60, b=40),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    fig.update_xaxes(title_text='Edad (años)')
    return fig


# ──────────────────────────────────────────────────────────
# Figura 5 – Radar chart
# ──────────────────────────────────────────────────────────

def plot_radar(
    angles: np.ndarray,
    vals0: np.ndarray,
    vals1: np.ndarray,
    labels: list[str],
) -> go.Figure:
    """
    Radar chart (perfil medio normalizado) para los dos grupos AV.

    Parameters
    ----------
    angles : array de ángulos cerrados (rad) – salida de get_radar_data()
    vals0  : valores normalizados AV=0 (cerrados)
    vals1  : valores normalizados AV=1 (cerrados)
    labels : etiquetas de los marcadores
    """
    labels_closed = labels + [labels[0]]

    fig = go.Figure()

    for vals, color, label in [
        (vals0, COLOR_AV0, LABEL_AV0),
        (vals1, COLOR_AV1, LABEL_AV1),
    ]:
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=labels_closed,
            fill='toself',
            fillcolor=color,
            opacity=0.25,
            line=dict(color=color, width=2),
            name=label,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            tickvals=[20, 40, 60, 80, 100],
                            ticktext=['20%', '40%', '60%', '80%', '100%'],
                            tickfont=dict(size=8)),
            angularaxis=dict(tickfont=dict(size=10)),
        ),
        title='Perfil medio normalizado por grupo AV (Radar Chart)',
        showlegend=True,
        height=500,
        margin=dict(t=80, b=40, l=60, r=60),
    )
    return fig


# ──────────────────────────────────────────────────────────
# Figura 6 – Scatter + Contornos KDE (LVEF vs LV MASS)
# ──────────────────────────────────────────────────────────

def plot_scatter_kde(
    df: pd.DataFrame,
    x_col: str = 'LVEF',
    y_col: str = 'LV MASS (g)',
) -> go.Figure:
    """
    Scatter + elipses de densidad para los dos grupos AV.
    Aproxima los contornos KDE con elipses de confianza (1σ, 2σ).
    """
    fig = go.Figure()

    for av, color, label in [(0, COLOR_AV0, LABEL_AV0), (1, COLOR_AV1, LABEL_AV1)]:
        sub = df[df['AV'] == av][[x_col, y_col]].dropna().values

        fig.add_trace(go.Scatter(
            x=sub[:, 0], y=sub[:, 1],
            mode='markers',
            marker=dict(color=color, size=7, opacity=0.75,
                        line=dict(color='white', width=0.5)),
            name=label,
        ))

        if len(sub) >= 3:
            mu = sub.mean(axis=0)
            cov = np.cov(sub, rowvar=False)
            for n_std, opacity in [(1, 0.55), (2, 0.25)]:
                ex, ey = _confidence_ellipse(mu, cov, n_std=n_std)
                fig.add_trace(go.Scatter(
                    x=ex, y=ey,
                    mode='lines',
                    line=dict(color=color, width=1.5, dash='dash'),
                    opacity=opacity,
                    showlegend=False,
                    legendgroup=label,
                ))

    fig.update_layout(
        title=f"Scatter + Contornos de densidad: '{x_col}' vs '{y_col}'",
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=480,
        margin=dict(t=60, b=50, l=60, r=20),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#eee', gridwidth=0.5),
        yaxis=dict(showgrid=True, gridcolor='#eee', gridwidth=0.5),
    )
    return fig


# ──────────────────────────────────────────────────────────
# Helpers internos
# ──────────────────────────────────────────────────────────

def _nivel_significancia(p: float) -> str:
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'ns'


def _confidence_ellipse(
    mu: np.ndarray,
    cov: np.ndarray,
    n_std: float = 1.96,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera los puntos de una elipse de confianza paramétrica.

    Parameters
    ----------
    mu      : centro (2,)
    cov     : matriz de covarianza (2,2)
    n_std   : número de desviaciones estándar (1.96 → ≈95 %)
    """
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        t = np.linspace(0, 2 * np.pi, n_points)
        circle = np.array([np.cos(t), np.sin(t)])
        ellipse = eigvecs @ np.diag(n_std * np.sqrt(np.abs(eigvals))) @ circle
        return mu[0] + ellipse[0], mu[1] + ellipse[1]
    except Exception:
        return np.array([mu[0]]), np.array([mu[1]])
