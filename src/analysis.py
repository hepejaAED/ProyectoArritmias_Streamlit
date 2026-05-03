"""
analysis.py – Lógica de análisis exploratorio de arritmias.
Funciones puras (sin imports de streamlit) que operan sobre DataFrames.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────
# Constantes de estilo compartidas
# ─────────────────────────────────────────────────
COLOR_AV0 = '#4C72B0'
COLOR_AV1 = '#DD8452'
PALETTE = {0: COLOR_AV0, 1: COLOR_AV1}
LABEL_AV0 = 'AV = 0 (sin arritmia)'
LABEL_AV1 = 'AV = 1 (con arritmia)'

SHORT_NAMES = {
    'LV MASS (g)': 'LV Mass(g)',
    'BZ + CORE (g)': 'BZ+Core(g)',
    'BZ + CORE (%)': 'BZ+Core(%)',
    'BZ (g)': 'BZ(g)',
    'BZ (%)': 'BZ(%)',
    'CORE (g)': 'Core(g)',
    'CORE (%)': 'Core(%)',
    'CHANNEL MASS (g)': 'Ch.Mass(g)',
    'LVEF': 'LVEF',
}


# ─────────────────────────────────────────────────
# Análisis univariante
# ─────────────────────────────────────────────────

def get_top_n_markers(p_values: dict, n: int = 5) -> list[str]:
    """Devuelve los n marcadores con menor p-valor."""
    return sorted(p_values, key=p_values.get)[:n]


def normalize_minmax(df: pd.DataFrame, marker_cols) -> pd.DataFrame:
    """
    Normalización Min-Max global (0–100 %) sobre las columnas indicadas.
    Devuelve un nuevo DataFrame con las mismas columnas escaladas.
    """
    df_norm = df[marker_cols].copy()
    for col in marker_cols:
        mn, mx = df[col].min(), df[col].max()
        df_norm[col] = (df[col] - mn) / (mx - mn) * 100 if mx != mn else 0.0
    return df_norm


# ─────────────────────────────────────────────────
# Análisis multivariante
# ─────────────────────────────────────────────────

def compute_correlation_matrices(
    df: pd.DataFrame,
    marker_cols,
    short_names: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula las matrices de correlación de Pearson para cada grupo AV.

    Returns
    -------
    corr0 : DataFrame – correlación grupo AV = 0
    corr1 : DataFrame – correlación grupo AV = 1
    """
    df0 = df[df['AV'] == 0]
    df1 = df[df['AV'] == 1]

    rename = short_names or {}

    corr0 = df0[marker_cols].rename(columns=rename).corr(method='pearson')
    corr1 = df1[marker_cols].rename(columns=rename).corr(method='pearson')
    return corr0, corr1


def top_correlation_differences(
    corr0: pd.DataFrame,
    corr1: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Devuelve los top_n pares de variables con mayor diferencia |r_AV1 − r_AV0|.
    """
    diff = (corr1 - corr0).abs()
    mask_lower = np.triu(np.ones_like(diff, dtype=bool), k=0)
    diff_masked = diff.where(~mask_lower)

    result = (
        diff_masked.stack()
        .reset_index()
        .rename(columns={'level_0': 'Var A', 'level_1': 'Var B', 0: '|Δr|'})
        .sort_values('|Δr|', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return result


def compute_mahalanobis_matrix(
    df: pd.DataFrame,
    top_markers: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Calcula la distancia de Mahalanobis entre los centroides de AV=0 y AV=1
    para cada par de marcadores de top_markers.

    Returns
    -------
    D      : ndarray (n×n) – matriz simétrica de distancias
    short  : list[str]     – etiquetas cortas (nombre antes del paréntesis)
    """
    df0 = df[df['AV'] == 0]
    df1 = df[df['AV'] == 1]
    n = len(top_markers)

    def _mahal_pair(col_a: str, col_b: str) -> float:
        X0 = df0[[col_a, col_b]].dropna().values
        X1 = df1[[col_a, col_b]].dropna().values
        mu0, mu1 = X0.mean(axis=0), X1.mean(axis=0)
        n0, n1 = len(X0), len(X1)
        cov_pooled = (
            (n0 - 1) * np.cov(X0, rowvar=False)
            + (n1 - 1) * np.cov(X1, rowvar=False)
        ) / (n0 + n1 - 2)
        try:
            return mahalanobis(mu0, mu1, np.linalg.inv(cov_pooled))
        except np.linalg.LinAlgError:
            return np.nan

    pairs = list(combinations(top_markers, 2))
    pair_scores = {(a, b): _mahal_pair(a, b) for a, b in pairs}

    D = np.zeros((n, n))
    for (a, b), d in pair_scores.items():
        i, j = top_markers.index(a), top_markers.index(b)
        D[i, j] = d
        D[j, i] = d

    short = [c.split('(')[0].strip() for c in top_markers]
    return D, short, pair_scores


def get_top4_mahal_pairs(pair_scores: dict) -> list[tuple]:
    """Devuelve los 4 pares con mayor distancia de Mahalanobis."""
    sorted_pairs = sorted(pair_scores.items(), key=lambda x: -x[1])
    return sorted_pairs[:4]


# ─────────────────────────────────────────────────
# Variables demográficas
# ─────────────────────────────────────────────────

def get_demographic_groups(df: pd.DataFrame) -> dict:
    """
    Separa el DataFrame en los 4 subgrupos demográficos: 
    hombres/mujeres × arritmia/no arritmia.

    Returns
    -------
    dict con claves: 'hombres_av0', 'hombres_av1', 'mujeres_av0', 'mujeres_av1'
    Cada valor es una Series con las edades del subgrupo.
    """
    return {
        'hombres_av0': df[(df['SEXO'] == 1) & (df['AV'] == 0)]['EDAD'],
        'hombres_av1': df[(df['SEXO'] == 1) & (df['AV'] == 1)]['EDAD'],
        'mujeres_av0': df[(df['SEXO'] == 2) & (df['AV'] == 0)]['EDAD'],
        'mujeres_av1': df[(df['SEXO'] == 2) & (df['AV'] == 1)]['EDAD'],
    }


# ─────────────────────────────────────────────────
# Radar chart
# ─────────────────────────────────────────────────

def get_radar_data(
    df: pd.DataFrame,
    marker_cols,
    short_names: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    """
    Prepara los datos para el radar chart (perfil medio normalizado).

    Returns
    -------
    mean0   : valores medios normalizados para AV=0
    mean1   : valores medios normalizados para AV=1
    labels  : etiquetas de los marcadores
    angles  : ángulos del radar (cerrado)
    vals0   : valores cerrados para AV=0
    vals1   : valores cerrados para AV=1
    """
    df_norm = normalize_minmax(df, marker_cols)
    mean0 = df_norm[df['AV'] == 0].mean()
    mean1 = df_norm[df['AV'] == 1].mean()

    rename = short_names or {}
    labels = [rename.get(c, c) for c in marker_cols]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    vals0 = mean0.tolist() + [mean0.iloc[0]]
    vals1 = mean1.tolist() + [mean1.iloc[0]]

    return mean0, mean1, labels, np.array(angles_closed), np.array(vals0), np.array(vals1)


# ─────────────────────────────────────────────────
# PCA (análisis adicional)
# ─────────────────────────────────────────────────

def compute_pca(
    df: pd.DataFrame,
    marker_cols,
    n_components: int = 2,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Aplica PCA estandarizado sobre marker_cols.

    Returns
    -------
    df_pca          : DataFrame con columnas PC1, PC2, … y AV
    explained_var   : array con la varianza explicada por cada componente
    """
    X = df[marker_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(X_scaled)

    col_names = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(components, columns=col_names, index=X.index)
    df_pca['AV'] = df.loc[X.index, 'AV'].values

    return df_pca, pca.explained_variance_ratio_
