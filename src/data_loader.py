import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

@pd.api.extensions.register_dataframe_accessor("arritmia")
class ArrimiaAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
 
    def preprocess(self):
        """Preprocesa el dataset: convierte , a . y formatea tipos"""
        df = self._obj.copy()
        cols = df.columns[1:-4]
        
        for col in cols:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(",", ".").astype(float)
        
        return df
    

def load_data(file_path) -> pd.DataFrame:
    """Carga el dataset desde un archivo CSV y lo preprocesa."""
    df = pd.read_csv(file_path)
    return df.arritmia.preprocess()

def get_marker_cols(df):
    """Retorna las columnas de marcadores cardíacos"""
    return df.columns[1:-3].tolist()
 
 
def get_feature_cols(df):
    """Retorna las columnas demográficas"""
    return df.columns[-3:-1].tolist()

def split_by_group(df, target_col='AV'):
    """Divide dataset en dos grupos según AV"""
    df0 = df[df[target_col] == 0]
    df1 = df[df[target_col] == 1]
    return df0, df1


def get_descriptive_stats(df, marker_cols):
    """Calcula estadísticos descriptivos por grupo AV"""
    df0, df1 = split_by_group(df)
    
    stats_rows = []
    for col in marker_cols:
        for av, grupo in [(0, df0), (1, df1)]:
            s = grupo[col].dropna()
            stats_rows.append({
                'Marcador': col,
                'Grupo': f'AV = {av}',
                'Media': round(s.mean(), 3),
                'Desv. Típica': round(s.std(), 3),
                'Mín': round(s.min(), 3),
                'Q1': round(s.quantile(0.25), 3),
                'Mediana': round(s.median(), 3),
                'Q3': round(s.quantile(0.75), 3),
                'Máx': round(s.max(), 3)
            })
    
    return pd.DataFrame(stats_rows).set_index(['Marcador', 'Grupo'])


def get_significance_level(p):
    """Retorna nivel de significancia basado en p-valor"""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


def mann_whitney_test(df, marker_cols):
    """Realiza test Mann-Whitney U para cada marcador"""
    df0, df1 = split_by_group(df)
    
    mw_results = []
    p_values = {}
    
    for col in marker_cols:
        stat, p = mannwhitneyu(
            df0[col].dropna(),
            df1[col].dropna(),
            alternative='two-sided'
        )
        p_values[col] = p
        
        significance = get_significance_level(p)
        mw_results.append({
            'Marcador': col,
            'Estadístico U': round(stat, 2),
            'p-valor': round(p, 5),
            'Significancia': significance
        })
    
    mw_df = pd.DataFrame(mw_results).sort_values('p-valor').set_index('Marcador')
    return mw_df, p_values
 

def get_top_markers(p_values, n=5): 
    # Poner que n sea un parámetro para elegir cuantos marcadores mostrar
    """Retorna los n marcadores más significativos"""
    sorted_markers = sorted(p_values.items(), key=lambda x: x[1])
    return [marker for marker, _ in sorted_markers[:n]]


def get_significant_markers(p_values, threshold=0.05):
    """Retorna marcadores significativos (p < threshold)"""
    return [col for col, p in p_values.items() if p < threshold]
 


def normalize_minmax(df, marker_cols, global_range=True):
    """Normaliza marcadores a escala 0-100 (Min-Max)"""
    df_norm = df[marker_cols].copy()
    
    for col in marker_cols:
        mn, mx = df[col].min(), df[col].max()
        df_norm[col] = (df[col] - mn) / (mx - mn) * 100
    
    return df_norm
 
 
def prepare_model_data(df, target_col='AV', exclude_cols=None):
    """Prepara X, y para modelado"""
    if exclude_cols is None:
        exclude_cols = ['AV', 'PACIENTES']
    
    X = df.drop(columns=exclude_cols)
    y = df[target_col]
    
    return X, y
 
 
def get_group_summary(df):
    """Retorna resumen de distribución de grupos"""
    counts = df['AV'].value_counts().sort_index()
    return {
        'Sin arritmia (AV=0)': counts.get(0, 0),
        'Con arritmia (AV=1)': counts.get(1, 0),
        'Total': len(df),
        'Ratio desbalance': counts.get(1, 0) / counts.get(0, 1)
    }
