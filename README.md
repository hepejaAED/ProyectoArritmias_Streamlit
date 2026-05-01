# ProyectoArritmias_Streamlit


## Estructura de carpetas

```
proyecto-arritmias/
├── app.py                  # router principal (home)
├── pages/                  # páginas de la app
│   ├── 1_analisis_univariante.py
│   ├── 2_analisis_multivariante.py
│   ├── 3_modelo.py
│   └── 4_predictor.py
│
├── data/
│   └── Arritmias.csv
│
├── src/                    # lógica de la app
│   ├── __init__.py
│   ├── utils.py
│   ├── data_loader.py
│   ├── analysis.py
│   ├── model.py
│   └── visualizations.py
│
└── models/
│   ├── best_model.pkl
│   └── threshold.json

```

## Flujo de la app

```
Inicio
├── 📊 Análisis Exploratorio
│   ├── Estadísticas descriptivas
│   ├── Mann-Whitney test
│   └── Distribuciones (violin+box)
│
├── 🔍 Análisis Multivariante
│   ├── Heatmap correlaciones
│   ├── Pairplot
│   └── Mahalanobis
│
├── 🎯 Modelo
│   ├── Métricas
│   ├── ROC curve
│   ├── Confusion matrix
│   └── SHAP
│
└── 🔮 Predictor
    └── Formulario para nuevas predicciones
```