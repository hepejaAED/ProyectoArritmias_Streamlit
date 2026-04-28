# ProyectoArritmias_Streamlit


## Estructura de carpetas

```
proyecto-arritmias/
├── app.py                 # main
├── data/
│   └── Arritmias.csv
├── src/
│   ├── data_loader.py     # cargar y preprocesar
│   ├── analysis.py        # análisis exploratorio
│   ├── model.py           # entrenamiento
│   └── visualizations.py  # gráficas reutilizables
├── models/
│   └── best_model.pkl     # modelo entrenado
└── requirements.txt
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