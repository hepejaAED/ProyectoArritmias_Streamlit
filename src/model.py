import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, 
    classification_report, precision_recall_curve
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import shap


class ArrimiaClassifier:
    """Wrapper para modelos de clasificación de arritmias"""
    
    def __init__(self, model_type='logistic', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.scaler = StandardScaler()
        self.cv_results = None
        self.feature_names = None
        
    def build_pipeline(self, model_type='logistic'):
        """Construye pipeline con SMOTE + modelo"""
        if model_type == 'logistic':
            return ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=self.random_state, k_neighbors=2)),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
            ])
        elif model_type == 'random_forest':
            return ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=self.random_state, k_neighbors=2)),
                ("model", RandomForestClassifier(n_estimators=100, random_state=self.random_state))
            ])
        else:
            raise ValueError("model_type debe ser 'logistic' o 'random_forest'")

    def get_grid_params(self, model_type='logistic'):
        """Retorna parámetros para GridSearchCV"""
        if model_type == 'logistic':
            return {
                "model__C": [0.001, 0.01, 0.1, 1],
                "model__penalty": ["l2"]
            }
        elif model_type == 'random_forest':
            return {
                "model__max_depth": [2, 3, 4],
                "model__min_samples_split": [3, 5]
            }
    
    def train(self, X, y, cv_folds=5):
        """Entrena modelos y selecciona el mejor"""
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        self.feature_names = X.columns.tolist()
        
        # Logistic Regression
        pipe_lr = self.build_pipeline('logistic')
        grid_lr = GridSearchCV(
            pipe_lr,
            self.get_grid_params('logistic'),
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )
        grid_lr.fit(X, y)
        
        # Random Forest
        pipe_rf = self.build_pipeline('random_forest')
        grid_rf = GridSearchCV(
            pipe_rf,
            self.get_grid_params('random_forest'),
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )
        grid_rf.fit(X, y)
        
        # Seleccionar mejor modelo
        if grid_lr.best_score_ > grid_rf.best_score_:
            self.best_model = grid_lr.best_estimator_
            self.best_params = grid_lr.best_params_
            self.model_type = 'logistic'
            best_score = grid_lr.best_score_
        else:
            self.best_model = grid_rf.best_estimator_
            self.best_params = grid_rf.best_params_
            self.model_type = 'random_forest'
            best_score = grid_rf.best_score_
        
        # Cross-validation detallada
        self.cv_results = cross_validate(
            self.best_model,
            X, y,
            cv=cv,
            scoring=["roc_auc", "accuracy", "precision", "recall", "f1"],
            return_train_score=True
        )
        
        return {
            'model_type': self.model_type,
            'best_params': self.best_params,
            'cv_auc': best_score,
            'cv_results': self.cv_results
        }
    
    def evaluate(self, X, y):
        """Evalúa el modelo en datos"""
        if self.best_model is None:
            raise ValueError("Modelo no entrenado. Llama a train() primero.")
        
        y_pred = self.best_model.predict(X)
        y_prob = self.best_model.predict_proba(X)[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        auc = roc_auc_score(y, y_prob)
        cm = confusion_matrix(y, y_pred)
        
        report = classification_report(
            y, y_pred,
            target_names=['Sin arritmia', 'Con arritmia'],
            output_dict=True
        )
        
        return {
            'y_pred': y_pred,
            'y_prob': y_prob,
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'confusion_matrix': cm,
            'classification_report': report
        }
    
    def get_shap_values(self, X):
        """Calcula SHAP values para interpretabilidad"""
        if self.best_model is None:
            raise ValueError("Modelo no entrenado. Llama a train() primero.")
        
        # Transformar datos con scaler del pipeline
        X_transformed = self.best_model.named_steps["scaler"].transform(X)
        
        model = self.best_model.named_steps["model"]
        
        if hasattr(model, "coef_"):  # Logistic Regression
            explainer = shap.LinearExplainer(model, X_transformed)
        else:  # Random Forest
            explainer = shap.TreeExplainer(model)
        
        shap_values = explainer(X_transformed)
        return shap_values, X_transformed
    
    def predict(self, X):
        """Predice para nuevas instancias"""
        if self.best_model is None:
            raise ValueError("Modelo no entrenado.")
        
        y_pred = self.best_model.predict(X)
        y_prob = self.best_model.predict_proba(X)[:, 1]
        
        return y_pred, y_prob
    
    def save(self, filepath):
        """Guarda el modelo entrenado"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.best_model, f)
    
    def load(self, filepath):
        """Carga un modelo previamente guardado"""
        with open(filepath, 'rb') as f:
            self.best_model = pickle.load(f)
    
    def get_feature_importance(self):
        """Retorna importancia de features (solo Random Forest)"""
        model = self.best_model.named_steps["model"]
        
        if not hasattr(model, "feature_importances_"):
            return None
        
        importances = model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)


def get_cv_summary(cv_results):
    """Resume resultados de cross-validation"""
    return {
        'Train AUC': cv_results['train_roc_auc'].mean(),
        'Test AUC': cv_results['test_roc_auc'].mean(),
        'Diferencia': cv_results['train_roc_auc'].mean() - cv_results['test_roc_auc'].mean(),
        'Train Accuracy': cv_results['train_accuracy'].mean(),
        'Test Accuracy': cv_results['test_accuracy'].mean(),
        'Train Precision': cv_results['train_precision'].mean(),
        'Test Precision': cv_results['test_precision'].mean(),
        'Train Recall': cv_results['train_recall'].mean(),
        'Test Recall': cv_results['test_recall'].mean(),
    }


def get_confusion_matrix_stats(cm):
    """Calcula estadísticos de matriz de confusión"""
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Sensitivity (Recall)': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0
    }


def get_prediction_explanation(y_prob, threshold=0.5):
    """Genera explicación para una predicción"""
    prob_arritmia = y_prob
    
    if prob_arritmia >= threshold:
        prediction = "CON ARRITMIA"
        confidence = prob_arritmia
    else:
        prediction = "SIN ARRITMIA"
        confidence = 1 - prob_arritmia
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'prob_arritmia': prob_arritmia
    }



def load_model(filepath):
    import joblib
    return joblib.load(filepath)

