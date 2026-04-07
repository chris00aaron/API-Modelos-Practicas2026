"""
train_from_csv.py
=================
Entrena el modelo Churn desde Churn_Modelling.csv (Kaggle),
lo sube a DagsHub y lo deja como campeón en producción.

Uso:
    cd API-Modelos-Practicas2026
    python train_from_csv.py
"""

import os
import io
import sys
import time
import joblib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Ruta al CSV ──────────────────────────────────────────────────────────────
CSV_PATH = Path(__file__).parent.parent / "Churn_Modelling.csv"

# ── Añadir src al path para importar módulos de la API ───────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ── Cargar .env ───────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")


def load_csv(path: Path) -> pd.DataFrame:
    logger.info(f"Cargando CSV: {path}")
    df = pd.read_csv(path)
    logger.info(f"Registros: {len(df)} | Columnas: {list(df.columns)}")
    # Eliminar columnas irrelevantes
    drop_cols = [c for c in ["RowNumber", "CustomerId", "Surname"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    logger.info(f"Churn rate: {df['Exited'].mean():.2%}  ({df['Exited'].sum()} positivos)")
    return df


def preprocess(df: pd.DataFrame):
    """Mismo preprocesamiento que auto_training_service.preprocess_training_data."""
    df = df.copy()
    epsilon = 1e-9

    # Feature engineering
    df["TenureByAge"]         = df["Tenure"] / (df["Age"] + epsilon)
    df["BalanceSalaryRatio"]  = df["Balance"] / (df["EstimatedSalary"] + epsilon)
    df["CreditScoreGivenAge"] = df["CreditScore"] / (df["Age"] + epsilon)

    # Gender encoding (normalizado a minúsculas)
    df["Gender"] = df["Gender"].astype(str).str.strip().str.lower().map(
        {"male": 1, "female": 0, "hombre": 1, "mujer": 0}
    ).fillna(0).astype(int)

    # Geography encoding
    geo_dummies = pd.get_dummies(df["Geography"], prefix="Geography")
    if "Geography_Germany" not in geo_dummies.columns:
        geo_dummies["Geography_Germany"] = 0
    if "Geography_Spain" not in geo_dummies.columns:
        geo_dummies["Geography_Spain"] = 0

    df = pd.concat([df, geo_dummies[["Geography_Germany", "Geography_Spain"]]], axis=1)
    df.drop(columns=["Geography"], inplace=True)

    y = df["Exited"].astype(int)
    feature_names = [
        "CreditScore", "Gender", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary",
        "TenureByAge", "BalanceSalaryRatio", "CreditScoreGivenAge",
        "Geography_Germany", "Geography_Spain",
    ]
    X = df[feature_names]
    return X, y, feature_names


def train(X, y):
    """SMOTE + GridSearchCV idéntico al auto_training_service."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    logger.info(f"Train split — neg: {neg}  pos: {pos}")

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_s, y_train)
    logger.info(f"Tras SMOTE: {len(X_res)} muestras")

    param_grid = {
        "n_estimators":  [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth":     [3, 5],
    }
    grid = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric="logloss"),
        param_grid, cv=3, scoring="roc_auc", n_jobs=1, verbose=1,
    )
    grid.fit(X_res, y_res)
    model = grid.best_estimator_
    logger.info(f"Mejores params: {grid.best_params_}")

    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "f1_score":  round(f1_score(y_test, y_pred),        4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0),    4),
        "auc_roc":   round(roc_auc_score(y_test, y_proba),  4),
    }
    logger.info(f"Métricas: {metrics}")

    return model, scaler, metrics, len(X_train), len(X_test)


def upload_to_dagshub(model, scaler, feature_names, metrics, train_samples, test_samples):
    """Sube el modelo a DagsHub y lo deja como campeón."""
    from fuga import dagshub_client
    from fuga.models_files.loader import cargar_modelos_desde_local

    version_tag = f"v_{int(time.time())}"
    logger.info(f"Version tag: {version_tag}")

    combo_pack = {
        "modelo_prediccion": model,
        "scaler":            scaler,
        "feature_names":     feature_names,
        "meta_info": {
            "version":       version_tag,
            "accuracy":      metrics["accuracy"],
            "f1_score":      metrics["f1_score"],
            "precision":     metrics["precision"],
            "recall":        metrics["recall"],
            "auc_roc":       metrics["auc_roc"],
            "train_samples": train_samples,
            "test_samples":  test_samples,
            "source":        "Churn_Modelling.csv (Kaggle)",
            "descripcion":   "Modelo CHURN XGBoost entrenado desde CSV Kaggle",
        },
    }

    # Guardar localmente primero
    cargar_modelos_desde_local(model, scaler, feature_names)
    logger.info("Modelos guardados localmente.")

    # Subir a DagsHub
    dagshub_client.init_dagshub_connection()

    buf = io.BytesIO()
    joblib.dump(combo_pack, buf)
    model_bytes = buf.getvalue()

    upload_ok = dagshub_client.upload_champion(model_bytes, version_tag)
    if upload_ok:
        integrity_ok = dagshub_client.verify_champion_integrity(version_tag)
        logger.info(f"Upload DagsHub: OK  |  Integridad: {integrity_ok}")
        dagshub_client.upload_individual_files(model, scaler, feature_names, version_tag)
    else:
        logger.warning("Upload a DagsHub FALLO. El modelo quedó guardado solo localmente.")

    return version_tag, upload_ok


def main():
    logger.info("=" * 60)
    logger.info("  ENTRENAMIENTO DESDE CSV — Churn_Modelling.csv")
    logger.info("=" * 60)

    if not CSV_PATH.exists():
        logger.error(f"CSV no encontrado en: {CSV_PATH}")
        sys.exit(1)

    # 1. Cargar
    df = load_csv(CSV_PATH)

    # 2. Preprocesar
    X, y, feature_names = preprocess(df)
    logger.info(f"Features: {feature_names}")

    # 3. Entrenar
    logger.info("Iniciando entrenamiento (GridSearchCV — puede tardar 3-5 min)...")
    model, scaler, metrics, train_samples, test_samples = train(X, y)

    # 4. Mostrar métricas
    logger.info("=" * 60)
    logger.info(f"  AUC-ROC  : {metrics['auc_roc']*100:.2f}%")
    logger.info(f"  Recall   : {metrics['recall']*100:.2f}%")
    logger.info(f"  Precision: {metrics['precision']*100:.2f}%")
    logger.info(f"  F1-Score : {metrics['f1_score']*100:.2f}%")
    logger.info(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
    logger.info("=" * 60)

    # 5. Subir a DagsHub
    version_tag, uploaded = upload_to_dagshub(
        model, scaler, feature_names, metrics, train_samples, test_samples
    )

    logger.info("=" * 60)
    if uploaded:
        logger.info(f"  LISTO. Modelo {version_tag} subido a DagsHub.")
        logger.info("  Reinicia la API Python para cargar el nuevo modelo.")
    else:
        logger.info(f"  Modelo {version_tag} guardado localmente.")
        logger.info("  Reinicia la API Python para cargar el nuevo modelo.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
