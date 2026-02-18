# src/fuga/models_files/loader.py
"""
Cargador de modelos de Churn desde DagsHub.
Soporta:
  - Combo-pack (nuevo): un solo .pkl con modelo, scaler, features y metadata
  - Legacy (antiguo): 3 archivos separados (model, scaler, feature_names)
"""

import logging
import os
import io
import joblib
import requests

import dagshub
import mlflow

from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURACIÓN DE DAGSHUB - MODELO DE CHURN
# ============================================================
DAGSHUB_REPO_OWNER = "notificacionesbankmind"
DAGSHUB_REPO_NAME = "Modelos_BankMind_2026"

# Ruta del combo-pack (nuevo formato)
DAGSHUB_COMBO_PATH = "modelos/fuga/modelos_produccion/churn_champion.pkl"

# Rutas legacy (3 archivos separados - fallback)
DAGSHUB_MODEL_PATH = "modelos/fuga/modelos_produccion/best_model_churn.pkl"
DAGSHUB_SCALER_PATH = "modelos/fuga/modelos_produccion/scaler.pkl"
DAGSHUB_FEATURES_PATH = "modelos/fuga/modelos_produccion/feature_names.pkl"

# Token de DagsHub (desde variable de entorno)
DAGSHUB_TOKEN = os.environ.get("DAGSHUB_USER_TOKEN")

if DAGSHUB_TOKEN:
    os.environ["DAGSHUB_USER_TOKEN"] = DAGSHUB_TOKEN
    print(f"[CHURN] Token DagsHub configurado (len={len(DAGSHUB_TOKEN)})")
else:
    print("[CHURN] ⚠️ DAGSHUB_USER_TOKEN no configurado. Revisa el archivo .env")

# Variables globales (singleton)
_modelo = None
_scaler = None
_feature_names = None
_version = "v1.0"
_dagshub_initialized = False


def _init_dagshub():
    """Inicializa la conexión a DagsHub/MLflow."""
    global _dagshub_initialized
    if not _dagshub_initialized:
        try:
            dagshub.init(
                repo_owner=DAGSHUB_REPO_OWNER,
                repo_name=DAGSHUB_REPO_NAME,
                mlflow=True
            )
            _dagshub_initialized = True
            logger.info(f"[CHURN] Conectado a DagsHub: {DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}")
        except Exception as e:
            logger.error(f"[CHURN] Error conectando a DagsHub: {e}")


def _descargar_archivo_dagshub(file_path: str):
    """
    Descarga un archivo .pkl desde DagsHub y lo carga directamente en memoria.

    Returns:
        El contenido del archivo .pkl cargado, o None si falla.
    """
    try:
        _init_dagshub()

        auth = (DAGSHUB_TOKEN, "") if DAGSHUB_TOKEN else None
        branches = ["main", "master"]

        for branch in branches:
            raw_url = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}/raw/{branch}/{file_path}"
            logger.info(f"[CHURN] Descargando: {raw_url}")

            try:
                response = requests.get(raw_url, auth=auth, timeout=30)
                if response.status_code == 200:
                    logger.info(f"[CHURN] ✅ Descargado: {file_path}")
                    return joblib.load(io.BytesIO(response.content))
                else:
                    logger.warning(f"[CHURN] Rama '{branch}' falló (Status: {response.status_code})")
            except Exception as e_req:
                logger.warning(f"[CHURN] Error en rama '{branch}': {e_req}")

        logger.error(f"[CHURN] ❌ No se pudo descargar: {file_path}")
        return None

    except Exception as e:
        logger.error(f"[CHURN] Error general en descarga: {e}")
        return None


def _cargar_combo_pack():
    """
    Intenta descargar y cargar el combo-pack (nuevo formato).

    Returns:
        True si se cargó exitosamente, False en caso contrario.
    """
    global _modelo, _scaler, _feature_names, _version

    combo = _descargar_archivo_dagshub(DAGSHUB_COMBO_PATH)
    if combo is None:
        return False

    if not isinstance(combo, dict):
        logger.warning("[CHURN] Combo-pack no es dict, ignorando")
        return False

    _modelo = combo.get('modelo_prediccion')
    _scaler = combo.get('scaler')
    _feature_names = combo.get('feature_names')

    meta = combo.get('meta_info', {})
    _version = meta.get('version', _version)

    if _modelo is not None:
        logger.info(
            f"[CHURN] ✅ Combo-pack cargado (versión: {_version}, "
            f"modelo: OK, scaler: {'OK' if _scaler else 'N/A'}, "
            f"features: {len(_feature_names) if _feature_names else 'N/A'})"
        )
        return True

    logger.warning("[CHURN] Combo-pack descargado pero sin modelo_prediccion")
    return False


def _cargar_legacy():
    """Carga los 3 archivos separados (fallback para modelos viejos)."""
    global _modelo, _scaler, _feature_names

    _modelo = _descargar_archivo_dagshub(DAGSHUB_MODEL_PATH)
    _scaler = _descargar_archivo_dagshub(DAGSHUB_SCALER_PATH)
    _feature_names = _descargar_archivo_dagshub(DAGSHUB_FEATURES_PATH)


def cargar_modelos():
    """
    Carga los modelos de Churn desde DagsHub.
    Intenta primero el combo-pack (nuevo formato), si falla usa legacy (3 archivos).

    Returns:
        Tupla (modelo, scaler, feature_names)
    """
    global _modelo, _scaler, _feature_names

    if _modelo is not None:
        return _modelo, _scaler, _feature_names

    print("[CHURN] Descargando modelos desde DagsHub...")

    # Intentar combo-pack primero
    if _cargar_combo_pack():
        return _modelo, _scaler, _feature_names

    # Fallback: 3 archivos separados
    logger.info("[CHURN] Combo-pack no disponible, intentando legacy (3 archivos)...")
    _cargar_legacy()

    if _modelo:
        print("[CHURN] ✅ Modelo cargado correctamente (legacy)")
    else:
        print("[CHURN] ❌ Error cargando modelo")

    if _scaler:
        print("[CHURN] ✅ Scaler cargado correctamente")

    if _feature_names:
        print(f"[CHURN] ✅ Feature names cargados ({len(_feature_names)} features)")

    return _modelo, _scaler, _feature_names


def obtener_modelo():
    """Obtiene el modelo de predicción."""
    modelo, _, _ = cargar_modelos()
    if modelo is None:
        raise RuntimeError(
            "El modelo de Churn no está disponible. "
            "Verifica la conexión a DagsHub y las credenciales."
        )
    return modelo


def obtener_scaler():
    """Obtiene el scaler para preprocesamiento."""
    _, scaler, _ = cargar_modelos()
    return scaler


def obtener_feature_names():
    """Obtiene la lista de nombres de features."""
    _, _, feature_names = cargar_modelos()
    return feature_names


def obtener_version():
    """Obtiene la versión del modelo."""
    return _version


def recargar_modelos():
    """
    Invalida los modelos en caché para forzar una recarga.
    Debe llamarse después de un entrenamiento exitoso.
    """
    global _modelo, _scaler, _feature_names
    _modelo = None
    _scaler = None
    _feature_names = None
    logger.info("[CHURN] Caché de modelos invalidada. Se recargarán en la próxima predicción.")


def cargar_modelos_desde_local(model, scaler, feature_names):
    """
    Carga los modelos directamente desde objetos en memoria (post-entrenamiento).
    Útil para hot-reload inmediato sin esperar descarga de DagsHub.

    Args:
        model: Objeto del modelo entrenado (XGBClassifier, etc.)
        scaler: Objeto StandardScaler
        feature_names: Lista de nombres de features
    """
    global _modelo, _scaler, _feature_names
    try:
        _modelo = model
        _scaler = scaler
        _feature_names = feature_names
        logger.info("[CHURN] ✅ Modelos recargados en memoria post-entrenamiento.")
        return True
    except Exception as e:
        logger.error(f"[CHURN] ❌ Error recargando modelos: {e}")
        return False

