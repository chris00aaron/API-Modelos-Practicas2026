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

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURACIÓN HARDCODEADA — DagsHub Churn (Fuga)
# ============================================================
DAGSHUB_REPO_OWNER = "notificacionesbankmind"
DAGSHUB_REPO_NAME = "Modelos_BankMind_2026"
DAGSHUB_TOKEN = "1022993058d503226b5e83a649a067c0c2ef2e73"

# Ruta del combo-pack (nuevo formato)
DAGSHUB_COMBO_PATH = "modelos/fuga/modelos_produccion/churn_champion.pkl"

# Rutas legacy (3 archivos separados - fallback)
DAGSHUB_MODEL_PATH = "modelos/fuga/modelos_produccion/best_model_churn.pkl"
DAGSHUB_SCALER_PATH = "modelos/fuga/modelos_produccion/scaler.pkl"
DAGSHUB_FEATURES_PATH = "modelos/fuga/modelos_produccion/feature_names.pkl"

# Forzar el token en el entorno para que dagshub/mlflow lo encuentren
os.environ["DAGSHUB_USER_TOKEN"] = DAGSHUB_TOKEN

print(f"[CHURN] Token DagsHub hardcodeado (len={len(DAGSHUB_TOKEN)})")
print(f"[CHURN] Repo: {DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}")

# Variables globales (singleton)
_modelo = None
_scaler = None
_feature_names = None
_explainer = None
_version = "v1.0"
_dagshub_initialized = False

# Flag de actualización — True mientras se está descargando un nuevo modelo desde DagsHub.
# Durante este período, las predicciones en curso siguen usando el modelo anterior en memoria.
_actualizando = False


class ChurnModelActualizandoseError(Exception):
    """Se lanza cuando se intenta obtener el modelo mientras está siendo reemplazado."""
    pass


def _init_dagshub():
    """Inicializa la conexión a DagsHub/MLflow (opcional, no bloquea la descarga HTTP)."""
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
            _dagshub_initialized = True
            logger.warning(f"[CHURN] dagshub.init() falló ({e}). Descarga HTTP directa como fallback.")


def _descargar_archivo_dagshub(file_path: str):
    """
    Descarga un archivo .pkl desde DagsHub y lo carga directamente en memoria.

    Returns:
        El contenido del archivo .pkl cargado, o None si falla.
    """
    try:
        _init_dagshub()

        auth = (DAGSHUB_TOKEN, "")
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


def _cargar_local():
    """
    Carga los modelos desde los archivos .pkl locales en models_files/.
    Último recurso cuando DagsHub no está disponible.
    """
    global _modelo, _scaler, _feature_names

    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_path = os.path.join(base_dir, "best_model_churn.pkl")
    local_scaler_path = os.path.join(base_dir, "scaler.pkl")
    local_features_path = os.path.join(base_dir, "feature_names.pkl")

    try:
        if os.path.exists(local_model_path):
            _modelo = joblib.load(local_model_path)
            logger.info("[CHURN] ✅ Modelo cargado desde archivo local")
        else:
            logger.error(f"[CHURN] ❌ Archivo local no encontrado: {local_model_path}")

        if os.path.exists(local_scaler_path):
            _scaler = joblib.load(local_scaler_path)
            logger.info("[CHURN] ✅ Scaler cargado desde archivo local")

        if os.path.exists(local_features_path):
            _feature_names = joblib.load(local_features_path)
            logger.info(f"[CHURN] ✅ Features cargados desde archivo local ({len(_feature_names)} features)")

    except Exception as e:
        logger.error(f"[CHURN] ❌ Error cargando archivos locales: {e}")


def cargar_modelos():
    """
    Carga los modelos de Churn.
    Orden de prioridad:
      1. Combo-pack desde DagsHub (nuevo formato)
      2. Legacy 3 archivos desde DagsHub
      3. Archivos .pkl locales en models_files/

    Returns:
        Tupla (modelo, scaler, feature_names)
    """
    global _modelo, _scaler, _feature_names

    if _modelo is not None:
        return _modelo, _scaler, _feature_names

    print("[CHURN] Descargando modelos desde DagsHub...")

    # Prioridad 1: Combo-pack
    if _cargar_combo_pack():
        return _modelo, _scaler, _feature_names

    # Prioridad 2: Legacy (3 archivos separados en DagsHub)
    logger.info("[CHURN] Combo-pack no disponible, intentando legacy (3 archivos)...")
    _cargar_legacy()

    if _modelo:
        print("[CHURN] ✅ Modelo cargado correctamente (legacy DagsHub)")
    else:
        # Prioridad 3: Archivos locales
        print("[CHURN] ⚠️ DagsHub no disponible, cargando desde archivos locales...")
        _cargar_local()

    if _modelo:
        print("[CHURN] ✅ Modelo listo")
    else:
        print("[CHURN] ❌ Error cargando modelo")

    if _scaler:
        print("[CHURN] ✅ Scaler listo")

    if _feature_names:
        print(f"[CHURN] ✅ Feature names listos ({len(_feature_names)} features)")

    return _modelo, _scaler, _feature_names


def obtener_modelo():
    """Obtiene el modelo de predicción."""
    if _actualizando:
        raise ChurnModelActualizandoseError(
            "El modelo CHURN está siendo actualizado desde DagsHub. "
            "Por favor, intente nuevamente en unos momentos."
        )
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


def obtener_explainer():
    """
    Obtiene (o crea) el SHAP TreeExplainer para el modelo de churn.
    Se crea una sola vez y se cachea en memoria.

    Returns:
        shap.TreeExplainer, o None si SHAP no está disponible.
    """
    global _explainer
    if _explainer is not None:
        return _explainer

    modelo = obtener_modelo() if _modelo is not None else None
    if modelo is None:
        return None

    try:
        import shap
        _explainer = shap.TreeExplainer(modelo)
        logger.info("[CHURN] ✅ SHAP TreeExplainer creado correctamente")
    except Exception as e:
        logger.warning(f"[CHURN] ⚠️ No se pudo crear SHAP TreeExplainer: {e}")
        _explainer = None

    return _explainer


def recargar_modelos():
    """
    Invalida los modelos en caché para forzar una recarga.
    Debe llamarse después de un entrenamiento exitoso.
    """
    global _modelo, _scaler, _feature_names, _explainer
    _modelo = None
    _scaler = None
    _feature_names = None
    _explainer = None
    logger.info("[CHURN] Caché de modelos invalidada. Se recargarán en la próxima predicción.")


def info_modelo() -> dict:
    """
    Devuelve el estado actual del modelo en memoria.
    Equivalente al `info_modelo` property de AtmModelProvider.

    Returns:
        dict con version, status ("ready" | "updating" | "not_loaded"),
        has_explainer, has_scaler y feature_count.
    """
    if _actualizando:
        status = "updating"
    elif _modelo is not None:
        status = "ready"
    else:
        status = "not_loaded"

    return {
        "version": _version,
        "status": status,
        "has_explainer": _explainer is not None,
        "has_scaler": _scaler is not None,
        "feature_count": len(_feature_names) if _feature_names else 0,
    }


def recargar_desde_dagshub():
    """
    Descarga el modelo campeón más reciente desde DagsHub y reemplaza el modelo
    en memoria sin necesidad de re-entrenamiento.

    - Activa el flag `_actualizando` durante la descarga (las predicciones en curso
      siguen usando el objeto de modelo ya referenciado en ChurnService.model).
    - Al finalizar, actualiza también el singleton ChurnService para que las
      nuevas predicciones usen el modelo recién descargado.
    - Equivalente a `AtmModelProvider.recargar_modelo()`.
    """
    global _actualizando, _modelo, _scaler, _feature_names, _explainer, _version

    logger.info("[CHURN] Iniciando recarga de modelo desde DagsHub (hot-reload)...")
    _actualizando = True

    try:
        # Invalidar caché para forzar descarga
        _modelo = None
        _scaler = None
        _feature_names = None
        _explainer = None

        # Intentar combo-pack primero, luego legacy
        if not _cargar_combo_pack():
            logger.info("[CHURN] Hot-reload: combo-pack no disponible, intentando legacy...")
            _cargar_legacy()

        if _modelo is None:
            logger.error("[CHURN] Hot-reload fallido: no se pudo descargar el modelo.")
            return

        # Reconstruir el SHAP explainer con el nuevo modelo
        obtener_explainer()

        logger.info(f"[CHURN] Hot-reload completado. Versión activa: {_version}")

        # Actualizar el singleton ChurnService para que use el nuevo modelo
        try:
            from fuga.service.churn_service import churn_service as _cs
            _cs.model         = _modelo
            _cs.scaler        = _scaler
            _cs.feature_names = _feature_names
            _cs.explainer     = _explainer
            _cs.model_version = _version
            logger.info(f"[CHURN] ChurnService actualizado en caliente → versión {_version}.")
        except Exception as e_cs:
            logger.warning(f"[CHURN] No se pudo actualizar ChurnService en caliente: {e_cs}")

    except Exception as e:
        logger.error(f"[CHURN] Error durante hot-reload desde DagsHub: {e}")
    finally:
        _actualizando = False


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

