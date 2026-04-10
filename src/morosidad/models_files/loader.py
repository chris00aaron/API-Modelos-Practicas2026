
import logging
import os
import io
import joblib

import dagshub
import mlflow
import requests

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURACIÓN HARDCODEADA — DagsHub Morosidad
# ============================================================
DAGSHUB_REPO_OWNER = "notificacionesbankmind"
DAGSHUB_REPO_NAME = "Modelos_BankMind_2026"
DAGSHUB_MODEL_PATH = "modelos/morosidad/modelo.pkl"
DAGSHUB_TOKEN = "1022993058d503226b5e83a649a067c0c2ef2e73"

# Forzar el token en el entorno para que dagshub/mlflow lo encuentren
os.environ["DAGSHUB_USER_TOKEN"] = DAGSHUB_TOKEN

print(f"[MOROSIDAD] Token DagsHub hardcodeado (len={len(DAGSHUB_TOKEN)})")
print(f"[MOROSIDAD] Repo: {DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}")
print(f"[MOROSIDAD] Model path: {DAGSHUB_MODEL_PATH}")

# Variables globales (singleton)
_modelo = None
_explainer = None
_version = "v1.0"  # Valor por defecto
_dagshub_initialized = False


def _init_dagshub():
    """
    Inicializa la conexión a DagsHub/MLflow.
    Esta función es OPCIONAL para la descarga del modelo.
    Si falla, la descarga HTTP directa sigue funcionando.
    """
    global _dagshub_initialized
    if not _dagshub_initialized:
        try:
            dagshub.init(
                repo_owner=DAGSHUB_REPO_OWNER,
                repo_name=DAGSHUB_REPO_NAME,
                mlflow=True
            )
            _dagshub_initialized = True
            logger.info(f"Conectado a DagsHub: {DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}")
        except Exception as e:
            # dagshub.init() falló, pero la descarga HTTP directa puede funcionar.
            _dagshub_initialized = True
            logger.warning(
                f"dagshub.init() falló ({type(e).__name__}: {e}). "
                "Se usará descarga HTTP directa como fallback."
            )


def _descargar_modelo_dagshub():
    """
    Descarga el modelo desde DagsHub y lo carga directamente en memoria.
    No guarda ningún archivo en disco.
    
    Returns:
        El contenido del archivo .pkl cargado, o None si falla.
    """
    try:
        # Intentar inicializar dagshub (opcional, no bloquea si falla)
        _init_dagshub()
        
        auth = (DAGSHUB_TOKEN, "")
        branches = ["main", "master"]
        
        for branch in branches:
            raw_url = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}/raw/{branch}/{DAGSHUB_MODEL_PATH}"
            logger.info(f"Intentando descargar modelo desde: {raw_url}")
            
            try:
                response = requests.get(raw_url, auth=auth, timeout=60)
                if response.status_code == 200:
                    content_size = len(response.content)
                    logger.info(f"Modelo descargado desde DagsHub ({content_size:,} bytes), cargando en memoria...")
                    return joblib.load(io.BytesIO(response.content))
                else:
                    logger.warning(
                        f"Rama '{branch}' falló (Status: {response.status_code}). "
                        f"Respuesta: {response.text[:200]}"
                    )
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout descargando desde rama '{branch}' (60s)")
            except Exception as e_req:
                logger.warning(f"Error conectando a rama '{branch}': {e_req}")

        logger.error("No se pudo descargar el modelo de ninguna rama.")
        return None
        
    except Exception as e:
        logger.warning(f"Error general en descarga DagsHub: {e}")
        return None


def cargar_modelo(force_reload=False):
    """
    Carga el modelo de morosidad y el explainer SHAP desde DagsHub.
    El modelo se descarga y se mantiene únicamente en memoria.
    Se carga una sola vez (singleton pattern), a menos que force_reload=True.
    
    Args:
        force_reload (bool): Si es True, descarga de nuevo aunque ya exista en memoria.
    
    Returns:
        Tupla (modelo, explainer), o (None, None) si no se pudo descargar.
    """
    global _modelo, _explainer, _version
    
    if _modelo is not None and not force_reload:
        return _modelo, _explainer
    
    logger.info(f"Cargando modelo (force_reload={force_reload})...")
    
    # Descargar modelo desde DagsHub (solo fuente disponible)
    model_pack = _descargar_modelo_dagshub()
    
    if model_pack is None:
        logger.error("No se pudo cargar el modelo desde DagsHub")
        return None, None
    
    # Extraer componentes del paquete
    if isinstance(model_pack, dict):
        # Claves del script de empaquetado del usuario
        _modelo = model_pack.get('modelo_prediccion')
        _explainer = model_pack.get('shap_explainer')
        
        # Extraer metadatos si existen
        meta = model_pack.get('meta_info', {})
        if isinstance(meta, dict):
            _version = meta.get('version', "v1.0")
        
        logger.info(f"Modelo de morosidad cargado. Versión: {_version}")
        
        if _explainer:
            logger.info("SHAP Explainer cargado correctamente.")
        else:
            logger.warning("No se encontró 'shap_explainer' en el archivo .pkl")
    else:
        # Retrocompatibilidad: si es el modelo directamente
        _modelo = model_pack
        logger.info("Modelo de morosidad cargado (formato legacy)")
        logger.warning("El archivo no contiene explainer SHAP ni metadatos")
    
    return _modelo, _explainer


def obtener_modelo():
    """
    Obtiene el modelo ya cargado.
    Si no está cargado, intenta cargarlo.
    
    Returns:
        El modelo cargado.
    
    Raises:
        RuntimeError: Si el modelo no está disponible.
    """
    modelo, _ = cargar_modelo()
    if modelo is None:
        raise RuntimeError(
            "El modelo de morosidad no está disponible. "
            "Verifica la conexión a DagsHub."
        )
    return modelo


def obtener_explainer():
    """
    Obtiene el explainer SHAP ya cargado.
    Si no está cargado, intenta cargarlo.
    
    Returns:
        El explainer SHAP, o None si no está disponible.
    """
    _, explainer = cargar_modelo()
    return explainer


def obtener_version():
    """
    Obtiene la versión del modelo cargado.
    
    Returns:
        String con la versión (ej: "v1.0")
    """
    cargar_modelo()
    return _version
