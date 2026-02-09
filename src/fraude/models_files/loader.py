
import logging
import os
import io
import joblib

import dagshub
import mlflow
import requests

from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

logger = logging.getLogger(__name__)

# Configuración DagsHub
DAGSHUB_REPO_OWNER = "notificacionesbankmind"
DAGSHUB_REPO_NAME = "Modelos_BankMind_2026"
DAGSHUB_MODEL_PATH = "modelos/fraude/modelo_producción.pkl"

# Token de DagsHub (desde variable de entorno)
DAGSHUB_TOKEN = os.environ.get("DAGSHUB_USER_TOKEN")

# Configurar token en variables de entorno para dagshub/mlflow
if DAGSHUB_TOKEN:
    os.environ["DAGSHUB_USER_TOKEN"] = DAGSHUB_TOKEN
    print(f"[SETUP] Token DagsHub configurado (len={len(DAGSHUB_TOKEN)})")
else:
    print("[SETUP] ⚠️ DAGSHUB_USER_TOKEN no configurado. Revisa el archivo .env")

# Variables globales (singleton)
_model_pack = None
_version = "v1.0"  # Valor por defecto
_dagshub_initialized = False


def _verificar_token():
    """Verifica si hay token de DagsHub configurado."""
    token = os.environ.get("DAGSHUB_USER_TOKEN")
    if token:
        logger.debug(f"Token DagsHub encontrado (longitud: {len(token)} chars)")
        return True
    else:
        logger.warning("No se encontró DAGSHUB_USER_TOKEN en variables de entorno")
        logger.info("Configura el token en el archivo .env del proyecto")
        return False


def _init_dagshub():
    """Inicializa la conexión a DagsHub/MLflow."""
    global _dagshub_initialized
    if not _dagshub_initialized:
        logger.debug(f"Intentando conectar a DagsHub: {DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}")
        _verificar_token()
        
        try:
            dagshub.init(
                repo_owner=DAGSHUB_REPO_OWNER,
                repo_name=DAGSHUB_REPO_NAME,
                mlflow=True
            )
            _dagshub_initialized = True
            logger.info(f"Conectado a DagsHub: {DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}")
            logger.debug(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        except Exception as e:
            logger.error(f"Error conectando a DagsHub: {type(e).__name__}: {e}")


def _descargar_modelo_dagshub():
    """
    Descarga el modelo desde DagsHub y lo carga directamente en memoria.
    No guarda ningún archivo en disco.
    
    Returns:
        El contenido del archivo .pkl cargado, o None si falla.
    """
    try:
        _init_dagshub()
        
        auth = (DAGSHUB_TOKEN, "") if DAGSHUB_TOKEN else None
        branches = ["main", "master"]
        
        for branch in branches:
            raw_url = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}/raw/{branch}/{DAGSHUB_MODEL_PATH}"
            logger.info(f"Intentando descargar modelo desde: {raw_url}")
            
            try:
                response = requests.get(raw_url, auth=auth)
                if response.status_code == 200:
                    # Cargar directamente en memoria (sin escribir a disco)
                    logger.info("Modelo descargado desde DagsHub, cargando en memoria...")
                    return joblib.load(io.BytesIO(response.content))
                else:
                    logger.warning(f"Rama '{branch}' falló o no existe el archivo (Status: {response.status_code})")
            except Exception as e_req:
                logger.warning(f"Error conectando a rama '{branch}': {e_req}")

        logger.error("No se pudo descargar el modelo de ninguna rama.")
        return None
        
    except Exception as e:
        logger.warning(f"Error general en descarga DagsHub: {e}")
        return None


def cargar_modelo():
    """
    Carga el modelo de fraude desde DagsHub.
    El modelo se descarga y se mantiene únicamente en memoria.
    Se carga una sola vez (singleton pattern).
    
    Returns:
        Diccionario con los componentes del modelo, o None si no se pudo descargar.
    """
    global _model_pack, _version
    
    # ⚠️ CRITICAL: Verificar si ya está cargado ANTES de descargar
    if _model_pack is not None:
        logger.debug("Modelo de fraude ya está en memoria (singleton), reutilizando...")
        return _model_pack
    
    logger.info("Iniciando carga del modelo de fraude desde DagsHub...")
    
    # Descargar modelo desde DagsHub (solo fuente disponible)
    model_pack = _descargar_modelo_dagshub()
    
    if model_pack is None:
        logger.error("No se pudo cargar el modelo desde DagsHub")
        return None
    
    # Verificar estructura del modelo
    if isinstance(model_pack, dict):
        # Verificar que tenga los componentes esperados
        required_keys = ['scaler', 'model_xgb', 'model_if', 'encoders']
        missing_keys = [key for key in required_keys if key not in model_pack]
        
        if missing_keys:
            logger.warning(f"El modelo no contiene las claves esperadas: {missing_keys}")
        
        # Extraer metadatos si existen
        meta = model_pack.get('meta_info', {})
        if isinstance(meta, dict):
            _version = meta.get('version', "v1.0")
        
        # Guardar en cache global
        _model_pack = model_pack
        
        logger.info(f"Modelo de fraude cargado correctamente. Versión: {_version}")
        
        if model_pack.get('explainer'):
            logger.info("SHAP Explainer cargado correctamente.")
        else:
            logger.warning("No se encontró 'explainer' en el archivo .pkl")
    else:
        logger.error("El formato del modelo no es el esperado (debería ser un diccionario)")
        return None
    
    return _model_pack


def obtener_modelo_pack():
    """
    Obtiene el model pack completo ya cargado.
    Si no está cargado, intenta cargarlo.
    
    Returns:
        Diccionario con todos los componentes del modelo.
    
    Raises:
        RuntimeError: Si el modelo no está disponible.
    """
    modelo_pack = cargar_modelo()
    if modelo_pack is None:
        raise RuntimeError(
            "El modelo de fraude no está disponible. "
            "Verifica la conexión a DagsHub y las credenciales en el archivo .env"
        )
    return modelo_pack


def obtener_version():
    """
    Obtiene la versión del modelo cargado.
    
    Returns:
        String con la versión (ej: "v1.0")
    """
    cargar_modelo()
    return _version
