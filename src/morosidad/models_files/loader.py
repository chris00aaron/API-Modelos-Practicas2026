
import logging

logger = logging.getLogger(__name__)

# src/morosidad/models_files/loader.py
import os
import joblib

import dagshub
import mlflow

import requests
import tempfile

# Configuración DagsHub
DAGSHUB_REPO_OWNER = "notificacionesbankmind"
DAGSHUB_REPO_NAME = "Modelos_BankMind_2026"
DAGSHUB_MODEL_PATH = "modelos/morosidad/modelo.pkl"

# Token de DagsHub (⚠️ REEMPLAZAR con tu token real)
# Obtén tu token en: https://dagshub.com/user/settings/tokens
DAGSHUB_TOKEN = "1022993058d503226b5e83a649a067c0c2ef2e73"  # <-- PEGA TU TOKEN AQUÍ

# Configurar token en variables de entorno automáticamente
if DAGSHUB_TOKEN:
    os.environ["DAGSHUB_USER_TOKEN"] = DAGSHUB_TOKEN
    print(f"[SETUP] Token DagsHub configurado desde código (len={len(DAGSHUB_TOKEN)})")

# Ruta al archivo del modelo (fallback local)
_MODELO_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# Variables globales (singleton)
_modelo = None
_explainer = None
_version = "v1.0"  # Valor por defecto
_dagshub_initialized = False


def _verificar_token():
    """Verifica si hay token de DagsHub configurado."""
    token = os.environ.get("DAGSHUB_TOKEN") or os.environ.get("DAGSHUB_USER_TOKEN")
    if token:
        logger.debug(f"Token DagsHub encontrado (longitud: {len(token)} chars)")
        return True
    else:
        logger.warning("No se encontró DAGSHUB_TOKEN en variables de entorno")
        logger.info("Para configurar el token, ejecuta:")
        logger.info("       set DAGSHUB_TOKEN=tu_token_aqui")
        print("       O usa: dagshub login")
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
    Descarga el modelo desde DagsHub (raw) intentando varias ramas.
    Guarda el archivo descargado localmente (_MODELO_PATH) para caché.
    
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
                    # Guardar en archivo local (caché persistente)
                    with open(_MODELO_PATH, "wb") as f:
                        f.write(response.content)
                    
                    logger.info(f"Modelo descargado y actualizado en: {_MODELO_PATH}")
                    return joblib.load(_MODELO_PATH)
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
    Carga el modelo de morosidad y el explainer SHAP.
    Primero intenta descargar desde DagsHub, luego usa fallback a archivo local.
    El archivo debe ser un diccionario con las claves 'modelo_prediccion' y 'shap_explainer'.
    Se carga una sola vez (singleton pattern).
    
    Returns:
        Tupla (modelo, explainer), o (None, None) si no existe el archivo.
    """
    global _modelo, _explainer, _version
    
    if _modelo is not None:
        return _modelo, _explainer
    
    # Primero: intentar descargar desde DagsHub
    # Esto actualizará el archivo local (_MODELO_PATH) si tiene éxito
    model_pack = _descargar_modelo_dagshub()
    
    # Fallback: Usar archivo local si la descarga falló
    if model_pack is None and os.path.exists(_MODELO_PATH):
        logger.warning("Falló la descarga -> Usando modelo local (Offline Mode)")
        try:
            model_pack = joblib.load(_MODELO_PATH)
        except Exception as e:
            logger.error(f"Error cargando fallback local: {e}")
            model_pack = None
    
    if model_pack is None:
        logger.error("No se pudo cargar el modelo desde ninguna fuente (ni DagsHub ni local)")
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
            "Verifica la conexión a DagsHub o agrega el archivo model.pkl "
            "en la carpeta src/morosidad/models_files/"
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
