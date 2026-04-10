import numpy as np
import logging
import os
import io
import joblib
import requests
import dagshub
from typing import Optional, Any

logger = logging.getLogger("retiro_atm")

# ============================================================
# CONFIGURACIÓN HARDCODEADA — DagsHub ATM
# ============================================================
_REPO_OWNER = "notificacionesbankmind"
_REPO_NAME = "Modelos_BankMind_2026"
_MODEL_PATH = "modelos/atm/modelo.pkl"
_TOKEN = "1022993058d503226b5e83a649a067c0c2ef2e73"

# Forzar el token en el entorno para que dagshub/mlflow lo encuentren
os.environ["DAGSHUB_USER_TOKEN"] = _TOKEN

print(f"[ATM] Token DagsHub hardcodeado (len={len(_TOKEN)})")
print(f"[ATM] Repo: {_REPO_OWNER}/{_REPO_NAME}")

class ModeloActualizandoseError(Exception):
    """Excepción para cuando se intenta predecir mientras el modelo carga."""
    pass

class AtmModelProvider:
    _instancia = None
    _modelo: Optional[Any] = None
    _explainer: Optional[Any] = None
    _version: str = "v1.0"
    _actualizando: bool = False

    def __new__(cls):
        if cls._instancia is None:
            cls._instancia = super(AtmModelProvider, cls).__new__(cls)
        return cls._instancia

    def __init__(self):
        # Evitar recarga si ya existe instancia
        if not self.esta_listo:
            self.recargar_modelo()

    def obtener_modelo(self) -> Any:
        """
        Método centralizado para obtener el modelo.
        Si no está listo o está actualizando, lanza la excepción.
        """
        if not self.esta_listo:
            raise ModeloActualizandoseError(
                "Error: API de predicción actualizándose. Por favor, intente en unos segundos."
            )
        return self._modelo

    def recargar_modelo(self):
        """Carga o actualiza el modelo desde DagsHub."""
        self._actualizando = True
        logger.info("Iniciando carga de modelo desde DagsHub...")
        
        try:
            model_pack = self._descargar_desde_dagshub()
            print(f"[DEBUG] model_pack obtenido: {type(model_pack)}, keys: {list(model_pack.keys()) if isinstance(model_pack, dict) else 'N/A'}")
            
            if model_pack:
                self._procesar_paquete_modelo(model_pack)
                logger.info(f"Modelo cargado exitosamente. Versión: {self._version}")
                return self._modelo
            else:
                logger.error("No se pudo obtener el paquete del modelo.")
                
        finally:
            self._actualizando = False

    def _descargar_desde_dagshub(self) -> Optional[Any]:
        """Lógica interna de descarga (sin acceso a disco)."""
        # Inicializar DagsHub (opcional, no bloquea si falla)
        try:
            dagshub.init(repo_owner=_REPO_OWNER, repo_name=_REPO_NAME, mlflow=True) # type: ignore
        except Exception as e:
            logger.warning(f"dagshub.init() falló: {e}. Descarga HTTP directa como fallback.")

        auth = (_TOKEN, "")
        # Intentar en ramas principales
        for branch in ["main", "master"]:
            url = f"https://dagshub.com/{_REPO_OWNER}/{_REPO_NAME}/raw/{branch}/{_MODEL_PATH}"
            logger.info(f"Intentando descargar modelo desde: {url}")
            try:
                response = requests.get(url, auth=auth, timeout=30)
                if response.status_code == 200:
                    logger.info("Descarga exitosa, cargando modelo con joblib...")
                    return joblib.load(io.BytesIO(response.content))
                else:
                    logger.warning(f"Fallo en rama {branch}. HTTP {response.status_code}: {response.text[:200]}")
            except Exception as e:
                logger.debug(f"Fallo en rama {branch}: {e}")
        
        logger.error(f"No se pudo descargar el modelo de ninguna rama. Último intento hacia {url}")
        return None

    def _procesar_paquete_modelo(self, model_pack: Any):
        """Desempaqueta el diccionario o el modelo directo."""
        if isinstance(model_pack, dict):
            self._modelo = model_pack.get('modelo_prediccion')
            self._explainer = model_pack.get('shap_explainer')
            meta = model_pack.get('meta_info', {})
            self._version = meta.get('version', "v1.0") if isinstance(meta, dict) else "v1.0"
        else:
            self._modelo = model_pack
            self._version = "v1.0-legacy"

    @property
    def esta_listo(self) -> bool:
        return self._modelo is not None and not self._actualizando

    @property
    def info_modelo(self):
        return {
            "version": self._version,
            "status": "ready" if self.esta_listo else "updating",
            "has_explainer": self._explainer is not None
        }