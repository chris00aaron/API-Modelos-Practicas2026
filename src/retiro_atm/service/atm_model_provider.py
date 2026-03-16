import numpy as np
import logging
import os
import io
import logging
import joblib
import requests
import dagshub
from typing import Optional, Any
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

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
        if self._modelo is None and not self._actualizando:
            self.recargar_modelo()

    def obtener_modelo(self) -> Any:
        """
        Método centralizado para obtener el modelo.
        Si no está listo o está actualizando, lanza la excepción.
        """
        if not self.esta_listo or self._modelo is None:
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
        repo_owner = os.environ.get("DAGSHUB_REPO_OWNER", "notificacionesbankmind")
        repo_name = os.environ.get("DAGSHUB_REPO_NAME", "Modelos_BankMind_2026")
        model_path = os.environ.get("DAGSHUB_MODEL_ATM_PATH", "modelos/atm/modelo_producción.pkl")
        token = os.environ.get("DAGSHUB_USER_TOKEN")
        
        # Inicializar DagsHub
        try:
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True) # type: ignore
        except Exception as e:
            logger.warning(f"Fallo al inicializar DagsHub: {e}")

        auth = (token, "") if token else None
        # Intentar en ramas principales
        for branch in ["main", "master"]:
            url = f"https://dagshub.com/{repo_owner}/{repo_name}/raw/{branch}/{model_path}"
            try:
                response = requests.get(url, auth=auth, timeout=30)
                if response.status_code == 200:
                    return joblib.load(io.BytesIO(response.content))
            except Exception as e:
                logger.debug(f"Fallo en rama {branch}: {e}")
        
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