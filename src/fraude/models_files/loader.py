"""
loader.py — Modelo de Fraude: descarga y caché en memoria desde DagsHub.

Responsabilidad única: descargar el .pkl del champion y exponerlo como singleton
thread-safe mediante una clase. No guarda archivos en disco.
"""
import io
import logging
import os

import dagshub
import joblib
import mlflow
import requests
import threading

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración DagsHub (constantes a nivel de módulo — solo lectura)
# ---------------------------------------------------------------------------
_REPO_OWNER = "notificacionesbankmind"
_REPO_NAME = "Modelos_BankMind_2026"
_MODEL_PATH = "modelos/fraude/modelo.pkl"
_BRANCHES = ["main", "master"]
_TOKEN = "1022993058d503226b5e83a649a067c0c2ef2e73"

# Timeout para HTTP: (connect_timeout, read_timeout) en segundos
_HTTP_TIMEOUT = (10, 60)

# Forzar el token en el entorno para que dagshub/mlflow lo encuentren
os.environ["DAGSHUB_USER_TOKEN"] = _TOKEN

print(f"[FRAUDE] Token DagsHub hardcodeado (len={len(_TOKEN)})")
print(f"[FRAUDE] Repo: {_REPO_OWNER}/{_REPO_NAME}")


# ---------------------------------------------------------------------------
# ModelLoader — encapsula el estado global eliminando variables globales
# ---------------------------------------------------------------------------
class ModelLoader:
    """
    Singleton que gestiona el ciclo de vida del modelo champion de fraude.

    - Descarga el .pkl desde DagsHub (solo si no está en memoria).
    - Expone reload() para actualizaciones en caliente.
    - Thread-safe en CPython gracias al GIL; si se usa multiprocessing
      cada worker gestiona su propia instancia.
    """

    def __init__(self):
        self._model_pack: dict | None = None
        self._version: str = "v1.0"
        self._dagshub_initialized: bool = False
        self._token: str = _TOKEN
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Pública
    # ------------------------------------------------------------------

    @property
    def version(self) -> str:
        return self._version

    def get(self) -> dict:
        """
        Devuelve el model_pack (singleton). Lo descarga si aún no está en
        memoria.

        Raises:
            RuntimeError: Si el modelo no pudo cargarse desde DagsHub.
        """
        if self._model_pack is None:
            with self._lock:
                if self._model_pack is None:
                    self._load()
        if self._model_pack is None:
            raise RuntimeError(
                "El modelo de fraude no está disponible. "
                "Verifica la conexión a DagsHub y las credenciales en .env"
            )
        return self._model_pack

    def reload(self) -> dict:
        """
        Fuerza la descarga y re-carga del champion desde DagsHub.
        Si la descarga o validación falla, el modelo anterior se preserva.

        Returns:
            El nuevo model_pack cargado.

        Raises:
            ValueError: Si el nuevo modelo no pudo cargarse. El pack anterior
                        se restaura automáticamente.
        """
        with self._lock:
            previous_pack = self._model_pack
            previous_version = self._version
            logger.info("🔄 Recarga solicitada — descargando nuevo modelo de fraude...")
            self._model_pack = None
            self._dagshub_initialized = False
            try:
                self._load()
                logger.info("✅ Recarga completada. Nueva versión: %s", self._version)
            except ValueError as exc:
                # Restaurar modelo anterior — el servicio sigue operativo
                self._model_pack = previous_pack
                self._version = previous_version
                logger.error(
                    "❌ Recarga fallida: %s. Manteniendo modelo anterior (v%s).",
                    exc, previous_version,
                )
                raise
        return self._model_pack

    # ------------------------------------------------------------------
    # Privada
    # ------------------------------------------------------------------

    def _init_dagshub(self) -> None:
        if self._dagshub_initialized:
            return
        try:
            logger.debug("Inicializando conexión DagsHub: %s/%s", _REPO_OWNER, _REPO_NAME)
            dagshub.init(repo_owner=_REPO_OWNER, repo_name=_REPO_NAME, mlflow=True)
            self._dagshub_initialized = True
            logger.info("DagsHub inicializado. MLflow URI: %s", mlflow.get_tracking_uri())
        except Exception as exc:
            self._dagshub_initialized = True
            logger.warning("dagshub.init() falló (%s: %s). Descarga HTTP directa como fallback.", type(exc).__name__, exc)

    def _download(self) -> dict | None:
        """
        Descarga el .pkl desde DagsHub directamente en memoria.

        Returns:
            dict con los componentes del modelo, o None si falla.
        """
        self._init_dagshub()
        auth = (self._token, "")

        for branch in _BRANCHES:
            url = (
                f"https://dagshub.com/{_REPO_OWNER}/{_REPO_NAME}"
                f"/raw/{branch}/{_MODEL_PATH}"
            )
            logger.info("Descargando modelo de fraude desde: %s", url)
            try:
                response = requests.get(url, auth=auth, timeout=_HTTP_TIMEOUT)
                if response.status_code == 200:
                    logger.info("Descarga completada (%d bytes). Cargando en memoria...",
                                len(response.content))
                    return joblib.load(io.BytesIO(response.content))
                logger.warning(
                    "Rama '%s' respondió %d — probando siguiente rama.",
                    branch, response.status_code,
                )
            except requests.exceptions.Timeout:
                logger.error("Timeout descargando modelo desde rama '%s' (%s)", branch, url)
            except Exception as exc:
                logger.warning("Error conectando a rama '%s': %s", branch, exc)

        logger.error("No se pudo descargar el modelo de ninguna rama de DagsHub.")
        return None

    def _load(self) -> None:
        """
        Descarga el modelo, lo valida y lo guarda en caché.

        Raises:
            ValueError: Si la descarga falla, el formato es incorrecto, o
                        el pack no contiene las claves críticas del modelo.
        """
        logger.info("Iniciando carga del modelo de fraude desde DagsHub...")
        pack = self._download()

        if pack is None:
            raise ValueError(
                "Descarga falló — DagsHub no devolvió el modelo. "
                "Verifica conectividad y credenciales."
            )

        if not isinstance(pack, dict):
            raise ValueError(
                f"El .pkl descargado no tiene el formato esperado (dict). "
                f"Tipo recibido: {type(pack).__name__}"
            )

        required = {"scaler", "model_xgb", "model_if", "encoders"}
        missing = required - pack.keys()
        if missing:
            raise ValueError(
                f"model_pack incompleto — faltan claves críticas: {missing}. "
                "El modelo no puede operar sin ellas."
            )

        meta = pack.get("meta_info", {})
        if isinstance(meta, dict):
            self._version = meta.get("version", "v1.0")

        if pack.get("explainer"):
            logger.info("SHAP Explainer cargado correctamente.")
        else:
            logger.warning(
                "No se encontró 'explainer' en el .pkl. "
                "La explicabilidad SHAP no estará disponible."
            )

        self._model_pack = pack
        logger.info("✅ Modelo de fraude cargado. Versión: %s", self._version)


# ---------------------------------------------------------------------------
# Instancia singleton del módulo — importada por fraud_service.py
# ---------------------------------------------------------------------------
model_loader = ModelLoader()