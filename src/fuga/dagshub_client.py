# src/fuga/dagshub_client.py
"""
Cliente DagsHub exclusivo para el módulo CHURN.
Maneja upload/download de modelo combo-pack con verificación de integridad.

Combo-pack (.pkl): {
    'modelo_prediccion': XGBClassifier,
    'scaler': StandardScaler,
    'feature_names': list[str],
    'meta_info': {
        'version': 'v_1739700000',
        'accuracy': 0.86,
        'f1_score': 0.83,
        ...
    }
}
"""

import os
import logging
import io
import joblib
import requests
import dagshub
import mlflow
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configuración DagsHub
DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER", "notificacionesbankmind")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "Modelos_BankMind_2026")
DAGSHUB_CHURN_MODEL_PATH = "modelos/fuga/modelos_produccion/churn_champion.pkl"

# Rutas legacy (3 archivos separados — fallback para loader.py)
DAGSHUB_LEGACY_MODEL_PATH = "modelos/fuga/modelos_produccion/best_model_churn.pkl"
DAGSHUB_LEGACY_SCALER_PATH = "modelos/fuga/modelos_produccion/scaler.pkl"
DAGSHUB_LEGACY_FEATURES_PATH = "modelos/fuga/modelos_produccion/feature_names.pkl"

DAGSHUB_TOKEN = os.getenv("DAGSHUB_USER_TOKEN")

_dagshub_initialized = False


def init_dagshub_connection():
    """Inicializa la conexión a DagsHub y MLflow tracking."""
    global _dagshub_initialized
    if not _dagshub_initialized:
        if not DAGSHUB_TOKEN:
            logger.warning("⚠️ DAGSHUB_USER_TOKEN no configurado. MLflow no funcionará remotamente.")
            return

        try:
            dagshub.init(
                repo_owner=DAGSHUB_REPO_OWNER,
                repo_name=DAGSHUB_REPO_NAME,
                mlflow=True
            )
            _dagshub_initialized = True
            logger.info(f"[CHURN] ✅ Conectado a DagsHub: {DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}")
            logger.info(f"[CHURN] 📡 MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        except Exception as e:
            logger.error(f"[CHURN] ❌ Error inicializando DagsHub: {e}")


def check_token_permissions() -> dict:
    """
    Verifica si el token de DagsHub tiene permisos de lectura y escritura.
    Útil para diagnóstico al arranque de la API.

    Returns:
        dict con 'read', 'write', 'token_present', y mensajes de diagnóstico.
    """
    result = {
        'token_present': bool(DAGSHUB_TOKEN),
        'read': False,
        'write': False,
        'messages': []
    }

    if not DAGSHUB_TOKEN:
        result['messages'].append(
            "❌ DAGSHUB_USER_TOKEN no está configurado en .env. "
            "El modelo NO podrá subirse a DagsHub."
        )
        return result

    headers = {'Authorization': f'token {DAGSHUB_TOKEN}'}

    # Test lectura
    try:
        repo_url = f"https://dagshub.com/api/v1/repos/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}"
        resp = requests.get(repo_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            result['read'] = True
            result['messages'].append("✅ Token tiene permisos de lectura")
        else:
            result['messages'].append(
                f"❌ Token sin permisos de lectura (HTTP {resp.status_code})"
            )
    except Exception as e:
        result['messages'].append(f"❌ Error verificando lectura: {e}")

    # Test escritura: intentar obtener branches (requiere auth)
    try:
        branch_url = (
            f"https://dagshub.com/api/v1/repos/{DAGSHUB_REPO_OWNER}/"
            f"{DAGSHUB_REPO_NAME}/branches/main"
        )
        resp = requests.get(branch_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            last_commit = resp.json().get('commit', {}).get('id', '')
            if last_commit:
                # Intentar upload de prueba minúscula a un archivo temporal
                test_dir = "modelos/fuga/modelos_produccion"
                upload_url = (
                    f"https://dagshub.com/api/v1/repos/{DAGSHUB_REPO_OWNER}/"
                    f"{DAGSHUB_REPO_NAME}/content/main/{test_dir}"
                )
                test_content = b"token_write_test"
                files = {
                    'files': ('.write_test', test_content, 'application/octet-stream')
                }
                data = {
                    'commit_summary': '[CHURN] Token write permission test',
                    'commit_choice': 'direct',
                    'versioning': 'dvc',
                    'last_commit': last_commit
                }
                write_resp = requests.put(
                    upload_url, files=files, data=data,
                    headers=headers, timeout=30
                )
                if write_resp.status_code in (200, 201):
                    result['write'] = True
                    result['messages'].append("✅ Token tiene permisos de escritura")
                elif write_resp.status_code == 403:
                    result['messages'].append(
                        "❌ Token SIN permisos de escritura (HTTP 403). "
                        "Genera un nuevo token con scope 'write' en: "
                        "https://dagshub.com/user/settings/tokens"
                    )
                else:
                    result['messages'].append(
                        f"⚠️ Test de escritura retornó HTTP {write_resp.status_code}: "
                        f"{write_resp.text[:200]}"
                    )
        else:
            result['messages'].append(
                f"⚠️ No se pudo verificar escritura (branch info HTTP {resp.status_code})"
            )
    except Exception as e:
        result['messages'].append(f"⚠️ Error verificando escritura: {e}")

    return result


def download_current_champion():
    """
    Descarga el modelo Champion actual (combo-pack) desde DagsHub.

    Returns:
        tuple: (modelo, scaler, feature_names, metadata) o (None, None, None, None) si falla.
    """
    if not DAGSHUB_TOKEN:
        logger.warning("[CHURN] No se puede descargar Champion: Falta Token")
        return None, None, None, None

    headers = {'Authorization': f'token {DAGSHUB_TOKEN}'}
    branches = ["main", "master"]

    for branch in branches:
        url = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}/raw/{branch}/{DAGSHUB_CHURN_MODEL_PATH}"
        logger.info(f"[CHURN] ⬇️ Intentando descargar Champion desde: {url}")

        try:
            response = requests.get(url, headers=headers, timeout=60)
            if response.status_code == 200:
                # Verificar que el contenido no esté vacío
                if len(response.content) < 100:
                    logger.warning(
                        f"[CHURN] Archivo descargado pero muy pequeño ({len(response.content)} bytes). "
                        "Posiblemente es un stub DVC vacío."
                    )
                    continue

                logger.info("[CHURN] ✅ Champion descargado exitosamente.")
                model_pack = joblib.load(io.BytesIO(response.content))

                if isinstance(model_pack, dict):
                    return (
                        model_pack.get('modelo_prediccion'),
                        model_pack.get('scaler'),
                        model_pack.get('feature_names'),
                        model_pack.get('meta_info', {})
                    )
                else:
                    # Legacy: solo el modelo
                    return model_pack, None, None, {}
            else:
                logger.warning(f"[CHURN] Rama '{branch}' retornó status {response.status_code}")
        except Exception as e:
            logger.error(f"[CHURN] Error descargando de {branch}: {e}")

    logger.warning("[CHURN] ⚠️ No se encontró Champion combo-pack en DagsHub")
    return None, None, None, None


def _get_last_commit_sha() -> str:
    """
    Obtiene el SHA del último commit de la rama main.
    Requerido para la API de upload de DagsHub.

    Returns:
        SHA del commit o string vacío si falla.
    """
    headers = {'Authorization': f'token {DAGSHUB_TOKEN}'}
    branch_url = (
        f"https://dagshub.com/api/v1/repos/{DAGSHUB_REPO_OWNER}/"
        f"{DAGSHUB_REPO_NAME}/branches/main"
    )
    try:
        branch_resp = requests.get(branch_url, headers=headers, timeout=15)
        if branch_resp.status_code != 200:
            logger.error(f"[CHURN] ❌ No se pudo obtener last_commit: HTTP {branch_resp.status_code}")
            return ""
        sha = branch_resp.json().get('commit', {}).get('id', '')
        logger.info(f"[CHURN] 📌 Last commit SHA: {sha[:12]}...")
        return sha
    except Exception as e:
        logger.error(f"[CHURN] ❌ Error obteniendo last_commit: {e}")
        return ""


def _upload_file_to_dagshub(file_bytes: bytes, dagshub_path: str,
                             version_tag: str, last_commit: str) -> bool:
    """
    Sube un archivo individual a DagsHub usando la API REST.

    Args:
        file_bytes: Bytes del archivo a subir.
        dagshub_path: Ruta completa en el repositorio (ej: 'modelos/fuga/.../file.pkl')
        version_tag: Tag de versión para el commit message.
        last_commit: SHA del último commit (requerido por DagsHub).

    Returns:
        True si upload exitoso, False en caso contrario.
    """
    headers = {'Authorization': f'token {DAGSHUB_TOKEN}'}
    model_dir = os.path.dirname(dagshub_path)
    model_filename = os.path.basename(dagshub_path)

    upload_url = (
        f"https://dagshub.com/api/v1/repos/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}"
        f"/content/main/{model_dir}"
    )

    files = {
        'files': (model_filename, file_bytes, 'application/octet-stream')
    }
    data = {
        'commit_summary': f"[CHURN] Update {model_filename}: {version_tag}",
        'commit_message': f"Modelo CHURN archivo {model_filename} actualizado. Versión: {version_tag}",
        'commit_choice': 'direct',
        'versioning': 'dvc',
        'last_commit': last_commit
    }

    try:
        response = requests.put(
            upload_url,
            files=files,
            data=data,
            headers=headers,
            timeout=120
        )

        if response.status_code in (200, 201):
            logger.info(f"[CHURN] ✅ Archivo '{model_filename}' subido a DagsHub (HTTP {response.status_code})")
            return True
        elif response.status_code == 403:
            logger.error(
                f"[CHURN] ❌ SIN PERMISOS DE ESCRITURA (HTTP 403) para '{model_filename}'. "
                "El token DAGSHUB_USER_TOKEN no tiene scope 'write'. "
                "Genera un nuevo token en: https://dagshub.com/user/settings/tokens"
            )
            return False
        else:
            logger.error(
                f"[CHURN] ❌ Error subiendo '{model_filename}': "
                f"HTTP {response.status_code}: {response.text[:300]}"
            )
            return False
    except Exception as e:
        logger.error(f"[CHURN] ❌ Exception subiendo '{model_filename}': {e}")
        return False


def upload_champion(model_bytes: bytes, version_tag: str) -> bool:
    """
    Sube el nuevo modelo Champion (combo-pack) a DagsHub.
    Usa la API REST con last_commit SHA para evitar conflictos.

    Args:
        model_bytes: Bytes del combo-pack serializado con joblib.
        version_tag: Tag de versión (ej: 'v_1739700000')

    Returns:
        True si el upload fue exitoso, False en caso contrario.
    """
    if not DAGSHUB_TOKEN:
        logger.error("[CHURN] ❌ No se puede subir Champion: Falta Token")
        return False

    try:
        repo_fullname = f"{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}"
        logger.info(f"[CHURN] 🚀 Subiendo Champion ({version_tag}) a {repo_fullname}...")

        # 1) Obtener último commit SHA
        last_commit = _get_last_commit_sha()
        if not last_commit:
            return False

        # 2) Subir combo-pack
        return _upload_file_to_dagshub(
            model_bytes, DAGSHUB_CHURN_MODEL_PATH, version_tag, last_commit
        )

    except Exception as e:
        logger.error(f"[CHURN] ❌ Error subiendo Champion: {e}")
        return False


def upload_individual_files(model, scaler, feature_names, version_tag: str) -> bool:
    """
    Sube los 3 archivos individuales (legacy) a DagsHub:
    - best_model_churn.pkl (modelo)
    - scaler.pkl (scaler)
    - feature_names.pkl (feature names)

    Esto asegura compatibilidad con el loader.py que usa el fallback legacy.

    Args:
        model: Objeto del modelo entrenado
        scaler: Objeto StandardScaler
        feature_names: Lista de nombres de features
        version_tag: Tag de versión

    Returns:
        True si todos los uploads fueron exitosos, False si alguno falló.
    """
    if not DAGSHUB_TOKEN:
        logger.error("[CHURN] ❌ No se puede subir archivos individuales: Falta Token")
        return False

    logger.info(f"[CHURN] 📦 Subiendo archivos individuales (legacy) - {version_tag}...")

    all_ok = True

    # Obtener último commit SHA (se necesita para cada upload y cambia tras cada commit)
    last_commit = _get_last_commit_sha()
    if not last_commit:
        return False

    # 1) Subir modelo
    try:
        buf = io.BytesIO()
        joblib.dump(model, buf)
        ok = _upload_file_to_dagshub(buf.getvalue(), DAGSHUB_LEGACY_MODEL_PATH, version_tag, last_commit)
        if not ok:
            all_ok = False
        else:
            # Actualizar last_commit para el siguiente archivo
            last_commit = _get_last_commit_sha()
    except Exception as e:
        logger.error(f"[CHURN] ❌ Error serializando modelo: {e}")
        all_ok = False

    # 2) Subir scaler
    if scaler is not None and last_commit:
        try:
            buf = io.BytesIO()
            joblib.dump(scaler, buf)
            ok = _upload_file_to_dagshub(buf.getvalue(), DAGSHUB_LEGACY_SCALER_PATH, version_tag, last_commit)
            if not ok:
                all_ok = False
            else:
                last_commit = _get_last_commit_sha()
        except Exception as e:
            logger.error(f"[CHURN] ❌ Error serializando scaler: {e}")
            all_ok = False

    # 3) Subir feature_names
    if feature_names is not None and last_commit:
        try:
            buf = io.BytesIO()
            joblib.dump(feature_names, buf)
            ok = _upload_file_to_dagshub(buf.getvalue(), DAGSHUB_LEGACY_FEATURES_PATH, version_tag, last_commit)
            if not ok:
                all_ok = False
        except Exception as e:
            logger.error(f"[CHURN] ❌ Error serializando feature_names: {e}")
            all_ok = False

    if all_ok:
        logger.info("[CHURN] ✅ Todos los archivos individuales subidos correctamente")
    else:
        logger.warning("[CHURN] ⚠️ Algunos archivos individuales no se pudieron subir")

    return all_ok


def verify_champion_integrity(expected_version: str) -> bool:
    """
    Verifica la integridad del upload re-descargando el combo-pack desde DagsHub
    y comparando la versión embebida con la esperada.

    Args:
        expected_version: Version tag recién subido (ej: 'v_1739700000')

    Returns:
        True si el archivo existe, se puede deserializar, y la versión coincide.
    """
    logger.info(f"[CHURN] 🔍 Verificando integridad del upload (esperado: {expected_version})...")

    if not DAGSHUB_TOKEN:
        logger.error("[CHURN] ❌ No se puede verificar: Falta Token")
        return False

    headers = {'Authorization': f'token {DAGSHUB_TOKEN}'}
    url = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}/raw/main/{DAGSHUB_CHURN_MODEL_PATH}"

    try:
        import time as _time
        _time.sleep(3)  # Espera para que DagsHub registre el commit

        response = requests.get(url, headers=headers, timeout=60)

        if response.status_code != 200:
            logger.error(f"[CHURN] ❌ Verificación falló: HTTP {response.status_code}")
            return False

        # Verificar que no sea un stub vacío
        if len(response.content) < 100:
            logger.error(
                f"[CHURN] ❌ Verificación falló: archivo demasiado pequeño "
                f"({len(response.content)} bytes), posiblemente stub DVC vacío"
            )
            return False

        model_pack = joblib.load(io.BytesIO(response.content))

        if not isinstance(model_pack, dict):
            logger.error("[CHURN] ❌ Verificación falló: formato inesperado (no es dict)")
            return False

        actual_version = model_pack.get('meta_info', {}).get('version', 'UNKNOWN')

        if actual_version != expected_version:
            logger.error(
                f"[CHURN] ❌ Verificación falló: esperada={expected_version}, obtenida={actual_version}"
            )
            return False

        has_model = model_pack.get('modelo_prediccion') is not None
        has_scaler = model_pack.get('scaler') is not None
        has_features = model_pack.get('feature_names') is not None

        if not has_model:
            logger.error("[CHURN] ❌ Verificación falló: 'modelo_prediccion' no encontrado")
            return False

        logger.info(
            f"[CHURN] ✅ Integridad verificada: version={actual_version}, "
            f"modelo={'OK' if has_model else 'FALTA'}, "
            f"scaler={'OK' if has_scaler else 'FALTA'}, "
            f"features={'OK' if has_features else 'FALTA'}"
        )
        return True

    except Exception as e:
        logger.error(f"[CHURN] ❌ Error en verificación de integridad: {e}")
        return False
