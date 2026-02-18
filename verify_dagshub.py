"""
Script de verificación de conectividad y permisos de DagsHub para CHURN.
Lee el token desde la variable de entorno DAGSHUB_USER_TOKEN (o .env).

NOTA: Los archivos se suben con versioning='dvc', por lo que la API de
contenido (/content/) muestra el tamaño del pointer DVC (0 bytes).
Para verificar el contenido real, se usa el endpoint /raw/ que sirve
el archivo completo desde DVC storage.
"""
import os
import sys
import requests

from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv("DAGSHUB_USER_TOKEN")
if not TOKEN:
    print("❌ DAGSHUB_USER_TOKEN no encontrado en variables de entorno ni en .env")
    print("   Configura el token en el archivo .env del proyecto")
    sys.exit(1)

OWNER = "notificacionesbankmind"
REPO = "Modelos_BankMind_2026"
BASE = f"https://dagshub.com/api/v1/repos/{OWNER}/{REPO}"
RAW_BASE = f"https://dagshub.com/{OWNER}/{REPO}/raw/main"
h = {"Authorization": f"token {TOKEN}"}

print(f"Token: {TOKEN[:8]}...{TOKEN[-4:]} (len={len(TOKEN)})")
print()

# ============================================================
# 1. ACCESO AL REPOSITORIO
# ============================================================
print("1. ACCESO AL REPOSITORIO")
try:
    r = requests.get(BASE, headers=h, timeout=15)
    print("   Status:", r.status_code)
    if r.status_code == 200:
        d = r.json()
        print("   Repo:", d.get("full_name", "N/A"))
        print("   Privado:", d.get("private", "N/A"))
    else:
        print("   Error:", r.text[:200])
except Exception as e:
    print("   Error:", e)
    sys.exit(1)

# ============================================================
# 2. ARCHIVOS DEL MODELO DE CHURN (descarga real via /raw/)
# ============================================================
print()
print("2. ARCHIVOS DEL MODELO DE CHURN (descarga real via /raw/)")
print("   (Los archivos están en DVC, la API /content/ muestra 0 bytes)")
print("   (Verificación real: descargar contenido via /raw/ endpoint)")
print()

files = [
    "modelos/fuga/modelos_produccion/churn_champion.pkl",
    "modelos/fuga/modelos_produccion/best_model_churn.pkl",
    "modelos/fuga/modelos_produccion/scaler.pkl",
    "modelos/fuga/modelos_produccion/feature_names.pkl",
]

for f in files:
    basename = os.path.basename(f)
    try:
        raw_url = f"{RAW_BASE}/{f}"
        r = requests.get(raw_url, headers=h, timeout=30)
        if r.status_code == 200:
            size = len(r.content)
            if size > 100:
                print(f"   ✅ {basename} - {size:,} bytes (contenido real)")
            elif size > 0:
                print(f"   ⚠️ {basename} - {size} bytes (muy pequeño, posiblemente vacío)")
            else:
                print(f"   ❌ {basename} - 0 bytes (vacío)")
        elif r.status_code == 404:
            print(f"   ❌ {basename} - NO ENCONTRADO")
        else:
            print(f"   ⚠️ {basename} - HTTP {r.status_code}")
    except Exception as e:
        print(f"   ❌ {basename} - Error: {e}")

# ============================================================
# 3. VERIFICAR COMBO-PACK (intentar deserializar)
# ============================================================
print()
print("3. VERIFICACIÓN DEL COMBO-PACK (deserialización)")
try:
    import io
    import joblib

    combo_url = f"{RAW_BASE}/modelos/fuga/modelos_produccion/churn_champion.pkl"
    r = requests.get(combo_url, headers=h, timeout=60)
    if r.status_code == 200 and len(r.content) > 100:
        model_pack = joblib.load(io.BytesIO(r.content))
        if isinstance(model_pack, dict):
            meta = model_pack.get('meta_info', {})
            has_model = model_pack.get('modelo_prediccion') is not None
            has_scaler = model_pack.get('scaler') is not None
            has_features = model_pack.get('feature_names') is not None
            version = meta.get('version', 'N/A')
            print(f"   ✅ Combo-pack válido")
            print(f"      Versión: {version}")
            print(f"      Modelo:  {'✅ OK' if has_model else '❌ FALTA'}")
            print(f"      Scaler:  {'✅ OK' if has_scaler else '❌ FALTA'}")
            print(f"      Features: {'✅ OK (' + str(len(model_pack.get('feature_names', []))) + ')' if has_features else '❌ FALTA'}")
            if meta:
                print(f"      Accuracy: {meta.get('accuracy', 'N/A')}")
                print(f"      F1 Score: {meta.get('f1_score', 'N/A')}")
                print(f"      AUC-ROC:  {meta.get('auc_roc', 'N/A')}")
        else:
            print(f"   ⚠️ Archivo descargado pero no es un dict (tipo: {type(model_pack).__name__})")
    elif r.status_code == 200:
        print(f"   ⚠️ Archivo muy pequeño ({len(r.content)} bytes) - posiblemente stub DVC vacío")
    else:
        print(f"   ❌ No se pudo descargar combo-pack (HTTP {r.status_code})")
except ImportError:
    print("   ⚠️ joblib no disponible, no se puede verificar deserialización")
except Exception as e:
    print(f"   ❌ Error verificando combo-pack: {e}")

# ============================================================
# 4. EXPERIMENTOS MLFLOW
# ============================================================
print()
print("4. EXPERIMENTOS MLFLOW")
try:
    mlflow_url = f"https://dagshub.com/{OWNER}/{REPO}.mlflow/api/2.0/mlflow/experiments/list"
    r = requests.get(mlflow_url, headers=h, timeout=15)
    print("   MLflow Status:", r.status_code)
    if r.status_code == 200:
        exps = r.json().get("experiments", [])
        print("   Total experimentos:", len(exps))
        for exp in exps[:5]:
            print(f"     - {exp.get('name', '?')} (ID: {exp.get('experiment_id', '?')})")
    else:
        print("  ", r.text[:200])
except Exception as e:
    print(f"   Error: {e}")

# ============================================================
# 5. TEST DE PERMISOS DE ESCRITURA
# ============================================================
print()
print("5. TEST DE PERMISOS DE ESCRITURA")
try:
    branch_url = f"{BASE}/branches/main"
    r = requests.get(branch_url, headers=h, timeout=15)
    if r.status_code != 200:
        print(f"   ❌ No se pudo obtener branch info: HTTP {r.status_code}")
    else:
        last_commit = r.json().get("commit", {}).get("id", "")
        print(f"   Last commit SHA: {last_commit[:12]}...")

        test_dir = "modelos/fuga/modelos_produccion"
        upload_url = f"{BASE}/content/main/{test_dir}"
        test_content = b"write_permission_test"
        files_data = {
            "files": (".dagshub_write_test", test_content, "application/octet-stream")
        }
        form_data = {
            "commit_summary": "Test write permission (auto-delete)",
            "commit_choice": "direct",
            "versioning": "dvc",
            "last_commit": last_commit
        }
        r = requests.put(upload_url, files=files_data, data=form_data, headers=h, timeout=30)
        print(f"   Upload Status: {r.status_code}")

        if r.status_code in (200, 201):
            print("   ✅ PERMISOS DE ESCRITURA: OK")
        elif r.status_code == 403:
            print("   ❌ PERMISOS DE ESCRITURA: DENEGADO (403)")
            print("   → Genera un nuevo token con scope 'write' en:")
            print("     https://dagshub.com/user/settings/tokens")
        elif r.status_code == 401:
            print("   ❌ TOKEN INVÁLIDO (401)")
        else:
            print(f"   ⚠️ HTTP {r.status_code}: {r.text[:300]}")

except Exception as e:
    print(f"   Error: {e}")

print()
print("=" * 50)
print("VERIFICACIÓN COMPLETADA")
print("=" * 50)
