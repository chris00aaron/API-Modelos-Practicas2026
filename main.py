# main.py
import logging
import sys
import os

# ── Force UTF-8 on Windows ──────────────────────────────────────────────
# Windows cmd.exe uses cp1252 by default, which crashes on Unicode emojis
# (✅, ⚠️, ❌, etc.) used in log messages throughout the codebase.
# Reconfigure stdout/stderr to use UTF-8 with error replacement.
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
# ─────────────────────────────────────────────────────────────────────────
from fastapi import BackgroundTasks, FastAPI, HTTPException
from contextlib import asynccontextmanager

logger_main = logging.getLogger("main")

# Herramientas de log y monitoreo

# Agregar src al path para importaciones
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/'))

from src.configuration.logging_config import setup_logging

#Iniciamos el loggin
setup_logging()

# 2. Importaciones
try:
    from fuga.schema.inputs import ChurnInput
    from fuga.service.churn_service import churn_service
except ImportError as e:
    print(f"Error de importación: {e}")
    sys.exit(1)


# Iniciar la aplicación FastAPI
app = FastAPI(
    title="BankMind API",
    description="API para detección de fraudes y otros modelos bancarios",
    version="1.0.0"
)

# Importar Routers
from src.fraude.router import router as fraud_router
from src.morosidad.router import router as morosidad_router
from src.retiro_atm.router import router as retiro_atm_router
from src.fuga.router import router as fuga_router

# Registrar Routers
app.include_router(fraud_router)
app.include_router(morosidad_router)
app.include_router(retiro_atm_router)
app.include_router(fuga_router)

# Precargar modelos al iniciar la API
@app.on_event("startup")
async def startup_event():
    """Precarga los modelos al iniciar la API para ver logs de conexión."""
    print("[STARTUP] Precargando modelo de morosidad...")
    try:
        from morosidad.models_files import cargar_modelo
        cargar_modelo()
        print("[STARTUP] Modelo de morosidad precargado correctamente")
    except Exception as e:
        print(f"[STARTUP] Error precargando modelo de morosidad: {e}")
    
    logger_main.info("[STARTUP] Inicializando servicio de fraude...")
    try:
        from fraude.service.fraud_service import FraudService
        app.state.fraud_service = FraudService()
        logger_main.info("[STARTUP] FraudService inicializado y disponible en app.state")
    except Exception as e:
        logger_main.error("[STARTUP] Error inicializando FraudService: %s", e)
        app.state.fraud_service = None

    print("[STARTUP] Precargando modelo de Churn (Fuga)...")
    try:
        from fuga.models_files.loader import cargar_modelos
        cargar_modelos()
        print("[STARTUP] Modelo de Churn precargado correctamente")
    except Exception as e:
        print(f"[STARTUP] Error precargando modelo de Churn: {e}")


    # Nota: el scheduler del monitor Churn vive en api-self-training-Bankmind/main.py


#Codigo base
@app.get("/vivo",tags=["Verificación de la disponibilidad de la api"])
async def health():
    """
    Endpoint para verificar si esta funcionando la API
    """
    return {"mensaje": "ESTOY VIVO."}

#Inicializacion del servidor local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)