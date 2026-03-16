# main.py
import logging
import sys
import os
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

# ============================================================
# APScheduler para Performance Monitor del modelo de Churn
# ============================================================
_scheduler = None


def _setup_churn_monitor():
    """Configura el scheduler para el monitor de rendimiento de Churn."""
    global _scheduler
    try:
        from apscheduler.schedulers.background import BackgroundScheduler # type: ignore
        from fuga.service.performance_monitor import (
            performance_monitor,
            MONITOR_ENABLED,
            MONITOR_INTERVAL_HOURS,
        )

        if not MONITOR_ENABLED:
            print("[CHURN MONITOR] Monitor deshabilitado por configuración (CHURN_MONITOR_ENABLED=false)")
            return

        _scheduler = BackgroundScheduler()
        _scheduler.add_job(
            performance_monitor.run_scheduled_evaluation,
            'interval',
            hours=MONITOR_INTERVAL_HOURS,
            id='churn_performance_monitor',
            name='Churn Performance Monitor',
            replace_existing=True,
            max_instances=1,  # Solo una instancia a la vez
        )
        _scheduler.start()
        print(
            f"[CHURN MONITOR] ✅ Scheduler iniciado. "
            f"Evaluación cada {MONITOR_INTERVAL_HOURS} horas."
        )
    except ImportError:
        print(
            "[CHURN MONITOR] ⚠️ APScheduler no instalado. "
            "El monitor automático NO está activo. "
            "Instala con: pip install apscheduler"
        )
    except Exception as e:
        print(f"[CHURN MONITOR] ❌ Error iniciando scheduler: {e}")


def _shutdown_churn_monitor():
    """Detiene el scheduler del monitor de rendimiento."""
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        print("[CHURN MONITOR] Scheduler detenido.")
        _scheduler = None


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

    # Verificar permisos del token DagsHub para escritura (CHURN)
    print("[STARTUP] Verificando permisos del token DagsHub para CHURN...")
    try:
        from fuga import dagshub_client as churn_dagshub
        token_info = churn_dagshub.check_token_permissions()
        for msg in token_info.get('messages', []):
            print(f"[STARTUP] [CHURN TOKEN] {msg}")
        if not token_info.get('write'):
            print(
                "[STARTUP] ⚠️ CHURN: El token DagsHub NO tiene permisos de escritura. "
                "El reentrenamiento NO podrá subir modelos a DagsHub."
            )
    except Exception as e:
        print(f"[STARTUP] ⚠️ No se pudo verificar token DagsHub: {e}")

    # Iniciar monitor de rendimiento de Churn
    print("[STARTUP] Iniciando monitor de rendimiento de Churn...")
    _setup_churn_monitor()


@app.on_event("shutdown")
async def shutdown_event():
    """Limpia recursos al detener la API."""
    _shutdown_churn_monitor()


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