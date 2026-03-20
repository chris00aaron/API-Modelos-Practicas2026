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

# Iniciar la aplicación FastAPI
app = FastAPI(
    title="BankMind API",
    description="API para detección de fraudes",
    version="1.0.0"
)

# Importar Routers
from src.fraude.router import router as fraud_router

# Registrar Routers
app.include_router(fraud_router)

# Precargar modelos al iniciar la API
@app.on_event("startup")
async def startup_event():
    """Precarga los modelos al iniciar la API para ver logs de conexión."""
    logger_main.info("[STARTUP] Inicializando servicio de fraude...")
    try:
        from fraude.service.fraud_service import FraudService
        app.state.fraud_service = FraudService()
        logger_main.info("[STARTUP] FraudService inicializado y disponible en app.state")
    except Exception as e:
        logger_main.error("[STARTUP] Error inicializando FraudService: %s", e)
        app.state.fraud_service = None

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
