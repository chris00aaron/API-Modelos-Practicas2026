# main.py
import logging
import sys
import os
from fastapi import FastAPI

logger_main = logging.getLogger("main")

# Agregar src al path para importaciones
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/'))

from src.configuration.logging_config import setup_logging

#Iniciamos el loggin
setup_logging()

# Iniciar la aplicación FastAPI
app = FastAPI(
    title="BankMind API PREDICCION - Modulo ATM",
    description="API para prediccion de retiro de efectivo en cajeros automaticos",
    version="1.0.0"
)

# Importar Routers
from src.retiro_atm.router import router as retiro_atm_router

# Registrar Routers
app.include_router(retiro_atm_router)

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