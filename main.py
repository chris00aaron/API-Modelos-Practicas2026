# main.py
import sys
import os
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

# Agregar src al path para importaciones
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/'))

# Iniciar la aplicación FastAPI
app = FastAPI(
    title="BankMind API",
    description="API para detección de fraudes y otros modelos bancarios",
    version="1.0.0"
)

# Importar Routers
from src.fraude.router import router as fraud_router

# Registrar Routers
app.include_router(fraud_router)

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de BankMind v2. Visita /docs para ver la documentación."}

#Codigo base


#Inicializacion del servidor local
if __name__ == "__main__":
    import uvicorn  
    uvicorn.run(app, host="0.0.0.0", port=8000)