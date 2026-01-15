# main.py
import sys
import os
from fastapi import FastAPI

# Agregar src al path para importaciones
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/'))

# Iniciar la aplicación FastAPI
app = FastAPI(
    title="API de Modelos de Predicción",
    description="API para modelos de Machine Learning",
    version="1.0.0"
)

# ==================== ROUTERS ====================
# Importar y registrar routers de cada módulo

# Módulo de Morosidad
from morosidad import router as morosidad_router
from morosidad.models_files import cargar_modelo as cargar_modelo_morosidad
app.include_router(morosidad_router)


# ==================== STARTUP EVENT ====================
@app.on_event("startup")
async def startup_event():
    """Pre-carga los modelos al iniciar la API para evitar latencia en la primera predicción."""
    print("[INFO] Iniciando carga de modelos...")
    cargar_modelo_morosidad()
    print("[INFO] Todos los modelos han sido cargados correctamente.")

# Aquí se agregarán otros módulos (fraude, fuga, retiro_atm)
# from fraude import router as fraude_router
# app.include_router(fraude_router)


# ==================== ROOT ====================
@app.get("/", tags=["Root"])
def root():
    """Endpoint raíz de la API."""
    return {"message": "API de Modelos de Predicción", "status": "running"}


# ==================== RUN ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)