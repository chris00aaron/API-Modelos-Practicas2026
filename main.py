# main.py
import sys
import os
from fastapi import BackgroundTasks, FastAPI, HTTPException

# Herramientas de log y monitoreo

# Agregar src al path para importaciones
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/'))


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

# Registrar Routers
app.include_router(fraud_router)
app.include_router(morosidad_router)
app.include_router(retiro_atm_router)

#Codigo base
@app.get("/vivo",tags=["Verificación de la disponibilidad de la api"])
async def health():
    """
    Endpoint para verificar si esta funcionando la API
    """
    return {"mensaje": "ESTOY VIVO."}

@app.post("/fuga/predecir")
def predict_churn(data: ChurnInput):
    input_data = data.model_dump()
    result = churn_service.predict(input_data)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

#Inicializacion del servidor local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)