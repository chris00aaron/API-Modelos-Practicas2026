# main.py
import sys
import os
from fastapi import BackgroundTasks, FastAPI, HTTPException

# Agregar src al path para importaciones
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/'))

#Dependencis Internas
from src.retiro_atm.schema.input_retiro_atm import InputDataRetiroAtm
from src.retiro_atm.schema.output_retiro_atm import OutputDataRetiroAtm
from src.retiro_atm.service.service_prediction_retiro_atm import ServicioPredicticionRetiroAtm

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

# Registrar Routers
app.include_router(fraud_router)
app.include_router(morosidad_router)

#Instanciamos Servicios
servicioPrediccionRetiro = ServicioPredicticionRetiroAtm()

#Codigo base
@app.post("/retiro_atm/predecir",tags=["Predicción del Retiro de Efectivo en ATM"])
async def predecir_temperatura(input_data: InputDataRetiroAtm ) -> OutputDataRetiroAtm:
    """
    Endpoint para predecir el monto ha retirar en un solo dia en un ATM.
    """
    try:
        return servicioPrediccionRetiro.predecir_retiro(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error interno del servidor")
    
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
    uvicorn.run(app, host="0.0.0.0", port=8000)