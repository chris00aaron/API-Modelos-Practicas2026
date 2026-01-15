# main.py
import sys
import os
from fastapi import BackgroundTasks, FastAPI, HTTPException

#Dependencis Internas
from src.retiro_atm.schema.input_retiro_atm import InputDataRetiroAtm
from src.retiro_atm.schema.output_retiro_atm import OutputDataRetiroAtm
from src.retiro_atm.service.service_prediction_retiro_atm import ServicioPredicticionRetiroAtm

# Agregar src al path para importaciones
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/'))

# Iniciar la aplicación FastAPI
app = FastAPI()

# Instancia global de los servicios
servicioPrediccionRetiro = ServicioPredicticionRetiroAtm()

#Codigo base
@app.post("/predecir/retiro",tags=["Predicción de Temperatura del Estátor"])
async def predecir_temperatura(input_data: InputDataRetiroAtm ) -> OutputDataRetiroAtm:
    """
    Endpoint para predecir la temperatura dado un conjunto de datos de entrada.
    """
    try:
        return servicioPrediccionRetiro.predecir_retiro(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error interno del servidor")
    
#Codigo base
@app.get("/vivo",tags=["Verificación de la disponibilidad de la api"])
async def health():
    """
    Endpoint para verificar si esta funcionando lka API
    """
    return {"mensaje": "ESTOY VIVO."}

#Inicializacion del servidor local
if __name__ == "__main__":
    import uvicorn  
    uvicorn.run(app, host="0.0.0.0", port=8000)