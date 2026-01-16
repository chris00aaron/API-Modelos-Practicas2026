import sys
import os
from fastapi import FastAPI, HTTPException
import uvicorn

# 1. Configuración del Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 2. Importaciones
try:
    from fuga.schema.inputs import ChurnInput
    from fuga.service.churn_service import churn_service
except ImportError as e:
    print(f"Error de importación: {e}")
    sys.exit(1)

# 3. INICIAR LA APLICACIÓN (Esta es la línea que busca Uvicorn)
app = FastAPI(title="API Predicción de Churn", version="1.0.0")

@app.get("/")
def home():
    return {"message": "API Fuga restaurada y funcionando "}

@app.post("/predict")
def predict_churn(data: ChurnInput):
    input_data = data.model_dump()
    result = churn_service.predict(input_data)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

if __name__ == "__main__":
    uvicorn.run("main_fuga:app", host="127.0.0.1", port=8000, reload=True)