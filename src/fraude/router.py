from fastapi import APIRouter, HTTPException
from fraude.service.fraud_service import FraudService
from fraude.schema.inputs import FraudInput, FraudOutput

router = APIRouter(
    prefix="/api/v1/fraud",
    tags=["Fraud Detection"]
)

# Instanciamos el servicio una única vez al cargar el módulo
try:
    fraud_service = FraudService()
except Exception as e:
    print(f"CRITICAL ERROR: No se pudo inicializar el servicio de fraude: {e}")
    fraud_service = None

@router.post("/predict", response_model=FraudOutput)
async def predict_fraud(input_data: FraudInput):
    if not fraud_service:
        raise HTTPException(status_code=503, detail="El servicio de fraude no está disponible. Error de inicialización.")
    
    try:
        result = fraud_service.predict(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la transacción: {str(e)}")
