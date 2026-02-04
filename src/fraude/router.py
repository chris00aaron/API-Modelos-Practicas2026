from fastapi import APIRouter, HTTPException
from fraude.service.fraud_service import FraudService
from fraude.schema.inputs import FraudInput, FraudOutput, BatchFraudInput, BatchFraudOutput

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
    """Predice fraude para una sola transacción"""
    if not fraud_service:
        raise HTTPException(status_code=503, detail="El servicio de fraude no está disponible. Error de inicialización.")
    
    try:
        result = fraud_service.predict(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la transacción: {str(e)}")

@router.post("/predict-batch", response_model=BatchFraudOutput)
async def predict_fraud_batch(batch_input: BatchFraudInput):
    """
    Predice fraude para múltiples transacciones en un solo lote.
    Más eficiente que llamar /predict múltiples veces.
    Máximo recomendado: 100 transacciones por lote.
    """
    if not fraud_service:
        raise HTTPException(status_code=503, detail="El servicio de fraude no está disponible. Error de inicialización.")
    
    if len(batch_input.transactions) > 100:
        raise HTTPException(status_code=400, detail="Máximo 100 transacciones por lote")
    
    try:
        result = fraud_service.predict_batch(batch_input)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el lote: {str(e)}")

