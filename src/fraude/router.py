from fastapi import APIRouter, HTTPException
from fraude.service.fraud_service import FraudService
from fraude.schema.inputs import FraudInput, FraudOutput, BatchFraudInput, BatchFraudOutput

router = APIRouter(
    prefix="/api/v1/fraud",
    tags=["Fraud Detection"]
)

# Singleton con lazy loading para evitar doble descarga del modelo
_fraud_service = None

def get_fraud_service():
    """
    Obtiene la instancia del servicio de fraude (singleton).
    La instancia se crea solo la primera vez que se llama.
    """
    global _fraud_service
    if _fraud_service is None:
        try:
            _fraud_service = FraudService()
        except Exception as e:
            print(f"CRITICAL ERROR: No se pudo inicializar el servicio de fraude: {e}")
            raise HTTPException(
                status_code=503, 
                detail="El servicio de fraude no está disponible. Error de inicialización."
            )
    return _fraud_service

@router.post("/predict", response_model=FraudOutput)
async def predict_fraud(input_data: FraudInput):
    """Predice fraude para una sola transacción"""
    fraud_service = get_fraud_service()
    
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
    fraud_service = get_fraud_service()
    
    if len(batch_input.transactions) > 100:
        raise HTTPException(status_code=400, detail="Máximo 100 transacciones por lote")
    
    try:
        result = fraud_service.predict_batch(batch_input)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el lote: {str(e)}")




@router.post("/reload")
async def reload_model_endpoint():
    """
    Endpoint administrativo para recargar el modelo en caliente desde DagsHub.
    Útil cuando se ha promovido un nuevo modelo champion.
    """
    fraud_service = get_fraud_service()
    
    try:
        result = fraud_service.reload_model()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recargando modelo: {str(e)}")
