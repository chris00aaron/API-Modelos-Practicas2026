# src/morosidad/router.py
from fastapi import APIRouter, HTTPException

from morosidad.schema import MorosidadRequest, MorosidadResponse
from morosidad.service import predecir_morosidad


router = APIRouter(
    prefix="/morosidad",
    tags=["Morosidad"]
)


@router.post(
    "/predict",
    response_model=MorosidadResponse,
    summary="Predecir morosidad",
    description="Predice la probabilidad de incumplimiento de pago de tarjeta de crédito."
)
def predict(request: MorosidadRequest) -> MorosidadResponse:
    """
    Realiza una predicción de morosidad basada en los datos del cliente.
    
    - **default**: True si se predice incumplimiento, False si no.
    - **probabilidad_default**: Probabilidad de incumplimiento (0.0 - 1.0)
    """
    try:
        return predecir_morosidad(request)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")
