# src/morosidad/router.py
from fastapi import APIRouter, HTTPException

from morosidad.schema import MorosidadRequest, MorosidadResponse, BatchMorosidadRequest, BatchMorosidadResponse
from morosidad.service import predecir_morosidad, predecir_morosidad_batch
from morosidad.models_files.loader import cargar_modelo, obtener_version


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


@router.post(
    "/predict/batch",
    response_model=BatchMorosidadResponse,
    summary="Predecir morosidad en lote",
    description="Predice la probabilidad de incumplimiento para múltiples clientes (vectorizado)."
)
def predict_batch(request: BatchMorosidadRequest) -> BatchMorosidadResponse:
    """
    Realiza predicciones de morosidad en lote.
    Mucho más eficiente que múltiples llamadas individuales.
    
    - **items**: Lista de datos de clientes
    - **predictions**: Lista de predicciones con índice para mantener orden
    """
    try:
        return predecir_morosidad_batch(request)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción batch: {str(e)}")


@router.post(
    "/refresh-model",
    summary="Hot Reload del Modelo",
    description="Fuerza la descarga del último modelo desde DagsHub y actualiza la memoria."
)
def refresh_model():
    """
    Endpoint administrativo para actualizar el modelo en caliente sin reiniciar el servicio.
    """
    try:
        model, _ = cargar_modelo(force_reload=True)
        if model:
            version = obtener_version()
            return {"status": "success", "message": f"Modelo recargado correctamente. Versión: {version}"}
        else:
            raise HTTPException(status_code=500, detail="Fallo al descargar el modelo desde DagsHub")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Hot Reload: {str(e)}")


@router.get(
    "/model-version",
    summary="Versión del Modelo Cargado",
    description="Retorna la versión del modelo actualmente cargado en memoria."
)
def model_version():
    """
    Retorna la versión del modelo cargado para verificación cruzada
    con el registro en production_model_default.
    """
    try:
        version = obtener_version()
        return {"version": version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo versión: {str(e)}")

