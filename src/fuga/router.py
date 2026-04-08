from fastapi import APIRouter, HTTPException, BackgroundTasks
from fuga.schema.inputs import ChurnInput, ChurnBatchInput
from fuga.service.churn_service import churn_service
from fuga.models_files.loader import (
    ChurnModelActualizandoseError,
    info_modelo as loader_info_modelo,
    recargar_desde_dagshub,
)

router = APIRouter(
    prefix="/churn",
    tags=["Churn Prediction"]
)


@router.post("/predict")
def predict_churn(data: ChurnInput):
    input_data = data.model_dump()
    try:
        result = churn_service.predict(input_data)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except ChurnModelActualizandoseError as e:
        raise HTTPException(status_code=503, detail="El modelo está siendo actualizado. Por favor, intente nuevamente más tarde.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-batch")
def predict_churn_batch(data: ChurnBatchInput):
    """
    Predice la fuga para múltiples clientes en una sola llamada vectorizada.
    Más eficiente que llamar a /predict N veces. No incluye explicaciones SHAP.
    Retorna una lista con id, churn_probability, risk_level, is_churn,
    prediction_confidence y model_version por cada cliente.
    """
    try:
        items = [item.model_dump() for item in data.customers]
        result = churn_service.predict_batch(items)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except ChurnModelActualizandoseError as e:
        raise HTTPException(status_code=503, detail="El modelo está siendo actualizado. Por favor, intente nuevamente más tarde.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# LIVE MODEL STATUS (equivalente a ATM /v1/withdrawal/info y /update)
# ============================================================

@router.get("/model/info",
    summary="Obtener información en tiempo real del modelo CHURN en producción.",
    description="Devuelve versión, estado (ready/updating/not_loaded), disponibilidad de SHAP y scaler.")
def get_model_info():
    """
    Endpoint de monitoreo en tiempo real del modelo CHURN.
    Equivalente a GET /api/atm/v1/withdrawal/info.
    """
    try:
        return loader_info_modelo()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener info del modelo: {str(e)}")


@router.post("/model/reload",
    summary="Recargar el modelo CHURN desde DagsHub (hot-reload sin reentrenamiento).",
    description="Descarga el modelo campeón más reciente desde DagsHub y lo reemplaza en memoria.")
def reload_model(background_tasks: BackgroundTasks):
    """
    Descarga el champion más reciente desde DagsHub y lo reemplaza en memoria.
    La descarga ocurre en background — las predicciones siguen disponibles mientras tanto.
    Equivalente a POST /api/atm/v1/withdrawal/update.
    """
    try:
        background_tasks.add_task(recargar_desde_dagshub)
        return {"mensaje": "Recarga del modelo iniciada. El nuevo modelo estará disponible en unos momentos."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al iniciar recarga: {str(e)}")
