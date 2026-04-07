from fastapi import APIRouter, HTTPException, BackgroundTasks
from fuga.schema.inputs import ChurnInput, ChurnBatchInput
from fuga.service.churn_service import churn_service
from fuga.service.auto_training_service import auto_training_service
from fuga.service.performance_monitor import performance_monitor
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


@router.post("/train")
def train_churn_model():
    """
    Inicia el proceso de auto-entrenamiento para el modelo de Churn.
    Ejecuta de forma síncrona y retorna las métricas del entrenamiento.
    """
    try:
        result = auto_training_service.train_model()
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Error desconocido"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/train/status")
def get_training_status():
    """
    Retorna el estado actual del entrenamiento.
    Útil para polling desde el frontend.
    """
    return auto_training_service.get_status()


# ============================================================
# PERFORMANCE MONITOR ENDPOINTS
# ============================================================

@router.get("/monitor/status")
def get_monitor_status():
    """
    Retorna el estado actual del monitor de rendimiento.
    Incluye métricas de la última evaluación, próxima evaluación programada,
    y configuración del monitor.
    """
    return performance_monitor.get_status()


@router.post("/monitor/evaluate")
def trigger_evaluation():
    """
    Dispara manualmente una evaluación del rendimiento del modelo.
    Compara predicciones históricas contra ground truth (account_details.exited).
    Si el Recall cae por debajo del umbral, dispara re-entrenamiento automático.
    Devuelve el estado completo del monitor (mismo formato que GET /monitor/status).
    """
    try:
        result = performance_monitor.evaluate_model_performance()

        # Si el modelo está degradado, disparar re-entrenamiento
        if result.get("status") == "degraded":
            performance_monitor._trigger_auto_retrain(result)

        # Devolver el estado completo (incluye maturation_days, monitor_enabled, etc.)
        return performance_monitor.get_status()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al evaluar rendimiento: {str(e)}"
        )


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
