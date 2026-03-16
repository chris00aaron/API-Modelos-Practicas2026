from fastapi import APIRouter, HTTPException
from fuga.schema.inputs import ChurnInput
from fuga.service.churn_service import churn_service
from fuga.service.auto_training_service import auto_training_service
from fuga.service.performance_monitor import performance_monitor

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
