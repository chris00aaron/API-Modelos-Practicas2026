"""
router.py — Endpoints HTTP del módulo de fraude.

Cambios respecto a la versión anterior:
- Singleton de FraudService gestionado por FastAPI lifespan (app.state),
  eliminando el patrón `global` sin thread-safety.
- Endpoint /reload protegido con X-Internal-Token para evitar DoS.
- Logging estructurado en lugar de print().
"""
import logging
import os

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from fraude.schema.inputs import BatchFraudInput, BatchFraudOutput, FraudInput, FraudOutput
from fraude.service.fraud_service import FraudService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/fraud",
    tags=["Fraud Detection"],
)

# Token interno para el endpoint /reload — configúralo en .env
# Ejemplo:  FRAUD_RELOAD_TOKEN=supersecreto123
_RELOAD_TOKEN = os.environ.get("FRAUD_RELOAD_TOKEN", "")


# ---------------------------------------------------------------------------
# Dependency — obtiene FraudService desde app.state (inyectado en lifespan)
# ---------------------------------------------------------------------------

def get_fraud_service(request: Request) -> FraudService:
    """
    Extrae la instancia de FraudService desde app.state.
    El servicio se crea una sola vez durante el arranque del servidor (lifespan).
    Thread-safe: FastAPI garantiza que app.state es immutable tras el startup.
    """
    service: FraudService | None = getattr(request.app.state, "fraud_service", None)
    if service is None:
        logger.critical(
            "FraudService no está disponible en app.state. "
            "Verifica que el lifespan de la aplicación lo inicializó correctamente."
        )
        raise HTTPException(
            status_code=503,
            detail="El servicio de IA de fraude no está disponible.",
        )
    return service


def _verify_reload_token(x_internal_token: str = Header(default="")) -> None:
    """
    Valida el token de seguridad para el endpoint /reload.
    Configura FRAUD_RELOAD_TOKEN en .env. Si no está configurado, el endpoint
    quedará bloqueado para evitar exposición accidental.
    """
    if not _RELOAD_TOKEN:
        raise HTTPException(
            status_code=503,
            detail=(
                "El endpoint /reload está deshabilitado. "
                "Configura FRAUD_RELOAD_TOKEN en .env para activarlo."
            ),
        )
    if x_internal_token != _RELOAD_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Token de recarga inválido.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=FraudOutput)
async def predict_fraud(
    input_data: FraudInput,
    fraud_service: FraudService = Depends(get_fraud_service),
):
    """Predice fraude para una sola transacción."""
    try:
        return fraud_service.predict(input_data)
    except Exception as exc:
        logger.error("Error en /predict para tx %s: %s", input_data.transaction_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la transacción: {exc}",
        )


@router.post("/predict-batch", response_model=BatchFraudOutput)
async def predict_fraud_batch(
    batch_input: BatchFraudInput,
    fraud_service: FraudService = Depends(get_fraud_service),
):
    """
    Predice fraude para múltiples transacciones en un solo lote (vectorizado).
    Máximo recomendado: 100 transacciones por lote.
    """
    if len(batch_input.transactions) > 100:
        raise HTTPException(
            status_code=400,
            detail="Máximo 100 transacciones por lote.",
        )
    try:
        return fraud_service.predict_batch(batch_input)
    except Exception as exc:
        logger.error("Error en /predict-batch (%d txs): %s",
                     len(batch_input.transactions), exc)
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando el lote: {exc}",
        )


@router.post("/reload", dependencies=[Depends(_verify_reload_token)])
async def reload_model(
    fraud_service: FraudService = Depends(get_fraud_service),
):
    """
    Endpoint administrativo para recargar el modelo champion desde DagsHub.
    Requiere el header  X-Internal-Token: <FRAUD_RELOAD_TOKEN>.
    """
    try:
        result = fraud_service.reload()
        logger.info("Modelo recargado correctamente. Versión: %s", result.get("version"))
        return result
    except Exception as exc:
        logger.error("Error recargando modelo: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Error recargando modelo: {exc}",
        )
