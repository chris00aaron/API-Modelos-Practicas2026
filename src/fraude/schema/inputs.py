"""
inputs.py — Esquemas Pydantic de request y response para el módulo de fraude.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# REQUEST
# ---------------------------------------------------------------------------

class FraudInput(BaseModel):
    """Datos de una transacción a evaluar."""

    transaction_id: str = Field(..., example="TXN-9834")
    id_cliente: str = Field(..., example="CLI-5502")
    trans_date_trans_time: str = Field(..., example="2026-01-08 03:24:15")
    amt: float = Field(..., gt=0, example=15420.0,
                       description="Monto de la transacción (debe ser positivo)")
    category: str = Field(..., example="shopping_net")
    gender: str = Field(..., example="F")
    job: str = Field(..., example="Scientist")
    city_pop: int = Field(..., ge=0, example=15000,
                          description="Población de la ciudad (0 si desconocida)")
    dob: str = Field(..., example="1985-01-15")

    # Coordenadas del cliente — ge=0 permite fallback Java con 0.0
    lat: float = Field(..., ge=-90.0, le=90.0, example=-12.0463)
    long: float = Field(..., ge=-180.0, le=180.0, example=-77.0427)

    # Coordenadas del comercio — mismo criterio
    merch_lat: float = Field(..., ge=-90.0, le=90.0, example=-13.1631)
    merch_long: float = Field(..., ge=-180.0, le=180.0, example=-74.2239)


# ---------------------------------------------------------------------------
# RESPONSE
# ---------------------------------------------------------------------------

class RiskFactor(BaseModel):
    """Factor de riesgo explicado por SHAP."""

    feature_name: str
    feature_value: str
    shap_value: float
    risk_description: str
    impact_direction: str  # "AUMENTA_RIESGO" | "DISMINUYE_RIESGO"


class AuditData(BaseModel):
    """
    Datos de auditoría de la predicción.
    Tipo fijo (antes era Dict[str, Any]) para garantizar contrato de API.
    """

    xgboost_score: float
    iforest_score: float
    base_score: float = 0.0
    threshold_used: float
    unknown_encoder_values: Optional[Dict[str, str]] = None


class FraudOutput(BaseModel):
    """Respuesta de predicción de fraude para una transacción."""

    transaction_id: str
    veredicto: str              # "ALTO RIESGO" | "LEGÍTIMO" | "ERROR"
    score_final: float
    detalles_riesgo: List[RiskFactor]
    datos_auditoria: AuditData
    recomendacion: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# BATCH
# ---------------------------------------------------------------------------

class BatchFraudInput(BaseModel):
    """Request para predicciones en lote."""

    transactions: List[FraudInput] = Field(
        ..., description="Lista de transacciones a procesar"
    )


class BatchFraudOutput(BaseModel):
    """Response de predicciones en lote."""

    total_processed: int
    total_frauds: int
    total_legitimate: int
    total_errors: int
    results: List[FraudOutput]
