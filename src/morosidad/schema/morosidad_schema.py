# src/morosidad/schema/morosidad_schema.py
from typing import List, Optional
from pydantic import BaseModel, Field


class MorosidadRequest(BaseModel):
    """
    Schema de entrada para predicción de morosidad.
    Contiene las 24 features requeridas por el modelo.
    """
    LIMIT_BAL: float = Field(..., description="Límite de crédito")
    SEX: int = Field(..., description="Sexo (1=masculino, 2=femenino)")
    EDUCATION: int = Field(..., description="Nivel educativo (1=postgrado, 2=universidad, 3=preparatoria, 4=otros)")
    MARRIAGE: int = Field(..., description="Estado civil (1=casado, 2=soltero, 3=otros)")
    AGE: int = Field(..., description="Edad en años")
    
    # Estado de pago de los últimos 6 meses
    PAY_0: int = Field(..., description="Estado de pago en septiembre")
    PAY_2: int = Field(..., description="Estado de pago en agosto")
    PAY_3: int = Field(..., description="Estado de pago en julio")
    PAY_4: int = Field(..., description="Estado de pago en junio")
    PAY_5: int = Field(..., description="Estado de pago en mayo")
    PAY_6: int = Field(..., description="Estado de pago en abril")
    
    # Monto de factura de los últimos 6 meses
    BILL_AMT1: float = Field(..., description="Monto de factura en septiembre")
    BILL_AMT2: float = Field(..., description="Monto de factura en agosto")
    BILL_AMT3: float = Field(..., description="Monto de factura en julio")
    BILL_AMT4: float = Field(..., description="Monto de factura en junio")
    BILL_AMT5: float = Field(..., description="Monto de factura en mayo")
    BILL_AMT6: float = Field(..., description="Monto de factura en abril")
    
    # Monto de pago de los últimos 6 meses
    PAY_AMT1: float = Field(..., description="Monto de pago en septiembre")
    PAY_AMT2: float = Field(..., description="Monto de pago en agosto")
    PAY_AMT3: float = Field(..., description="Monto de pago en julio")
    PAY_AMT4: float = Field(..., description="Monto de pago en junio")
    PAY_AMT5: float = Field(..., description="Monto de pago en mayo")
    PAY_AMT6: float = Field(..., description="Monto de pago en abril")
    
    UTILIZATION_RATE: float = Field(..., description="Tasa de utilización de crédito")

    class Config:
        json_schema_extra = {
            "example": {
                "LIMIT_BAL": 200000,
                "SEX": 2,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 24,
                "PAY_0": 2,
                "PAY_2": 2,
                "PAY_3": -1,
                "PAY_4": -1,
                "PAY_5": -2,
                "PAY_6": -2,
                "BILL_AMT1": 3913,
                "BILL_AMT2": 3102,
                "BILL_AMT3": 689,
                "BILL_AMT4": 0,
                "BILL_AMT5": 0,
                "BILL_AMT6": 0,
                "PAY_AMT1": 0,
                "PAY_AMT2": 689,
                "PAY_AMT3": 0,
                "PAY_AMT4": 0,
                "PAY_AMT5": 0,
                "PAY_AMT6": 0,
                "UTILIZATION_RATE": 0.02
            }
        }


class RiskFactor(BaseModel):
    """Factor de riesgo individual con su impacto SHAP."""
    name: str = Field(..., description="Nombre de la variable")
    impact: float = Field(..., description="Impacto normalizado (-100 a +100)")
    direction: str = Field(..., description="positive = aumenta riesgo, negative = reduce riesgo")


class MorosidadResponse(BaseModel):
    """
    Schema de salida para predicción de morosidad.
    """
    default: bool = Field(..., description="¿Habrá incumplimiento de pago?")
    probabilidad_default: float = Field(..., description="Probabilidad de incumplimiento (0.0 - 1.0)")
    main_risk_factor: str = Field(..., description="Factor de riesgo principal (feature más influyente)")
    risk_factors: List[RiskFactor] = Field(default=[], description="Top 5 factores de riesgo con impacto")
    model_version: str = Field(..., description="Versión del modelo utilizado para la predicción")

    class Config:
        json_schema_extra = {
            "example": {
                "default": True,
                "probabilidad_default": 0.75,
                "main_risk_factor": "PAY_0",
                "risk_factors": [
                    {"name": "PAY_0", "impact": 45.2, "direction": "positive"},
                    {"name": "UTILIZATION_RATE", "impact": 28.1, "direction": "positive"},
                    {"name": "AGE", "impact": -15.3, "direction": "negative"}
                ]
            }
        }


# ==================== BATCH SCHEMAS ====================

class BatchMorosidadRequest(BaseModel):
    """Schema de entrada para predicción en lote."""
    items: List[MorosidadRequest] = Field(..., description="Lista de clientes para predecir")
    include_shap: bool = Field(default=False, description="¿Incluir análisis SHAP agregado?")


class BatchItemResponse(BaseModel):
    """Respuesta individual dentro del batch."""
    index: int = Field(..., description="Índice original para mantener orden")
    default: bool = Field(..., description="¿Habrá incumplimiento de pago?")
    probabilidad_default: float = Field(..., description="Probabilidad de incumplimiento (0.0 - 1.0)")
    main_risk_factor: str = Field(..., description="Factor de riesgo principal")


class BatchMorosidadResponse(BaseModel):
    """Schema de salida para predicción en lote."""
    predictions: List[BatchItemResponse] = Field(..., description="Lista de predicciones")
    shap_summary: Optional[List[RiskFactor]] = Field(default=None, description="Resumen agregado de factores SHAP (si se solicitó)")
    model_version: str = Field(..., description="Versión del modelo utilizado")
    total_processed: int = Field(..., description="Total de registros procesados")
