# src/morosidad/schema/morosidad_schema.py
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


class MorosidadResponse(BaseModel):
    """
    Schema de salida para predicción de morosidad.
    """
    default: bool = Field(..., description="¿Habrá incumplimiento de pago?")
    probabilidad_default: float = Field(..., description="Probabilidad de incumplimiento (0.0 - 1.0)")

    class Config:
        json_schema_extra = {
            "example": {
                "default": True,
                "probabilidad_default": 0.75
            }
        }
