from pydantic import BaseModel, Field
from typing import List, Dict, Any

# --- MODELO DE ENTRADA (Request) ---
class FraudInput(BaseModel):
    transaction_id: str = Field(..., example="TXN-9834")
    id_cliente: str = Field(..., example="CLI-5502")
    trans_date_trans_time: str = Field(..., example="2026-01-08 03:24:15")
    amt: float = Field(..., example=15420.0)
    category: str = Field(..., example="shopping_net")
    gender: str = Field(..., example="F")
    job: str = Field(..., example="Scientist")
    city_pop: int = Field(..., example=15000)
    dob: str = Field(..., example="1985-01-15")
    lat: float = Field(..., example=-12.0463)
    long: float = Field(..., example=-77.0427)
    merch_lat: float = Field(..., example=-13.1631)
    merch_long: float = Field(..., example=-74.2239)

# --- MODELO DE SALIDA (Response) ---
class RiskFactor(BaseModel):
    feature_name: str       # Ej: "amt" (Coincide con BD)
    feature_value: str      # Ej: "15420.0" (Valor original legible)
    shap_value: float       # Ej: 1.25 (Importancia matemática)
    risk_description: str   # Ej: "El monto es inusualmente alto..."
    impact_direction: str   # "AUMENTA_RIESGO" o "DISMINUYE_RIESGO"

class FraudOutput(BaseModel):
    transaction_id: str
    veredicto: str  # "ALTO RIESGO" o "LEGÍTIMO"
    score_final: float  
    detalles_riesgo: List[RiskFactor]
    datos_auditoria: Dict[str, Any]
    recomendacion: str