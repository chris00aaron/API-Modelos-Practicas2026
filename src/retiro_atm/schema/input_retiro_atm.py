from pydantic import BaseModel, Field

class InputDataRetiroAtm(BaseModel):
    diaSemana: int = Field(
        ...,
        examples=[3],
        description="Día de la semana (1=Lunes ... 7=Domingo)",
        ge=1,
        le=7
    )
    tendencia_lags: float = Field(
        ...,
        examples=[4133.0],
        description="Tendencia general estimada a partir de lags previos",
        ge=0
    )
    lag1: float = Field(
        ...,
        examples=[13675.0],
        description="Retiro del día anterior",
        ge=0
    )
    lag5: float = Field(
        ...,
        examples=[16423.0],
        description="Retiro de hace 5 días",
        ge=0
    )
    lag11: float = Field(
        ...,
        examples=[9542.0],
        description="Retiro de hace 11 días",
        ge=0
    )
    caida_reciente: int = Field(
        ...,
        examples=[1],
        description="Indicador de caída reciente (1 = sí, 0 = no)",
        ge=0,
        le=1
    )
    retiros_finde_anterior: float = Field(
        ...,
        examples=[16016.21],
        description="Promedio de retiros del fin de semana anterior",
        ge=0
    )
    retiros_domingo_anterior: float = Field(
        ...,
        examples=[19856.0],
        description="Retiro del domingo anterior",
        ge=0
    )
    ratio_finde_vs_semana: float = Field(
        ...,
        examples=[1.057],
        description="Ratio entre retiros de fin de semana vs semana",
        ge=0
    )
    domingo_bajo: int = Field(
        ...,
        examples=[0],
        description="Indicador de domingo bajo (1 = bajo, 0 = normal)",
        ge=0,
        le=1
    )
    ubicacion: int = Field(
        ...,
        examples=[1],
        description="Tipo de ubicación del ATM (1=urbano, 2=rural.)"
    )
    ambiente: int = Field(
        ...,
        examples=[0],
        description="Ambiente donde se ubica el ATM"
    )