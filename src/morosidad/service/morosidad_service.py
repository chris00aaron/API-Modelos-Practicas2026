# src/morosidad/service/morosidad_service.py
import pandas as pd

from morosidad.models_files import obtener_modelo
from morosidad.schema import MorosidadRequest, MorosidadResponse


# Orden exacto de las columnas que espera el modelo
COLUMNAS_MODELO = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    'UTILIZATION_RATE'
]


def predecir_morosidad(request: MorosidadRequest) -> MorosidadResponse:
    """
    Realiza la predicción de morosidad usando el modelo cargado.
    
    Args:
        request: Datos de entrada con las 24 features.
    
    Returns:
        MorosidadResponse con el resultado de la predicción.
    """
    # Obtener el modelo
    modelo = obtener_modelo()
    
    # Convertir request a diccionario y luego a DataFrame
    datos = request.model_dump()
    df = pd.DataFrame([datos], columns=COLUMNAS_MODELO)
    
    # Realizar predicción
    prediccion = modelo.predict(df)[0]
    probabilidades = modelo.predict_proba(df)[0]
    
    # La probabilidad de default es la probabilidad de clase 1
    probabilidad_default = float(probabilidades[1])
    
    return MorosidadResponse(
        default=bool(prediccion == 1),
        probabilidad_default=probabilidad_default
    )
