# src/morosidad/service/morosidad_service.py
import pandas as pd
import numpy as np

from morosidad.models_files import obtener_modelo, obtener_explainer, obtener_version
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
        MorosidadResponse con el resultado de la predicción y factor de riesgo principal.
    """
    # Obtener el modelo y explainer
    modelo = obtener_modelo()
    explainer = obtener_explainer()
    
    # Convertir request a diccionario y luego a DataFrame
    datos = request.model_dump()
    df = pd.DataFrame([datos], columns=COLUMNAS_MODELO)
    
    # Asegurar que todos los valores sean numéricos
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Realizar predicción
    prediccion = modelo.predict(df)[0]
    probabilidades = modelo.predict_proba(df)[0]
    
    # La probabilidad de default es la probabilidad de clase 1
    probabilidad_default = float(probabilidades[1])

    # Calcular SHAP values para la interpretabilidad
    main_risk_factor = "Unknown"
    
    try:
        if explainer:
            # Convertir a numpy array para compatibilidad
            X_shap = df.values.astype(np.float64)
            
            # Calcular valores SHAP para esta instancia
            shap_values = explainer.shap_values(X_shap)
            
            # Manejar diferentes formatos de salida de SHAP
            if isinstance(shap_values, list):
                # Para clasificación binaria: [clase_0, clase_1]
                vals = shap_values[1][0]  # Fila 0, Clase 1 (Default)
            else:
                # Array directo
                if len(shap_values.shape) > 1:
                    vals = shap_values[0]
                else:
                    vals = shap_values

            # Encontrar el índice del valor absoluto máximo (mayor impacto)
            idx_max_impact = np.argmax(np.abs(vals))
            main_risk_factor = COLUMNAS_MODELO[idx_max_impact]
        else:
            print("[WARN] Explainer SHAP no disponible, usando 'Unknown' como factor de riesgo")
            
    except Exception as e:
        print(f"[WARN] Error calculando SHAP values: {e}")
        main_risk_factor = "Error calculating risk factor"
    
    return MorosidadResponse(
        default=bool(prediccion == 1),
        probabilidad_default=probabilidad_default,
        main_risk_factor=main_risk_factor,
        model_version=obtener_version()
    )

