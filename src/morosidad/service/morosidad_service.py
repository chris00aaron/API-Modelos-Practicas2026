# src/morosidad/service/morosidad_service.py
import pandas as pd
import numpy as np
from typing import List

from morosidad.models_files import obtener_modelo, obtener_explainer, obtener_version
from morosidad.schema import MorosidadRequest, MorosidadResponse, RiskFactor, BatchMorosidadRequest, BatchItemResponse, BatchMorosidadResponse


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
        MorosidadResponse con el resultado de la predicción y top 5 factores de riesgo.
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
    risk_factors: List[RiskFactor] = []
    
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

            # Normalizar valores SHAP a porcentaje (escala -100 a +100)
            max_abs = np.max(np.abs(vals)) if np.max(np.abs(vals)) > 0 else 1
            normalized = (vals / max_abs) * 100

            # Ordenar por impacto absoluto y tomar top 5
            indices_sorted = np.argsort(np.abs(vals))[::-1][:5]
            
            for idx in indices_sorted:
                impact_value = round(float(normalized[idx]), 1)
                risk_factors.append(RiskFactor(
                    name=COLUMNAS_MODELO[idx],
                    impact=impact_value,
                    direction="positive" if vals[idx] > 0 else "negative"
                ))
            
            # El principal es el primero (mayor impacto absoluto)
            main_risk_factor = COLUMNAS_MODELO[indices_sorted[0]]
        else:
            print("[WARN] Explainer SHAP no disponible, usando 'Unknown' como factor de riesgo")
            
    except Exception as e:
        print(f"[WARN] Error calculando SHAP values: {e}")
        main_risk_factor = "Error calculating risk factor"
    
    return MorosidadResponse(
        default=bool(prediccion == 1),
        probabilidad_default=probabilidad_default,
        main_risk_factor=main_risk_factor,
        risk_factors=risk_factors,
        model_version=obtener_version()
    )


def predecir_morosidad_batch(request: BatchMorosidadRequest) -> BatchMorosidadResponse:
    """
    Realiza predicción de morosidad en lote (vectorizado).
    Mucho más eficiente que llamar a predecir_morosidad individualmente.
    
    Args:
        request: Request con lista de items y opción include_shap.
    
    Returns:
        BatchMorosidadResponse con todas las predicciones y resumen SHAP si se solicitó.
    """
    requests = request.items
    
    if not requests:
        return BatchMorosidadResponse(
            predictions=[],
            model_version=obtener_version(),
            total_processed=0
        )
    
    # Obtener el modelo
    modelo = obtener_modelo()
    
    # Crear DataFrame con TODAS las filas de una vez (vectorización)
    datos = [req.model_dump() for req in requests]
    df = pd.DataFrame(datos, columns=COLUMNAS_MODELO)
    
    # Asegurar tipos numéricos
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Predicción vectorizada (muy rápida)
    predicciones = modelo.predict(df)
    probabilidades = modelo.predict_proba(df)[:, 1]  # Probabilidad de clase 1
    
    # Construir respuestas individuales
    results = []
    for i in range(len(requests)):
        results.append(BatchItemResponse(
            index=i,
            default=bool(predicciones[i] == 1),
            probabilidad_default=float(probabilidades[i]),
            main_risk_factor="Batch"  # Simplificado para velocidad
        ))
        
    # Lógica SHAP agregada (si se solicita)
    shap_summary = None
    if request.include_shap:
        try:
            explainer = obtener_explainer()
            if explainer:
                # 1. Muestreo (máximo 100 registros para velocidad)
                if len(df) <= 100:
                    X_sample = df.values.astype(np.float64)
                else:
                    sample_idx = np.random.choice(len(df), 100, replace=False)
                    X_sample = df.iloc[sample_idx].values.astype(np.float64)
                
                # 2. Calcular valores SHAP
                shap_values_raw = explainer.shap_values(X_sample)
                
                # Manejar formato (lista para clasificación binaria o array)
                if isinstance(shap_values_raw, list):
                    vals = shap_values_raw[1] # Clase 1
                else:
                    vals = shap_values_raw
                    
                # 3. Calcular impacto medio absoluto por feature
                # vals shape: (n_samples, n_features)
                mean_abs = np.mean(np.abs(vals), axis=0)
                
                # 4. Obtener Top 10
                top_idx = np.argsort(mean_abs)[::-1][:10]
                
                # 5. Normalizar
                max_val = np.max(mean_abs) if np.max(mean_abs) > 0 else 1.0
                
                summary = []
                for idx in top_idx:
                    # Determinar dirección predominante (media con signo)
                    mean_signed = np.mean(vals[:, idx])
                    
                    summary.append(RiskFactor(
                        name=COLUMNAS_MODELO[idx],
                        impact=round((float(mean_abs[idx]) / max_val) * 100, 1),
                        direction="positive" if mean_signed > 0 else "negative"
                    ))
                
                shap_summary = summary
            else:
                print("[WARN] SHAP Explainer no disponible para batch")
                
        except Exception as e:
            print(f"[ERROR] Fallo cálculo SHAP batch: {e}")
            # Continuar sin SHAP, no fallar todo el batch
    
    return BatchMorosidadResponse(
        predictions=results,
        shap_summary=shap_summary,
        model_version=obtener_version(),
        total_processed=len(requests)
    )

