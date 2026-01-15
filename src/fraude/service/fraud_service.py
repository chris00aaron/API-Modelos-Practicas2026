import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from fraude.schema.inputs import FraudInput, FraudOutput, RiskFactor

class FraudService:
    def __init__(self):
        # Ruta dinámica al modelo
        self.model_path = os.path.join(os.path.dirname(__file__), '../models_files/fraud_v1.pkl')
        self._load_model()

    def _load_model(self):
        """Carga el modelo y sus componentes en memoria una sola vez (Singleton implícito)"""
        try:
            print(f"Cargando modelo desde: {self.model_path}")
            model_pack = joblib.load(self.model_path)
            self.scaler = model_pack['scaler']
            self.xgb_model = model_pack['model_xgb']
            self.if_model = model_pack['model_if']
            self.encoders = model_pack['encoders']
            print("Modelo de Fraude cargado correctamente.")
        except Exception as e:
            print(f"Error cargando el modelo: {e}")
            raise RuntimeError("No se pudo iniciar el servicio de IA de Fraude")

    def _haversine(self, lon1, lat1, lon2, lat2):
        """Calcula distancia en km entre dos puntos geográficos"""
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        d = 2 * 6371 * np.arcsin(np.sqrt(np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2))
        return d

    def predict(self, input_data: FraudInput) -> FraudOutput:
        try:
            # 1. Convertir Pydantic a DataFrame
            data_dict = input_data.dict()
            df = pd.DataFrame([data_dict])

            # 2. Ingeniería de Características (Feature Engineering)
            # Fechas y Edad
            df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
            df['dob'] = pd.to_datetime(df['dob'])
            df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
            df['hour'] = df['trans_date_trans_time'].dt.hour
            
            # Distancia
            df['distance_km'] = self._haversine(df['long'], df['lat'], df['merch_long'], df['merch_lat'])

            # 3. Codificación (Encoding)
            for col in ['category', 'gender', 'job']:
                encoder = self.encoders[col]
                valor_entrada = str(df[col].iloc[0])
                
                # Verificamos si el valor que llegó existe en el diccionario del encoder
                if valor_entrada in encoder.classes_:
                    df[col] = encoder.transform([valor_entrada])
                else:
                    # CASO: Valor desconocido (ej: "Manager")
                    # Acción: Asignamos el primer valor conocido del encoder para no romper el flujo.
                    # Esto permite que el modelo evalúe la transacción basándose en los otros factores (monto, hora, etc.)
                    valor_por_defecto = encoder.classes_[0] 
                    print(f"Aviso: Valor desconocido '{valor_entrada}' en columna '{col}'. Usando por defecto: '{valor_por_defecto}'")
                    df[col] = encoder.transform([valor_por_defecto])

            # 4. Alineación de columnas con XGBoost
            # Obtenemos nombres exactos que espera el modelo
            cols_entrenamiento = self.xgb_model.get_booster().feature_names
            cols_base = [c for c in cols_entrenamiento if c != 'anomaly_score']
            
            X = df[cols_base].copy().astype(float)

            # 5. Escalado
            cols_to_scale = ['amt', 'city_pop', 'age', 'distance_km', 'hour']
            X[cols_to_scale] = self.scaler.transform(X[cols_to_scale])

            # 6. Isolation Forest (Anomaly Score)
            X['anomaly_score'] = self.if_model.decision_function(X[cols_base])

            # Reordenar final
            X_final = X[cols_entrenamiento]

            # 7. Predicción
            probabilidad = self.xgb_model.predict_proba(X_final)[0][1]
            
            # 8. Reglas de Negocio (Explicabilidad)
            veredicto = "ALTO RIESGO" if probabilidad > 0.5 else "LEGÍTIMO"
            risk_factors = []
            
            # Factor: Horario
            if df['hour'].iloc[0] <= 3 or df['hour'].iloc[0] >= 22:
                risk_factors.append(RiskFactor(
                    factor="Horario Inusual",
                    puntos="+35pts",
                    descripcion=f"Transacción realizada a las {df['hour'].iloc[0]}:00 h (Madrugada/Noche)"
                ))
            
            # Factor: Distancia
            dist = df['distance_km'].iloc[0]
            if dist > 100:
                risk_factors.append(RiskFactor(
                    factor="Distancia Anómala",
                    puntos="+30pts",
                    descripcion=f"Ubicación a {dist:.1f} km del domicilio habitual"
                ))
            
            # Factor: Monto (Umbral ejemplo, idealmente dinámico)
            if data_dict['amt'] > 1000: # Umbral simple de ejemplo
                risk_factors.append(RiskFactor(
                    factor="Monto Elevado",
                    puntos="+22pts",
                    descripcion=f"Monto superior al promedio estándar"
                ))

            # Construir Respuesta
            return FraudOutput(
                transaction_id=input_data.transaction_id,
                veredicto=veredicto,
                score_final=f"{probabilidad*100:.1f}%",
                detalles_riesgo=risk_factors,
                datos_auditoria={
                    "xgboost_score": float(probabilidad),
                    "iforest_score": float(X['anomaly_score'].iloc[0]),
                    "detection_scenario": len(risk_factors) + 1
                },
                recomendacion="Bloquear y Notificar" if veredicto == "ALTO RIESGO" else "Aprobar"
            )

        except Exception as e:
            # En producción, logguear el error real
            print(f"Error en predicción: {e}")
            raise e