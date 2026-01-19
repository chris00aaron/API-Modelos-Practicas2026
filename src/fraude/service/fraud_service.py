import pandas as pd
import numpy as np
import joblib
import os
import shap
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
            self.explainer = model_pack.get('explainer')
            print("Modelo y SHAP Explainer cargados correctamente.")
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

            # Guardamos una copia de los datos crudos/legibles para la descripción
            raw_values = df.iloc[0].to_dict()

            # 2. Ingeniería de Características (Feature Engineering)
            # Fechas y Edad
            df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
            df['dob'] = pd.to_datetime(df['dob'])
            df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
            df['hour'] = df['trans_date_trans_time'].dt.hour
            
            # Distancia
            df['distance_km'] = self._haversine(df['long'], df['lat'], df['merch_long'], df['merch_lat'])

            # Agregamos los calculados a raw_values para mostrarlos si SHAP los elige
            raw_values['age'] = df['age'].iloc[0]
            raw_values['hour'] = df['hour'].iloc[0]
            raw_values['distance_km'] = round(df['distance_km'].iloc[0], 2)

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
            # Guardamos el score calculado en raw_values para que salga en el JSON
            raw_values['anomaly_score'] = round(X['anomaly_score'].iloc[0], 4)

            # Reordenar final
            X_final = X[cols_entrenamiento]

            # 7. Predicción
            probabilidad = self.xgb_model.predict_proba(X_final)[0][1]
            
            # 8. Reglas de Negocio (Explicabilidad)
            veredicto = "ALTO RIESGO" if probabilidad > 0.5 else "LEGÍTIMO"
            risk_factors = []
            
            if self.explainer:
                # Calculamos valores SHAP para esta fila
                shap_values = self.explainer.shap_values(X_final)
                
                # Si es clasificación binaria, shap_values puede ser una lista [array_clase0, array_clase1]
                # o un solo array si es output margin. XGBoost suele dar directo.
                # Validamos forma:
                if isinstance(shap_values, list):
                    # Tomamos la clase 1 (Fraude)
                    shap_vals_row = shap_values[1][0] 
                elif len(shap_values.shape) == 2:
                    shap_vals_row = shap_values[0]
                else:
                    shap_vals_row = shap_values

                feature_names = X_final.columns
                
                # Creamos lista de tuplas (feature, shap_value)
                shap_dict = zip(feature_names, shap_vals_row)
                
                # Ordenamos por valor absoluto (magnitud de impacto)
                # O si prefieres solo mostrar por qué es FRAUDE, ordena descendente (los positivos)
                top_features = sorted(shap_dict, key=lambda x: x[1], reverse=True)[:5] # Top 5 factores que aumentan el riesgo

                for feat_name, shap_val in top_features:
                    # Filtramos: Solo nos interesa mostrar al usuario lo que AUMENTA el riesgo (shap > 0)
                    # o si es legítimo, qué lo hace seguro.
                    # Aquí asumo que queremos explicar el RIESGO:
                    
                    impact = "AUMENTA_RIESGO" if shap_val > 0 else "DISMINUYE_RIESGO"
                    
                    # Obtener valor original legible
                    val_original = raw_values.get(feat_name, "N/A")
                    
                    # Generar descripción dinámica
                    desc = f"El valor de '{feat_name}' ({val_original}) impacta el score en {shap_val:.2f}"
                    
                    # Caso especial: Descripciones más bonitas para campos conocidos
                    if feat_name == 'amt' and shap_val > 0:
                        desc = f"El monto de {val_original} es inusualmente alto para este perfil."
                    elif feat_name == 'distance_km' and shap_val > 0:
                        desc = f"La distancia ({val_original} km) indica una ubicación atípica."
                    elif feat_name == 'hour' and shap_val > 0:
                        desc = f"La hora de transacción ({val_original}h) es sospechosa."

                    risk_factors.append(RiskFactor(
                        feature_name=feat_name,
                        feature_value=str(val_original),
                        shap_value=float(shap_val),
                        risk_description=desc,
                        impact_direction=impact
                    ))

            # Si no hay explainer o falló, fallback a lista vacía o regla manual simple
            if not risk_factors and probabilidad > 0.5:
                 risk_factors.append(RiskFactor(
                     feature_name="Modelo General",
                     feature_value="N/A",
                     shap_value=probabilidad,
                     risk_description="Patrón general sospechoso detectado por XGBoost",
                     impact_direction="AUMENTA_RIESGO"
                 ))

            # Construir Respuesta
            return FraudOutput(
                transaction_id=input_data.transaction_id,
                veredicto=veredicto,
                score_final=float(probabilidad),
                detalles_riesgo=risk_factors,
                datos_auditoria={
                    "xgboost_score": float(probabilidad),
                    "iforest_score": float(X['anomaly_score'].iloc[0]),
                    "base_score": float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.0
                },
                recomendacion="Bloquear y Notificar" if veredicto == "ALTO RIESGO" else "Aprobar"
            )

        except Exception as e:
            # En producción, logguear el error real
            print(f"Error en predicción: {e}")
            raise e