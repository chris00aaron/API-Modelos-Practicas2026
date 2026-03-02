import pandas as pd
import numpy as np

# Import from DagsHub loader instead of local files
from fuga.models_files.loader import obtener_modelo, obtener_scaler, obtener_feature_names

class ChurnService:
    def __init__(self):
        # Load from DagsHub (downloads once and caches in memory)
        self.model = obtener_modelo()
        self.scaler = obtener_scaler()
        self.feature_names = obtener_feature_names()
        self.model_version = "v1.0.0"  # M8: Updated via hot-reload after training

    def preprocess_data(self, input_dict: dict):
        """
        Aquí replicamos EXACTAMENTE la lógica de tu función load_and_preprocess
        del Colab.
        """
        # 1. Convertir dict a DataFrame
        df = pd.DataFrame([input_dict])

        # ---------------------------------------------------------
        # A. INGENIERÍA DE CARACTERÍSTICAS (Tus fórmulas matemáticas)
        # ---------------------------------------------------------
        # Evitamos división por cero usando un valor pequeño (epsilon) o validación
        epsilon = 1e-9
        df['TenureByAge'] = df['Tenure'] / (df['Age'] + epsilon)
        df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + epsilon)
        df['CreditScoreGivenAge'] = df['CreditScore'] / (df['Age'] + epsilon)

        # ---------------------------------------------------------
        # B. CODIFICACIÓN (Encoding)
        # ---------------------------------------------------------
        
        # 1. Gender: LabelEncoder (0: Mujer/Female, 1: Hombre/Male)
        # Ajusta "Male"/"Female" según como vengan tus datos reales
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0, 'Hombre': 1, 'Mujer': 0})

        # 2. Geography: get_dummies(drop_first=True)
        # Como drop_first=True eliminó la primera columna (alfabéticamente France),
        # solo necesitamos crear las columnas Germany y Spain.
        input_geo = input_dict['Geography']
        
        df['Geography_Germany'] = 1 if input_geo == 'Germany' else 0
        df['Geography_Spain'] = 1 if input_geo == 'Spain' else 0
        
        # Eliminamos la columna original de texto 'Geography' ya que ya la procesamos
        if 'Geography' in df.columns:
            df = df.drop(columns=['Geography'])

        # ---------------------------------------------------------
        # C. ORDENAMIENTO Y ESCALADO FINAL
        # ---------------------------------------------------------
        
        # 1. Asegurar que las columnas estén en el MISMO orden que feature_names.pkl
        if self.feature_names:
            # Reindexamos el dataframe con las columnas esperadas. 
            # fill_value=0 rellena cualquier cosa faltante por seguridad.
            df = df.reindex(columns=self.feature_names, fill_value=0)
        
        # 2. Escalar (StandardScaler)
        if self.scaler:
            scaled_data = self.scaler.transform(df)
            return scaled_data
        
        return df

    def predict(self, input_data: dict):
        if not self.model:
            return {"error": "El modelo no está cargado."}
        
        try:
            # Procesamos los datos (Fórmulas + Encoding + Scaling)
            X_processed = self.preprocess_data(input_data)
            
            # Predicción
            prediction = self.model.predict(X_processed)
            probability = self.model.predict_proba(X_processed)
            
            result = int(prediction[0]) 
            prob_churn = float(probability[0][1])
            
            # Calcular nivel de riesgo y confianza
            risk_level = "Bajo"
            if prob_churn > 0.75:
                risk_level = "Alto"
            elif prob_churn > 0.45:
                risk_level = "Medio" # Ajustado para coincidir mejor con dashboard
            
            # Confianza simple: qué tan lejos está del umbral 0.5
            confidence = abs(prob_churn - 0.5) * 2 

            # Generar Explicaciones (XAI Lite)
            risk_factors = self.generate_explanations(input_data, prob_churn)
            
            # Recuperar Ground Truth si existe
            real_exit = input_data.get('Exited')
            is_real_exit = True if real_exit == 1 else False if real_exit == 0 else None

            return {
                "prediction": "Abandona (Churn)" if result == 1 else "Se Queda",
                "churn_probability": round(prob_churn, 4),
                "risk_level": risk_level,
                "is_churn": result,
                "prediction_confidence": round(confidence, 4),
                "model_version": self.model_version,  # M8: Dynamic version from hot-reload
                "risk_factors": risk_factors,
                "real_exit": is_real_exit # Devolvemos la realidad para validación
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc() # Esto te imprimirá el error exacto en la terminal
            return {"error": str(e)}

    def generate_explanations(self, data: dict, probability: float):
        """
        Genera factores de riesgo escalados por feature_importances_ reales del modelo (M5).
        Evalúa las 10 variables del modelo XGBoost para dar una explicación completa.
        Retorna una lista de objetos {feature, impact, type}.
        """
        # M5: Obtener importancias reales del modelo para escalar impactos
        feature_importances = {}
        if self.model and self.feature_names:
            try:
                fi = self.model.feature_importances_
                for fname, fimport in zip(self.feature_names, fi):
                    feature_importances[fname] = float(fimport)
            except Exception:
                pass  # Si falla, usamos impactos base sin escalar
        
        def scale_impact(base_impact: float, feature_key: str) -> int:
            """Escala el impacto base según la importancia real del feature en el modelo."""
            importance = feature_importances.get(feature_key, 0.1)
            # El multiplicador va de 0.5x a 3x según importancia (0.0 a 0.3+)
            multiplier = 0.5 + importance * 8.0
            return round(base_impact * multiplier)
        
        factors = []

        # 1. Edad (Suele ser factor de riesgo si es mayor de 40)
        age = data.get('Age', 0)
        if age > 45:
            factors.append({
                "feature": f"Edad Avanzada ({age} años)",
                "impact": scale_impact(30, 'Age'), 
                "type": "negative"
            })
        elif age > 35:
            factors.append({
                "feature": f"Edad Media ({age} años)",
                "impact": scale_impact(10, 'Age'),
                "type": "negative"
            })
        elif age < 30:
            factors.append({
                "feature": f"Edad Joven ({age} años)",
                "impact": scale_impact(-15, 'Age'), 
                "type": "positive"
            })

        # 2. Miembro Activo
        active = data.get('IsActiveMember', 0)
        if active == 1:
            factors.append({
                "feature": "Cliente Activo",
                "impact": scale_impact(-25, 'IsActiveMember'),
                "type": "positive"
            })
        else:
            factors.append({
                "feature": "Cliente Inactivo",
                "impact": scale_impact(20, 'IsActiveMember'),
                "type": "negative"
            })

        # 3. Balance
        balance = data.get('Balance', 0)
        if balance > 150000:
            factors.append({
                "feature": f"Balance Muy Alto (${balance:,.0f})",
                "impact": scale_impact(25, 'Balance'),
                "type": "negative"
            })
        elif balance > 100000:
             factors.append({
                "feature": f"Balance Alto (${balance:,.0f})",
                "impact": scale_impact(15, 'Balance'),
                "type": "negative"
            })
        elif balance == 0:
             factors.append({
                "feature": "Saldo Cero",
                "impact": scale_impact(10, 'Balance'),
                "type": "negative"
            })
        elif balance > 50000:
            factors.append({
                "feature": f"Balance Saludable (${balance:,.0f})",
                "impact": scale_impact(-10, 'Balance'),
                "type": "positive"
            })

        # 4. Productos
        products = data.get('NumOfProducts', 1)
        if products >= 3:
            factors.append({
                "feature": f"Exceso de Productos ({products})",
                "impact": scale_impact(40, 'NumOfProducts'),
                "type": "negative"
            })
        elif products == 2:
            factors.append({
                "feature": "Vinculación Óptima (2 productos)",
                "impact": scale_impact(-20, 'NumOfProducts'),
                "type": "positive"
            })
        elif products == 1:
            factors.append({
                "feature": "Baja Vinculación (1 producto)",
                "impact": scale_impact(10, 'NumOfProducts'),
                "type": "negative"
            })

        # 5. Score Crediticio
        score = data.get('CreditScore', 600)
        if score < 450:
            factors.append({
                "feature": f"Score Crediticio Muy Bajo ({score})",
                "impact": scale_impact(35, 'CreditScore'),
                "type": "negative"
            })
        elif score < 550:
            factors.append({
                "feature": f"Score Crediticio Bajo ({score})",
                "impact": scale_impact(20, 'CreditScore'),
                "type": "negative"
            })
        elif score > 750:
            factors.append({
                "feature": f"Score Crediticio Excelente ({score})",
                "impact": scale_impact(-15, 'CreditScore'),
                "type": "positive"
            })

        # 6. Geografía (Alemania tiene históricamente mayor tasa de churn)
        geography = data.get('Geography', 'Unknown')
        if geography in ('Germany', 'Alemania'):
            factors.append({
                "feature": f"Geografía de Riesgo ({geography})",
                "impact": scale_impact(25, 'Geography_Germany'),
                "type": "negative"
            })
        elif geography in ('France', 'Francia'):
            factors.append({
                "feature": f"Geografía Estable ({geography})",
                "impact": scale_impact(-10, 'Geography_Germany'),
                "type": "positive"
            })
        elif geography in ('Spain', 'España'):
            factors.append({
                "feature": f"Geografía Favorable ({geography})",
                "impact": scale_impact(-12, 'Geography_Spain'),
                "type": "positive"
            })

        # 7. Género (Mujeres tienen históricamente mayor tasa de churn en este dataset)
        gender = data.get('Gender', 'Unknown')
        if gender == 'Female':
            factors.append({
                "feature": "Género Femenino (mayor riesgo histórico)",
                "impact": scale_impact(15, 'Gender'),
                "type": "negative"
            })
        elif gender == 'Male':
            factors.append({
                "feature": "Género Masculino (menor riesgo histórico)",
                "impact": scale_impact(-8, 'Gender'),
                "type": "positive"
            })

        # 8. Antigüedad (Tenure)
        tenure = data.get('Tenure', 5)
        if tenure <= 1:
            factors.append({
                "feature": f"Cliente Nuevo ({tenure} año{'s' if tenure != 1 else ''})",
                "impact": scale_impact(20, 'Tenure'),
                "type": "negative"
            })
        elif tenure <= 3:
            factors.append({
                "feature": f"Baja Antigüedad ({tenure} años)",
                "impact": scale_impact(10, 'Tenure'),
                "type": "negative"
            })
        elif tenure >= 8:
            factors.append({
                "feature": f"Alta Fidelización ({tenure} años)",
                "impact": scale_impact(-18, 'Tenure'),
                "type": "positive"
            })

        # 9. Tarjeta de Crédito
        has_card = data.get('HasCrCard', 1)
        if has_card == 0:
            factors.append({
                "feature": "Sin Tarjeta de Crédito",
                "impact": scale_impact(12, 'HasCrCard'),
                "type": "negative"
            })

        # 10. Salario Estimado
        salary = data.get('EstimatedSalary', 50000)
        if salary < 30000:
            factors.append({
                "feature": f"Salario Bajo (${salary:,.0f})",
                "impact": scale_impact(15, 'EstimatedSalary'),
                "type": "negative"
            })
        elif salary > 150000:
            factors.append({
                "feature": f"Salario Alto (${salary:,.0f})",
                "impact": scale_impact(-10, 'EstimatedSalary'),
                "type": "positive"
            })

        # Ordenar por impacto absoluto y devolver top 7
        return sorted(factors, key=lambda x: abs(x['impact']), reverse=True)[:7]

churn_service = ChurnService()