import joblib
import pandas as pd
import numpy as np
import os

# Rutas dinámicas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models_files", "best_model_churn.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models_files", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models_files", "feature_names.pkl")

class ChurnService:
    def __init__(self):
        self.model = self._load_file(MODEL_PATH)
        self.scaler = self._load_file(SCALER_PATH)
        self.feature_names = self._load_file(FEATURES_PATH)

    def _load_file(self, path):
        try:
            if os.path.exists(path):
                print(f"✅ Cargado: {path}")
                return joblib.load(path)
            else:
                print(f"❌ Error: No se encuentra {path}")
                return None
        except Exception as e:
            print(f"❌ Error cargando {path}: {e}")
            return None

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
                "model_version": "v1.0.0",
                "risk_factors": risk_factors,
                "real_exit": is_real_exit # Devolvemos la realidad para validación
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc() # Esto te imprimirá el error exacto en la terminal
            return {"error": str(e)}

    def generate_explanations(self, data: dict, probability: float):
        """
        Genera factores de riesgo basados en la lógica del negocio y el modelo.
        Retorna una lista de objetos {feature, impact, type}.
        """
        factors = []

        # 1. Edad (Suele ser factor de riesgo si es mayor de 40)
        age = data.get('Age', 0)
        if age > 45:
            factors.append({
                "feature": f"Edad Avanzada ({age})",
                "impact": 30, 
                "type": "negative" # Negative para el cliente = Aumenta Riesgo de Fuga
            })
        elif age < 30:
            factors.append({
                "feature": "Edad Joven",
                "impact": -15, 
                "type": "positive" # Positive = Ayuda a retener
            })

        # 2. Miembro Activo
        active = data.get('IsActiveMember', 0)
        if active == 1:
            factors.append({
                "feature": "Cliente Activo",
                "impact": -25,
                "type": "positive"
            })
        else:
            factors.append({
                "feature": "Inactividad",
                "impact": 20,
                "type": "negative"
            })

        # 3. Balance
        balance = data.get('Balance', 0)
        if balance > 100000:
             factors.append({
                "feature": "Balance Alto",
                "impact": 15, # A veces saldos altos son más volátiles
                "type": "negative"
            })
        elif balance == 0:
             factors.append({
                "feature": "Saldo Cero",
                "impact": 10,
                "type": "negative"
            })

        # 4. Productos
        products = data.get('NumOfProducts', 1)
        if products >= 3:
            factors.append({
                "feature": "Exceso Productos",
                "impact": 40, # 3 o 4 productos suelen tener mucha fuga
                "type": "negative"
            })
        elif products == 2:
            factors.append({
                "feature": "Vinculación Óptima",
                "impact": -20,
                "type": "positive"
            })

        # 5. Score Crediticio
        score = data.get('CreditScore', 600)
        if score < 450:
            factors.append({
                "feature": "Score Crediticio Bajo",
                "impact": 35,
                "type": "negative"
            })

        # Ordenar por impacto absoluto y devolver top 5
        return sorted(factors, key=lambda x: abs(x['impact']), reverse=True)[:5]

churn_service = ChurnService()