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
        # Evitamos división por cero sumando un epsilon si es necesario, 
        # o asumimos que Age nunca es 0.
        df['TenureByAge'] = df['Tenure'] / df['Age']
        df['BalanceSalaryRatio'] = df['Balance'] / df['EstimatedSalary']
        df['CreditScoreGivenAge'] = df['CreditScore'] / df['Age']

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
            
            return {
                "prediction": "Abandona (Churn)" if result == 1 else "Se Queda",
                "churn_probability": round(prob_churn, 4),
                "risk_level": "Alto" if prob_churn > 0.45 else "Bajo", # Usando tu umbral de 0.45
                "is_churn": result
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc() # Esto te imprimirá el error exacto en la terminal
            return {"error": str(e)}

churn_service = ChurnService()