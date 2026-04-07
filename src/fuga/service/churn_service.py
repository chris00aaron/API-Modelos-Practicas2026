import pandas as pd
import numpy as np

# Import from DagsHub loader instead of local files
from fuga.models_files.loader import obtener_modelo, obtener_scaler, obtener_feature_names, obtener_explainer, obtener_version

class ChurnService:
    def __init__(self):
        # Load from DagsHub (downloads once and caches in memory)
        self.model = obtener_modelo()
        self.scaler = obtener_scaler()
        self.feature_names = obtener_feature_names()
        self.explainer = obtener_explainer()
        self.model_version = obtener_version()  # Reads actual version from loader (e.g. v_1772556687)

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
            # Usamos predict_proba + umbral de negocio (0.50) alineado con
            # el umbral por defecto del modelo entrenado en Colab.
            DECISION_THRESHOLD = 0.50
            probability = self.model.predict_proba(X_processed)
            prob_churn  = float(probability[0][1])
            result      = 1 if prob_churn >= DECISION_THRESHOLD else 0

            # Calcular nivel de riesgo y confianza
            risk_level = "Bajo"
            if prob_churn > 0.75:
                risk_level = "Alto"
            elif prob_churn >= DECISION_THRESHOLD:
                risk_level = "Medio"

            # Confianza: qué tan lejos está del umbral de decisión (acotado a [0, 1])
            confidence = min(1.0, abs(prob_churn - DECISION_THRESHOLD) * 2)

            # Generar Explicaciones (SHAP)
            risk_factors = self.generate_explanations(input_data, prob_churn, X_processed)
            
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

    # Mapa de nombre de feature procesado → etiqueta legible
    # Recibe el feature_name y el dict original de input para mostrar valores reales
    _FEATURE_LABELS = {
        'Age':                 lambda d: f"Edad ({d.get('Age', '?')} años)",
        'IsActiveMember':      lambda d: "Cliente Activo" if d.get('IsActiveMember') == 1 else "Cliente Inactivo",
        'Balance':             lambda d: f"Balance (€{d.get('Balance', 0):,.0f})",
        'NumOfProducts':       lambda d: f"N° Productos ({d.get('NumOfProducts', '?')})",
        'CreditScore':         lambda d: f"Score Crediticio ({d.get('CreditScore', '?')})",
        'Geography_Germany':   lambda d: "Geografía: Alemania",
        'Geography_Spain':     lambda d: "Geografía: España",
        'Gender':              lambda d: f"Género ({d.get('Gender', '?')})",
        'Tenure':              lambda d: f"Antigüedad ({d.get('Tenure', '?')} años)",
        'HasCrCard':           lambda d: "Con Tarjeta de Crédito" if d.get('HasCrCard') == 1 else "Sin Tarjeta de Crédito",
        'EstimatedSalary':     lambda d: f"Salario Est. (€{d.get('EstimatedSalary', 0):,.0f})",
        'TenureByAge':         lambda d: "Ratio Antigüedad/Edad",
        'BalanceSalaryRatio':  lambda d: "Ratio Balance/Salario",
        'CreditScoreGivenAge': lambda d: "Score Crediticio por Edad",
    }

    def generate_explanations(self, data: dict, probability: float, X_processed=None):
        """
        Genera factores de riesgo usando SHAP TreeExplainer sobre el modelo XGBoost.
        Si SHAP no está disponible, usa el sistema de reglas como fallback.
        Retorna una lista de objetos {feature, impact, type} — top 7 por |impacto|.

        - impact > 0  → aumenta riesgo de fuga   → type: "negative"
        - impact < 0  → reduce  riesgo de fuga   → type: "positive"
        Los valores se normalizan a escala -100..+100.
        """
        if self.explainer is not None and X_processed is not None and self.feature_names:
            try:
                shap_values = self.explainer.shap_values(X_processed)

                # XGBoost puede devolver lista [clase0, clase1] o array directo
                if isinstance(shap_values, list):
                    row = shap_values[1][0]   # clase 1 (churn), primera fila
                elif len(shap_values.shape) == 2:
                    row = shap_values[0]
                else:
                    row = shap_values

                # Normalizar a -100..+100 (clip garantiza que no se salga del rango)
                max_abs = np.max(np.abs(row)) if np.max(np.abs(row)) > 0 else 1.0
                normalized = np.clip((row / max_abs) * 100.0, -100.0, 100.0)

                factors = []
                for feat, shap_val in zip(self.feature_names, normalized):
                    if abs(shap_val) < 0.5:   # filtrar ruido
                        continue
                    label_fn = self._FEATURE_LABELS.get(feat)
                    label = label_fn(data) if label_fn else feat
                    factors.append({
                        "feature": label,
                        "impact":  round(float(shap_val), 2),
                        "type":    "negative" if shap_val > 0 else "positive",
                    })

                return sorted(factors, key=lambda x: abs(x['impact']), reverse=True)[:7]

            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[CHURN XAI] SHAP falló ({e}), usando fallback de reglas.")

        # ── Fallback: sistema de reglas (se usa si SHAP no está disponible) ──────
        feature_importances = {}
        if self.model and self.feature_names:
            try:
                fi = self.model.feature_importances_
                for fname, fimport in zip(self.feature_names, fi):
                    feature_importances[fname] = float(fimport)
            except Exception:
                pass

        def scale_impact(base_impact: float, feature_key: str) -> int:
            importance = feature_importances.get(feature_key, 0.1)
            multiplier = 0.5 + importance * 8.0
            return round(base_impact * multiplier)

        factors = []
        age = data.get('Age', 0)
        if age > 45:
            factors.append({"feature": f"Edad Avanzada ({age} años)", "impact": scale_impact(30, 'Age'), "type": "negative"})
        elif age > 35:
            factors.append({"feature": f"Edad Media ({age} años)", "impact": scale_impact(10, 'Age'), "type": "negative"})
        elif age < 30:
            factors.append({"feature": f"Edad Joven ({age} años)", "impact": scale_impact(-15, 'Age'), "type": "positive"})

        active = data.get('IsActiveMember', 0)
        if active == 1:
            factors.append({"feature": "Cliente Activo", "impact": scale_impact(-25, 'IsActiveMember'), "type": "positive"})
        else:
            factors.append({"feature": "Cliente Inactivo", "impact": scale_impact(20, 'IsActiveMember'), "type": "negative"})

        balance = data.get('Balance', 0)
        if balance > 150000:
            factors.append({"feature": f"Balance Muy Alto (€{balance:,.0f})", "impact": scale_impact(25, 'Balance'), "type": "negative"})
        elif balance > 100000:
            factors.append({"feature": f"Balance Alto (€{balance:,.0f})", "impact": scale_impact(15, 'Balance'), "type": "negative"})
        elif balance == 0:
            factors.append({"feature": "Saldo Cero", "impact": scale_impact(10, 'Balance'), "type": "negative"})
        elif balance > 50000:
            factors.append({"feature": f"Balance Saludable (€{balance:,.0f})", "impact": scale_impact(-10, 'Balance'), "type": "positive"})

        products = data.get('NumOfProducts', 1)
        if products >= 3:
            factors.append({"feature": f"Exceso de Productos ({products})", "impact": scale_impact(40, 'NumOfProducts'), "type": "negative"})
        elif products == 2:
            factors.append({"feature": "Vinculación Óptima (2 productos)", "impact": scale_impact(-20, 'NumOfProducts'), "type": "positive"})
        elif products == 1:
            factors.append({"feature": "Baja Vinculación (1 producto)", "impact": scale_impact(10, 'NumOfProducts'), "type": "negative"})

        score = data.get('CreditScore', 600)
        if score < 450:
            factors.append({"feature": f"Score Crediticio Muy Bajo ({score})", "impact": scale_impact(35, 'CreditScore'), "type": "negative"})
        elif score < 550:
            factors.append({"feature": f"Score Crediticio Bajo ({score})", "impact": scale_impact(20, 'CreditScore'), "type": "negative"})
        elif score > 750:
            factors.append({"feature": f"Score Crediticio Excelente ({score})", "impact": scale_impact(-15, 'CreditScore'), "type": "positive"})

        geography = data.get('Geography', 'Unknown')
        if geography in ('Germany', 'Alemania'):
            factors.append({"feature": f"Geografía de Riesgo ({geography})", "impact": scale_impact(25, 'Geography_Germany'), "type": "negative"})
        elif geography in ('France', 'Francia'):
            factors.append({"feature": f"Geografía Estable ({geography})", "impact": scale_impact(-10, 'Geography_Germany'), "type": "positive"})
        elif geography in ('Spain', 'España'):
            factors.append({"feature": f"Geografía Favorable ({geography})", "impact": scale_impact(-12, 'Geography_Spain'), "type": "positive"})

        gender = data.get('Gender', 'Unknown')
        if gender == 'Female':
            factors.append({"feature": "Género Femenino (mayor riesgo histórico)", "impact": scale_impact(15, 'Gender'), "type": "negative"})
        elif gender == 'Male':
            factors.append({"feature": "Género Masculino (menor riesgo histórico)", "impact": scale_impact(-8, 'Gender'), "type": "positive"})

        tenure = data.get('Tenure', 5)
        if tenure <= 1:
            factors.append({"feature": f"Cliente Nuevo ({tenure} año{'s' if tenure != 1 else ''})", "impact": scale_impact(20, 'Tenure'), "type": "negative"})
        elif tenure <= 3:
            factors.append({"feature": f"Baja Antigüedad ({tenure} años)", "impact": scale_impact(10, 'Tenure'), "type": "negative"})
        elif tenure >= 8:
            factors.append({"feature": f"Alta Fidelización ({tenure} años)", "impact": scale_impact(-18, 'Tenure'), "type": "positive"})

        has_card = data.get('HasCrCard', 1)
        if has_card == 0:
            factors.append({"feature": "Sin Tarjeta de Crédito", "impact": scale_impact(12, 'HasCrCard'), "type": "negative"})

        salary = data.get('EstimatedSalary', 50000)
        if salary < 30000:
            factors.append({"feature": f"Salario Bajo (€{salary:,.0f})", "impact": scale_impact(15, 'EstimatedSalary'), "type": "negative"})
        elif salary > 150000:
            factors.append({"feature": f"Salario Alto (€{salary:,.0f})", "impact": scale_impact(-10, 'EstimatedSalary'), "type": "positive"})

        # Normalizar fallback a -100..+100
        if factors:
            max_abs = max(abs(f['impact']) for f in factors) or 1
            for f in factors:
                f['impact'] = round(f['impact'] / max_abs * 100, 2)
        return sorted(factors, key=lambda x: abs(x['impact']), reverse=True)[:7]

    def preprocess_batch(self, input_list: list[dict]):
        """
        Versión vectorizada de preprocess_data para múltiples registros.
        Recibe una lista de dicts (sin el campo 'id') y devuelve la matriz
        escalada lista para model.predict / predict_proba.
        """
        df = pd.DataFrame(input_list)

        epsilon = 1e-9
        df['TenureByAge']         = df['Tenure'] / (df['Age'] + epsilon)
        df['BalanceSalaryRatio']  = df['Balance'] / (df['EstimatedSalary'] + epsilon)
        df['CreditScoreGivenAge'] = df['CreditScore'] / (df['Age'] + epsilon)

        df['Gender'] = df['Gender'].map(
            {'Male': 1, 'Female': 0, 'Hombre': 1, 'Mujer': 0}
        ).fillna(1).astype(int)

        df['Geography_Germany'] = (df['Geography'] == 'Germany').astype(int)
        df['Geography_Spain']   = (df['Geography'] == 'Spain').astype(int)
        df = df.drop(columns=['Geography'], errors='ignore')

        if self.feature_names:
            df = df.reindex(columns=self.feature_names, fill_value=0)

        if self.scaler:
            return self.scaler.transform(df)
        return df.values

    def predict_batch(self, items: list[dict]):
        """
        Predice la fuga para una lista de clientes en una sola llamada
        vectorizada al modelo. No genera explicaciones SHAP (uso batch).
        Cada item debe tener 'id' + los campos de ChurnBatchItem.
        Retorna una lista de dicts con id, churn_probability, risk_level,
        is_churn, prediction_confidence y model_version.
        """
        if not self.model:
            return {"error": "El modelo no está cargado."}

        try:
            ids        = [item['id'] for item in items]
            input_data = [{k: v for k, v in item.items() if k != 'id'} for item in items]

            DECISION_THRESHOLD = 0.50
            X     = self.preprocess_batch(input_data)
            probs = self.model.predict_proba(X)[:, 1]

            results = []
            for i, prob in enumerate(probs):
                prob       = float(prob)
                pred       = 1 if prob >= DECISION_THRESHOLD else 0
                risk_level = "Alto" if prob > 0.75 else "Medio" if prob >= DECISION_THRESHOLD else "Bajo"
                confidence = round(min(1.0, abs(prob - DECISION_THRESHOLD) * 2), 4)
                results.append({
                    "id":                   ids[i],
                    "churn_probability":    round(prob, 4),
                    "risk_level":           risk_level,
                    "is_churn":             int(pred),
                    "prediction_confidence": confidence,
                    "model_version":        self.model_version,
                })
            return results

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}


churn_service = ChurnService()