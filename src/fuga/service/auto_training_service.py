import pandas as pd
import numpy as np
import os
import io
import time
import joblib
import logging
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
import mlflow
import mlflow.xgboost
from fuga import dagshub_client
from fuga.models_files.loader import cargar_modelos_desde_local

logger = logging.getLogger(__name__)


class AutoTrainingService:
    """
    Servicio de auto-entrenamiento para el modelo de Churn.

    Flujo completo:
    1. Extrae datos de la BD (PostgreSQL)
    2. Preprocesa y genera features
    3. Entrena un XGBClassifier
    4. Evalúa métricas (accuracy, F1, precision, recall, AUC-ROC)
    5. Registra el experimento en MLflow/DagsHub
    6. Sube artefactos (.pkl) al repositorio de DagsHub
    7. Recarga el modelo en memoria (hot-reload)
    """

    def __init__(self):
        # DB connection from environment variables (defaults match application.properties)
        db_name = os.environ.get("DB_NAME", "BankMindDB")
        db_user = os.environ.get("DB_USER", "postgres")
        db_password = os.environ.get("DB_PASSWORD", "1234")
        db_host = os.environ.get("DB_HOST", "localhost")
        db_port = os.environ.get("DB_PORT", "5432")
        self.db_url = f"dbname='{db_name}' user='{db_user}' password='{db_password}' host='{db_host}' port='{db_port}'"

        # Training state (for status polling)
        self.is_training = False
        self.last_result = None

    def _get_data_from_db(self):
        """Obtiene datos de la base de datos para el entrenamiento."""
        query = """
        SELECT 
            ad.credit_score as "CreditScore",
            c.country_description as "Geography",
            g.gender_description as "Gender",
            cu.age as "Age",
            ad.tenure as "Tenure",
            ad.balance as "Balance",
            ad.num_of_products as "NumOfProducts",
            CASE WHEN ad.has_cr_card = true THEN 1 ELSE 0 END as "HasCrCard",
            CASE WHEN ad.is_active_member = true THEN 1 ELSE 0 END as "IsActiveMember",
            ad.estimated_salary as "EstimatedSalary",
            CASE WHEN ad.exited = true THEN 1 ELSE 0 END as "Exited"
        FROM public.account_details ad
        JOIN public.customer cu ON ad.id_customer = cu.id_customer
        JOIN public.country c ON cu.id_country = c.id_country
        JOIN public.gender g ON cu.id_gender = g.id_gender
        """
        conn = None
        try:
            conn = psycopg2.connect(self.db_url)
            df = pd.read_sql(query, conn)
            logger.info(f"Datos recuperados exitosamente: {len(df)} filas.")
            return df
        except psycopg2.OperationalError as e:
            logger.error(f"Error de conexión a la BD: {e}")
            raise RuntimeError(
                f"No se pudo conectar a la base de datos. "
                f"Verifique que PostgreSQL esté activo y las credenciales sean correctas. "
                f"Detalle: {e}"
            )
        except Exception as e:
            logger.error(f"Error al recuperar datos de la DB: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def preprocess_training_data(self, df: pd.DataFrame):
        """Replica la ingeniería de características del ChurnService."""
        df = df.copy()

        # Ingeniería de Características
        epsilon = 1e-9
        df['TenureByAge'] = df['Tenure'] / (df['Age'] + epsilon)
        df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + epsilon)
        df['CreditScoreGivenAge'] = df['CreditScore'] / (df['Age'] + epsilon)

        # Encoding Gender
        df['Gender'] = df['Gender'].map({
            'Male': 1, 'Female': 0,
            'Hombre': 1, 'Mujer': 0
        })
        # Fillna for unmapped genders
        df['Gender'] = df['Gender'].fillna(0).astype(int)

        # Encoding Geography (drop_first=True → eliminar France)
        geography_dummies = pd.get_dummies(df['Geography'], prefix='Geography')

        if 'Geography_Germany' not in geography_dummies.columns:
            geography_dummies['Geography_Germany'] = 0
        if 'Geography_Spain' not in geography_dummies.columns:
            geography_dummies['Geography_Spain'] = 0

        df = pd.concat([df, geography_dummies[['Geography_Germany', 'Geography_Spain']]], axis=1)
        df.drop(columns=['Geography'], inplace=True)

        # Definir target y features
        y = df['Exited'].astype(int)

        # El orden de las features debe ser consistente con churn_service.py
        feature_names = [
            'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'TenureByAge',
            'BalanceSalaryRatio', 'CreditScoreGivenAge', 'Geography_Germany', 'Geography_Spain'
        ]

        X = df[feature_names]

        return X, y, feature_names

    def train_model(self):
        """Ejecuta el flujo completo de auto-entrenamiento."""
        if self.is_training:
            return {"error": "Ya hay un entrenamiento en curso. Espere a que termine."}

        self.is_training = True
        self.last_result = None

        try:
            # 0. Verificar permisos del token DagsHub ANTES de entrenar
            logger.info("=== INICIO AUTO-ENTRENAMIENTO CHURN ===")
            try:
                token_info = dagshub_client.check_token_permissions()
                for msg in token_info.get('messages', []):
                    logger.info(f"[TOKEN CHECK] {msg}")
                if not token_info.get('write'):
                    logger.warning(
                        "⚠️ El token DagsHub NO tiene permisos de escritura. "
                        "El modelo se entrenará pero NO se subirá a DagsHub."
                    )
            except Exception as e_token:
                logger.warning(f"⚠️ No se pudo verificar token DagsHub: {e_token}")

            # 1. Obtener datos
            df_raw = self._get_data_from_db()
            if len(df_raw) < 100:
                result = {
                    "status": "error",
                    "error": f"Insuficientes datos para entrenar (encontrados: {len(df_raw)}, mínimo: 100)."
                }
                self.last_result = result
                return result

            logger.info(f"Datos cargados: {len(df_raw)} registros, {df_raw['Exited'].sum()} positivos (churn).")

            # 2. Preprocesar
            X, y, feature_names = self.preprocess_training_data(df_raw)

            # 3. Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 4. Escalar
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 5. Entrenar modelo con scale_pos_weight dinámico (M4)
            neg = (y_train == 0).sum()
            pos = (y_train == 1).sum()
            scale_pos_weight = neg / pos if pos > 0 else 1.0
            logger.info(f"Class balance: {neg} neg / {pos} pos → scale_pos_weight={scale_pos_weight:.2f}")

            model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            model.fit(X_train_scaled, y_train)

            # 6. Evaluar
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)

            logger.info(
                f"Entrenamiento completado. "
                f"Accuracy: {acc:.4f}, F1: {f1:.4f}, "
                f"Precision: {prec:.4f}, Recall: {rec:.4f}, AUC-ROC: {auc:.4f}"
            )

            # 7. MLflow / DagsHub tracking (OPCIONAL — no bloquea el entrenamiento)
            mlflow_run_id = None
            version_tag = f"v_{int(time.time())}"
            try:
                dagshub_client.init_dagshub_connection()
                mlflow.set_experiment("/churn_auto_training")

                with mlflow.start_run(run_name=f"churn_{version_tag}") as run:
                    mlflow.log_param("n_estimators", 100)
                    mlflow.log_param("learning_rate", 0.1)
                    mlflow.log_param("max_depth", 5)
                    mlflow.log_param("train_samples", len(X_train))
                    mlflow.log_param("test_samples", len(X_test))
                    mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 2))
                    mlflow.log_param("version_tag", version_tag)
                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("f1_score", f1)
                    mlflow.log_metric("precision", prec)
                    mlflow.log_metric("recall", rec)
                    mlflow.log_metric("auc_roc", auc)
                    mlflow.xgboost.log_model(model, "model")
                    mlflow_run_id = run.info.run_id
                    logger.info(f"✅ Experimento registrado en MLflow. Run ID: {mlflow_run_id}")
            except Exception as e_mlflow:
                logger.warning(
                    f"⚠️ MLflow tracking falló (no bloquea el entrenamiento): {e_mlflow}"
                )

            # 8. Empaquetar combo-pack (modelo + scaler + features + metadata)
            combo_pack = {
                'modelo_prediccion': model,
                'scaler': scaler,
                'feature_names': feature_names,
                'meta_info': {
                    'version': version_tag,
                    'accuracy': round(acc, 4),
                    'f1_score': round(f1, 4),
                    'precision': round(prec, 4),
                    'recall': round(rec, 4),
                    'auc_roc': round(auc, 4),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'mlflow_run_id': mlflow_run_id,
                    'descripcion': 'Modelo CHURN XGBoost con scaler y features'
                }
            }

            # 9. Hot-reload: actualizar modelos en memoria inmediatamente
            cargar_modelos_desde_local(model, scaler, feature_names)

            try:
                from fuga.service.churn_service import churn_service as _cs
                _cs.model = model
                _cs.scaler = scaler
                _cs.feature_names = feature_names
                _cs.model_version = version_tag  # M8: Propagate version to predictions
                logger.info(f"ChurnService actualizado con el nuevo modelo (version: {version_tag}).")
            except Exception as e_reload:
                logger.warning(f"No se pudo actualizar ChurnService en caliente: {e_reload}")

            # 10. Subir combo-pack a DagsHub con verificación de integridad
            dagshub_verified = False
            upload_errors = []
            try:
                buf = io.BytesIO()
                joblib.dump(combo_pack, buf)
                model_bytes = buf.getvalue()

                upload_ok = dagshub_client.upload_champion(model_bytes, version_tag)

                if upload_ok:
                    # Verificar integridad re-descargando
                    integrity_ok = dagshub_client.verify_champion_integrity(version_tag)
                    if integrity_ok:
                        dagshub_verified = True
                        logger.info("✅ Upload + Verificación OK — Modelo listo")
                    else:
                        upload_errors.append("Verificación de integridad falló")

                    # 10b. Subir archivos individuales (legacy) para compatibilidad
                    try:
                        legacy_ok = dagshub_client.upload_individual_files(
                            model, scaler, feature_names, version_tag
                        )
                        if legacy_ok:
                            logger.info("✅ Archivos legacy subidos correctamente")
                        else:
                            upload_errors.append("Algunos archivos legacy no se subieron")
                    except Exception as e_legacy:
                        upload_errors.append(f"Error subiendo archivos legacy: {e_legacy}")
                        logger.warning(f"⚠️ Error subiendo archivos legacy: {e_legacy}")
                else:
                    upload_errors.append("Upload a DagsHub falló")
            except Exception as e_upload:
                upload_errors.append(f"Error en upload: {e_upload}")
                logger.error(f"❌ Error en upload: {e_upload}")

            result = {
                "status": "success",
                "metrics": {
                    "accuracy": round(acc, 4),
                    "f1_score": round(f1, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "auc_roc": round(auc, 4)
                },
                "run_id": mlflow_run_id,
                "version_tag": version_tag,
                "dagshub_verified": dagshub_verified,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "message": "Modelo entrenado y guardado exitosamente."
            }

            if mlflow_run_id:
                result["message"] += " Registrado en MLflow/DagsHub."
            else:
                result["mlflow_warning"] = "MLflow tracking no disponible. El modelo se entrenó y guardó correctamente sin tracking."

            if dagshub_verified:
                result["message"] += " Modelo verificado en DagsHub."
            elif upload_errors:
                result["upload_warnings"] = upload_errors
                result["message"] += f" ({len(upload_errors)} advertencias de carga a DagsHub)"

            # 11. Persistir métricas en la BD (churn_training_history)
            self._persist_training_to_db(result)

            self.last_result = result
            logger.info("=== AUTO-ENTRENAMIENTO FINALIZADO EXITOSAMENTE ===")
            return result

        except Exception as e:
            logger.error(f"Error en auto-entrenamiento: {e}")
            import traceback
            traceback.print_exc()
            result = {"status": "error", "error": str(e)}
            self.last_result = result
            return result
        finally:
            self.is_training = False

    def get_status(self):
        """Retorna el estado actual del entrenamiento."""
        return {
            "is_training": self.is_training,
            "last_result": self.last_result
        }

    def _persist_training_to_db(self, result: dict):
        """Persiste las métricas del entrenamiento en la tabla churn_training_history."""
        import json
        from datetime import datetime

        metrics = result.get("metrics", {})
        # Usar version_tag como identificador principal, fallback a run_id
        version_tag = result.get("version_tag")
        run_id = result.get("run_id")
        model_version = version_tag[:100] if version_tag else (run_id[:100] if run_id else None)

        # Warnings en JSONB separado
        warnings_json = {}
        if result.get("mlflow_warning"):
            warnings_json["mlflow_warning"] = result["mlflow_warning"]
        if result.get("upload_warnings"):
            warnings_json["upload_warnings"] = result["upload_warnings"]

        insert_query = """
        INSERT INTO public.churn_training_history
            (training_date, trigger_reason, in_production, model_version,
             accuracy, f1_score, precision_score, recall_score, auc_roc,
             train_samples, test_samples, warnings)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        conn = None
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            cur.execute(
                insert_query,
                (
                    datetime.now(),
                    "manual_training",
                    True,
                    model_version,
                    metrics.get("accuracy"),
                    metrics.get("f1_score"),
                    metrics.get("precision"),
                    metrics.get("recall"),
                    metrics.get("auc_roc"),
                    result.get("train_samples"),
                    result.get("test_samples"),
                    json.dumps(warnings_json) if warnings_json else None,
                ),
            )
            conn.commit()
            cur.close()
            logger.info("✅ Métricas de entrenamiento guardadas en churn_training_history.")
        except Exception as e:
            logger.error(f"❌ Error guardando métricas en BD: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()




# Singleton (una sola instancia)
auto_training_service = AutoTrainingService()
