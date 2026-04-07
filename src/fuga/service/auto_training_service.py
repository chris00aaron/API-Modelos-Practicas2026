import pandas as pd
import numpy as np
import os
import io
import time
import joblib
import logging
import psycopg2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
import mlflow
import mlflow.xgboost
from fuga import dagshub_client
from fuga.models_files.loader import cargar_modelos_desde_local

logger = logging.getLogger(__name__)

# Force PostgreSQL to send all messages (including errors) in UTF-8
# so psycopg2 can decode them without UnicodeDecodeError on Windows.
os.environ.setdefault('PGCLIENTENCODING', 'UTF8')

# Mejora mínima en AUC-ROC (o F1 en desempate) para promover el challenger.
# Evita reemplazar el campeón por diferencias de ruido estadístico.
_MIN_IMPROVEMENT = 0.005  # 0.5 puntos porcentuales


class AutoTrainingService:
    """
    Servicio de auto-entrenamiento para el modelo de Churn.

    Flujo completo:
    1.  Extrae datos de la BD (PostgreSQL)
    2.  Preprocesa y genera features
    3.  Entrena un XGBClassifier
    4.  Evalúa métricas (accuracy, F1, precision, recall, AUC-ROC)
    5.  Registra el experimento en MLflow/DagsHub
    6.  Compara el challenger con el campeón actual (Champion/Challenger)
    7.  Si el challenger gana: hot-reload + sube artefactos a DagsHub
    8.  Persiste resultado (in_production=True/False) en churn_training_history
    """

    def __init__(self):
        # DB connection from environment variables (defaults match application.properties)
        self.db_params = {
            'host':     os.environ.get("DB_HOST",     "localhost"),
            'port':     int(os.environ.get("DB_PORT", "5432")),
            'dbname':   os.environ.get("DB_NAME",     "BankMindBetta_V3"),
            'user':     os.environ.get("DB_USER",     "postgres"),
            'password': os.environ.get("DB_PASSWORD", "1234"),
        }

        # Training state (for status polling)
        self.is_training = False
        self.last_result = None

    # ─────────────────────────────────────────────────────────────────────────
    # DATA
    # ─────────────────────────────────────────────────────────────────────────

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
            conn = psycopg2.connect(**self.db_params)
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
        df['TenureByAge']          = df['Tenure'] / (df['Age'] + epsilon)
        df['BalanceSalaryRatio']   = df['Balance'] / (df['EstimatedSalary'] + epsilon)
        df['CreditScoreGivenAge']  = df['CreditScore'] / (df['Age'] + epsilon)

        # Encoding Gender (normalizar a minúsculas para cubrir 'Male'/'male'/'Hombre')
        df['Gender'] = df['Gender'].str.strip().str.lower().map({
            'male': 1, 'female': 0,
            'hombre': 1, 'mujer': 0
        })
        df['Gender'] = df['Gender'].fillna(0).astype(int)

        # Encoding Geography (drop_first=True → eliminar France)
        geography_dummies = pd.get_dummies(df['Geography'], prefix='Geography')

        if 'Geography_Germany' not in geography_dummies.columns:
            geography_dummies['Geography_Germany'] = 0
        if 'Geography_Spain' not in geography_dummies.columns:
            geography_dummies['Geography_Spain'] = 0

        df = pd.concat([df, geography_dummies[['Geography_Germany', 'Geography_Spain']]], axis=1)
        df.drop(columns=['Geography'], inplace=True)

        y = df['Exited'].astype(int)

        feature_names = [
            'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'TenureByAge',
            'BalanceSalaryRatio', 'CreditScoreGivenAge', 'Geography_Germany', 'Geography_Spain'
        ]

        X = df[feature_names]
        return X, y, feature_names

    # ─────────────────────────────────────────────────────────────────────────
    # CHAMPION / CHALLENGER
    # ─────────────────────────────────────────────────────────────────────────

    def _get_champion_metrics(self):
        """
        Lee las métricas del modelo campeón actual desde churn_training_history.

        Returns:
            dict con auc_roc, f1_score, accuracy, precision_score, recall_score
            y model_version del último registro in_production=True,
            o None si todavía no existe ningún campeón (primer entrenamiento).
        """
        query = """
        SELECT auc_roc, f1_score, accuracy, precision_score, recall_score, model_version
        FROM public.churn_training_history
        WHERE in_production = true
        ORDER BY training_date DESC
        LIMIT 1
        """
        conn = None
        try:
            conn = psycopg2.connect(**self.db_params)
            cur = conn.cursor()
            cur.execute(query)
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "auc_roc":          float(row[0]) if row[0] is not None else 0.0,
                "f1_score":         float(row[1]) if row[1] is not None else 0.0,
                "accuracy":         float(row[2]) if row[2] is not None else 0.0,
                "precision_score":  float(row[3]) if row[3] is not None else 0.0,
                "recall_score":     float(row[4]) if row[4] is not None else 0.0,
                "model_version":    row[5],
            }
        except Exception as e:
            logger.warning(f"[Champion] No se pudieron leer métricas del campeón: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def _is_better_than_champion(self, new_metrics: dict, champion_metrics: dict):
        """
        Decide si el modelo challenger debe reemplazar al campeón.

        Usa un SCORE COMPUESTO para evitar que un modelo malo sea promovido
        solo porque una métrica individual es ligeramente mejor.

        Score compuesto = AUC·40% + F1·30% + Recall·30%

        Caso especial: Si el campeón tiene AUC-ROC ≈ 0.0 pero métricas razonables
        (F1 > 0.3), significa que el AUC fue almacenado incorrectamente (bug previo).
        En ese caso se ignora AUC y se compara solo por F1 + Recall.

        Returns:
            (promote: bool, reason: str)
        """
        new_auc   = new_metrics.get("auc_roc",  0.0)
        champ_auc = champion_metrics.get("auc_roc", 0.0)
        new_f1    = new_metrics.get("f1_score",  0.0)
        champ_f1  = champion_metrics.get("f1_score", 0.0)
        new_rec   = new_metrics.get("recall", new_metrics.get("recall_score", 0.0))
        champ_rec = champion_metrics.get("recall_score", 0.0)
        champ_ver = champion_metrics.get("model_version", "desconocido")

        # ── Detectar AUC-ROC no confiable en el campeón ──
        # Si AUC ≈ 0 pero F1 > 0.3, el AUC almacenado es artefacto de un bug
        champion_auc_unreliable = (champ_auc < 0.01 and champ_f1 > 0.3)

        if champion_auc_unreliable:
            # Ignorar AUC — comparar solo por F1 y Recall
            new_score   = 0.5 * new_f1  + 0.5 * new_rec
            champ_score = 0.5 * champ_f1 + 0.5 * champ_rec
            metric_label = "F1+Recall (AUC campeón no confiable=0.0)"
            logger.warning(
                f"[Champion/Challenger] AUC del campeón no confiable "
                f"({champ_auc:.4f}) con F1={champ_f1:.4f}. "
                f"Comparando solo por F1+Recall."
            )
        else:
            # Score compuesto normal: AUC 40% + F1 30% + Recall 30%
            new_score   = 0.4 * new_auc + 0.3 * new_f1  + 0.3 * new_rec
            champ_score = 0.4 * champ_auc + 0.3 * champ_f1 + 0.3 * champ_rec
            metric_label = "Score Compuesto (AUC·40% + F1·30% + Recall·30%)"

        delta = new_score - champ_score

        logger.info(
            f"[Champion/Challenger] {metric_label}: "
            f"challenger={new_score:.4f} vs campeón={champ_score:.4f} "
            f"(Δ{delta:+.4f}, mínimo={_MIN_IMPROVEMENT})"
        )

        if delta >= _MIN_IMPROVEMENT:
            return True, (
                f"{metric_label}: {champ_score:.4f} → {new_score:.4f} "
                f"(+{delta:.4f}) sobre campeón {champ_ver}."
            )

        if delta > -_MIN_IMPROVEMENT:
            # Prácticamente iguales
            return False, (
                f"Sin mejora significativa — {metric_label}: "
                f"{new_score:.4f} vs {champ_score:.4f} (Δ{delta:+.4f}). "
                f"Mínimo requerido: {_MIN_IMPROVEMENT}. Campeón {champ_ver} se mantiene."
            )

        # Empeoró claramente
        return False, (
            f"{metric_label} empeoró: {champ_score:.4f} → {new_score:.4f} "
            f"(Δ{delta:+.4f}). Campeón {champ_ver} se mantiene en producción."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING PIPELINE
    # ─────────────────────────────────────────────────────────────────────────

    def train_model(self):
        """Ejecuta el flujo completo de auto-entrenamiento con Champion/Challenger."""
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
                    safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
                    logger.info(f"[TOKEN CHECK] {safe_msg}")
                if not token_info.get('write'):
                    logger.warning(
                        "[WARN] El token DagsHub NO tiene permisos de escritura. "
                        "El modelo se entrenara pero NO se subira a DagsHub."
                    )
            except Exception as e_token:
                logger.warning(f"[WARN] No se pudo verificar token DagsHub: {e_token}")

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
            X_test_scaled  = scaler.transform(X_test)

            # 5. Entrenar con SMOTE + GridSearchCV (replicando el enfoque del Colab)
            neg = (y_train == 0).sum()
            pos = (y_train == 1).sum()
            logger.info(f"Class balance antes de SMOTE: {neg} neg / {pos} pos")

            # Aplicar SMOTE al conjunto de entrenamiento completo
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
            logger.info(f"Tras SMOTE: {len(X_res)} muestras, {int(y_res.sum())} positivos")

            # GridSearchCV sobre los datos balanceados
            param_grid = {
                'n_estimators':  [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth':     [3, 5],
            }

            grid_search = GridSearchCV(
                XGBClassifier(random_state=42, eval_metric='logloss'),
                param_grid,
                cv=3,
                scoring='roc_auc',
                n_jobs=1,
                verbose=0
            )
            grid_search.fit(X_res, y_res)

            model       = grid_search.best_estimator_
            best_params = grid_search.best_params_
            logger.info(f"[GridSearch] Mejores hiperparámetros: {best_params}")

            # 6. Evaluar challenger
            y_pred       = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            acc  = accuracy_score(y_test, y_pred)
            f1   = f1_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec  = recall_score(y_test, y_pred, zero_division=0)
            auc  = roc_auc_score(y_test, y_pred_proba)

            logger.info(
                f"[Challenger] Accuracy={acc:.4f} F1={f1:.4f} "
                f"Precision={prec:.4f} Recall={rec:.4f} AUC-ROC={auc:.4f}"
            )

            # 6.5 Champion / Challenger ──────────────────────────────────────
            challenger_metrics = {"auc_roc": auc, "f1_score": f1, "accuracy": acc, "recall": rec}
            champion = self._get_champion_metrics()

            if champion is None:
                promoted         = True
                promotion_reason = "Primer modelo registrado — promovido automáticamente."
                logger.info(f"[Champion/Challenger] Sin campeón previo. Promoción automática.")
            else:
                promoted, promotion_reason = self._is_better_than_champion(
                    challenger_metrics, champion
                )
                logger.info(f"[Champion/Challenger] promovido={promoted} | {promotion_reason}")
            # ────────────────────────────────────────────────────────────────

            # 7. MLflow / DagsHub tracking (siempre — queremos trazabilidad de todos los runs)
            mlflow_run_id = None
            version_tag   = f"v_{int(time.time())}"
            try:
                dagshub_client.init_dagshub_connection()
                mlflow.set_experiment("/churn_auto_training")

                with mlflow.start_run(run_name=f"churn_{version_tag}") as run:
                    mlflow.log_param("n_estimators",      best_params.get("n_estimators"))
                    mlflow.log_param("learning_rate",     best_params.get("learning_rate"))
                    mlflow.log_param("max_depth",         best_params.get("max_depth"))
                    mlflow.log_param("train_samples",     len(X_train))
                    mlflow.log_param("test_samples",      len(X_test))
                    mlflow.log_param("smote",             True)
                    mlflow.log_param("gridsearch_cv",     3)
                    mlflow.log_param("version_tag",       version_tag)
                    mlflow.log_param("promoted",          promoted)
                    mlflow.log_metric("accuracy",         acc)
                    mlflow.log_metric("f1_score",         f1)
                    mlflow.log_metric("precision",        prec)
                    mlflow.log_metric("recall",           rec)
                    mlflow.log_metric("auc_roc",          auc)
                    mlflow.xgboost.log_model(model, "model")
                    mlflow_run_id = run.info.run_id
                    logger.info(f"[OK] Experimento registrado en MLflow. Run ID: {mlflow_run_id}")
            except Exception as e_mlflow:
                logger.warning(f"[WARN] MLflow tracking fallo (no bloquea el entrenamiento): {e_mlflow}")

            # 8. Empaquetar combo-pack
            combo_pack = {
                'modelo_prediccion': model,
                'scaler':            scaler,
                'feature_names':     feature_names,
                'meta_info': {
                    'version':       version_tag,
                    'accuracy':      round(acc,  4),
                    'f1_score':      round(f1,   4),
                    'precision':     round(prec, 4),
                    'recall':        round(rec,  4),
                    'auc_roc':       round(auc,  4),
                    'train_samples': len(X_train),
                    'test_samples':  len(X_test),
                    'mlflow_run_id': mlflow_run_id,
                    'promoted':      promoted,
                    'descripcion':   'Modelo CHURN XGBoost con scaler y features'
                }
            }

            # 9. Hot-reload: SOLO si el challenger ganó
            dagshub_verified = False
            upload_errors    = []

            if promoted:
                cargar_modelos_desde_local(model, scaler, feature_names)
                try:
                    from fuga.service.churn_service import churn_service as _cs
                    _cs.model         = model
                    _cs.scaler        = scaler
                    _cs.feature_names = feature_names
                    _cs.model_version = version_tag
                    logger.info(f"[OK] ChurnService actualizado -> version {version_tag}.")
                except Exception as e_reload:
                    logger.warning(f"No se pudo actualizar ChurnService en caliente: {e_reload}")

                # 10. Subir a DagsHub: SOLO si el challenger ganó
                try:
                    buf = io.BytesIO()
                    joblib.dump(combo_pack, buf)
                    model_bytes = buf.getvalue()

                    upload_ok = dagshub_client.upload_champion(model_bytes, version_tag)

                    if upload_ok:
                        integrity_ok = dagshub_client.verify_champion_integrity(version_tag)
                        if integrity_ok:
                            dagshub_verified = True
                            logger.info("[OK] Upload + Verificacion DagsHub OK")
                        else:
                            upload_errors.append("Verificación de integridad falló")

                        try:
                            legacy_ok = dagshub_client.upload_individual_files(
                                model, scaler, feature_names, version_tag
                            )
                            if not legacy_ok:
                                upload_errors.append("Algunos archivos legacy no se subieron")
                        except Exception as e_legacy:
                            upload_errors.append(f"Error subiendo archivos legacy: {e_legacy}")
                            logger.warning(f"[WARN] Error archivos legacy: {e_legacy}")
                    else:
                        upload_errors.append("Upload a DagsHub falló")
                except Exception as e_upload:
                    upload_errors.append(f"Error en upload: {e_upload}")
                    logger.error(f"[ERROR] Error en upload: {e_upload}")
            else:
                logger.info(
                    f"[SKIP] Hot-reload y upload omitidos - "
                    f"campeon se mantiene en produccion. Motivo: {promotion_reason}"
                )

            # 11. Construir resultado
            result = {
                "status":           "success",
                "metrics": {
                    "accuracy":     round(acc,  4),
                    "f1_score":     round(f1,   4),
                    "precision":    round(prec, 4),
                    "recall":       round(rec,  4),
                    "auc_roc":      round(auc,  4),
                },
                "run_id":           mlflow_run_id,
                "version_tag":      version_tag,
                "in_production":    promoted,
                "promoted":         promoted,
                "promotion_reason": promotion_reason,
                "champion_metrics": champion,
                "dagshub_verified": dagshub_verified,
                "train_samples":    len(X_train),
                "test_samples":     len(X_test),
                "message":          "Modelo entrenado exitosamente." if promoted
                                    else "Modelo entrenado pero no promovido — campeón se mantiene.",
            }

            if mlflow_run_id:
                result["message"] += " Registrado en MLflow/DagsHub."
            else:
                result["mlflow_warning"] = (
                    "MLflow tracking no disponible. "
                    "El modelo se entrenó y guardó correctamente sin tracking."
                )

            if promoted and dagshub_verified:
                result["message"] += " Modelo verificado en DagsHub."
            elif promoted and upload_errors:
                result["upload_warnings"] = upload_errors
                result["message"] += f" ({len(upload_errors)} advertencias de carga a DagsHub)"

            # 12. Persistir en BD (in_production refleja si fue promovido o no)
            self._persist_training_to_db(result)

            self.last_result = result
            logger.info(
                f"=== AUTO-ENTRENAMIENTO FINALIZADO — "
                f"promovido={promoted} | {promotion_reason} ==="
            )
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

    # ─────────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────────

    def _persist_training_to_db(self, result: dict):
        """
        Persiste el resultado de un ciclo de entrenamiento en churn_training_history.

        El campo in_production refleja si el challenger fue promovido (True)
        o rechazado por el proceso Champion/Challenger (False).
        """
        import json
        from datetime import datetime

        metrics       = result.get("metrics", {})
        version_tag   = result.get("version_tag")
        run_id        = result.get("run_id")
        model_version = (version_tag[:100] if version_tag
                         else (run_id[:100] if run_id else None))
        in_production = result.get("in_production", True)

        # Incluir motivo de promoción/rechazo y métricas del campeón en warnings
        warnings_json = {}
        if result.get("mlflow_warning"):
            warnings_json["mlflow_warning"] = result["mlflow_warning"]
        if result.get("upload_warnings"):
            warnings_json["upload_warnings"] = result["upload_warnings"]
        if result.get("promotion_reason"):
            warnings_json["promotion_reason"] = result["promotion_reason"]
        if result.get("champion_metrics"):
            warnings_json["champion_metrics"] = result["champion_metrics"]

        insert_query = """
        INSERT INTO public.churn_training_history
            (training_date, trigger_reason, in_production, model_version,
             accuracy, f1_score, precision_score, recall_score, auc_roc,
             train_samples, test_samples, warnings)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        values = (
            datetime.now(),
            "manual_training",
            in_production,
            model_version,
            metrics.get("accuracy"),
            metrics.get("f1_score"),
            metrics.get("precision"),
            metrics.get("recall"),
            metrics.get("auc_roc"),
            result.get("train_samples"),
            result.get("test_samples"),
            json.dumps(warnings_json) if warnings_json else None,
        )

        conn = None
        try:
            conn = psycopg2.connect(**self.db_params)
            cur  = conn.cursor()
            cur.execute(insert_query, values)
            conn.commit()
            cur.close()
            logger.info(
                f"[OK] Resultado guardado en churn_training_history "
                f"(in_production={in_production}, version={model_version})."
            )
        except Exception as e:
            logger.error(f"[ERROR] Error guardando resultado en BD: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()


# Singleton (una sola instancia)
auto_training_service = AutoTrainingService()
