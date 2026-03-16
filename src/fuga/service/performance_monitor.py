"""
Performance Monitor para el modelo de Churn.

Evalúa la calidad del modelo en producción comparando predicciones históricas
contra el estado real del cliente (ground truth: account_details.exited).

Si el Recall cae por debajo del umbral, dispara un re-entrenamiento automático.
"""

import os
import logging
import psycopg2
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURACIÓN (desde variables de entorno)
# ============================================================
RECALL_THRESHOLD = float(os.environ.get("CHURN_RECALL_THRESHOLD", "0.75"))
MIN_FEEDBACK_SAMPLES = int(os.environ.get("CHURN_MIN_FEEDBACK_SAMPLES", "10"))
MATURATION_DAYS = int(os.environ.get("CHURN_MATURATION_DAYS", "1"))
MONITOR_INTERVAL_HOURS = int(os.environ.get("CHURN_MONITOR_INTERVAL_HOURS", "6"))
MONITOR_ENABLED = os.environ.get("CHURN_MONITOR_ENABLED", "true").lower() == "true"


class PerformanceMonitorService:
    """
    Servicio de monitoreo de rendimiento del modelo de Churn.

    Flujo:
    1. Consulta predicciones maduras (>30 días) de churn_predictions
    2. Cruza con account_details.exited (ground truth)
    3. Calcula Recall, F1-Score, Precision, Accuracy
    4. Si Recall < umbral → dispara re-entrenamiento
    5. Registra evaluación en churn_training_history
    """

    def __init__(self):
        db_name = os.environ.get("DB_NAME", "BankMindDB")
        db_user = os.environ.get("DB_USER", "postgres")
        db_password = os.environ.get("DB_PASSWORD", "1234")
        db_host = os.environ.get("DB_HOST", "localhost")
        db_port = os.environ.get("DB_PORT", "5432")
        self.db_url = (
            f"dbname='{db_name}' user='{db_user}' "
            f"password='{db_password}' host='{db_host}' port='{db_port}'"
        )

        # Estado del último análisis
        self.last_evaluation: Optional[dict] = None
        self.last_evaluation_time: Optional[datetime] = None
        self.next_evaluation_time: Optional[datetime] = None

    # ----------------------------------------------------------------
    # CORE: Evaluación del modelo contra Ground Truth
    # ----------------------------------------------------------------
    def evaluate_model_performance(self) -> dict:
        """
        Evalúa el rendimiento real del modelo comparando predicciones
        históricas contra el estado actual del cliente.

        Returns:
            dict con métricas, estado, y recomendación
        """
        logger.info("=== INICIO EVALUACIÓN DE RENDIMIENTO CHURN ===")

        try:
            # 1. Obtener datos de feedback (predicciones vs realidad)
            feedback_data = self._get_feedback_data()

            if feedback_data is None or len(feedback_data) == 0:
                result = {
                    "status": "insufficient_data",
                    "message": (
                        "No hay predicciones maduras disponibles para evaluar. "
                        f"Se requieren predicciones con más de {MATURATION_DAYS} días."
                    ),
                    "evaluated_samples": 0,
                    "min_samples_required": MIN_FEEDBACK_SAMPLES,
                    "recall_threshold": RECALL_THRESHOLD,
                    "maturation_days": MATURATION_DAYS,
                    "auto_training_triggered": False,
                }
                self._update_state(result)
                return result

            total_samples = len(feedback_data)

            if total_samples < MIN_FEEDBACK_SAMPLES:
                result = {
                    "status": "insufficient_data",
                    "message": (
                        f"Datos insuficientes: {total_samples} muestras "
                        f"(mínimo requerido: {MIN_FEEDBACK_SAMPLES}). "
                        "Esperando más feedback confirmado."
                    ),
                    "evaluated_samples": total_samples,
                    "min_samples_required": MIN_FEEDBACK_SAMPLES,
                    "recall_threshold": RECALL_THRESHOLD,
                    "maturation_days": MATURATION_DAYS,
                    "auto_training_triggered": False,
                }
                self._update_state(result)
                return result

            # 2. Calcular matriz de confusión
            tp, fp, fn, tn = self._compute_confusion_matrix(feedback_data)

            # 3. Calcular métricas
            metrics = self._compute_metrics(tp, fp, fn, tn)

            # 4. Determinar estado
            recall = metrics["recall"]
            if recall >= RECALL_THRESHOLD:
                status = "healthy"
                message = (
                    f"Modelo saludable. Recall={recall:.4f} "
                    f"(umbral={RECALL_THRESHOLD})"
                )
            else:
                status = "degraded"
                message = (
                    f"⚠️ Rendimiento degradado. Recall={recall:.4f} "
                    f"< umbral={RECALL_THRESHOLD}. "
                    "Se recomienda re-entrenamiento."
                )

            result = {
                "status": status,
                "message": message,
                "recall": round(recall, 4),
                "f1_score": round(metrics["f1_score"], 4),
                "precision": round(metrics["precision"], 4),
                "accuracy": round(metrics["accuracy"], 4),
                "recall_threshold": RECALL_THRESHOLD,
                "evaluated_samples": total_samples,
                "min_samples_required": MIN_FEEDBACK_SAMPLES,
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "auto_training_triggered": False,
                "last_evaluation_date": datetime.now().isoformat(),
            }

            logger.info(
                f"Evaluación completada: status={status}, "
                f"Recall={recall:.4f}, F1={metrics['f1_score']:.4f}, "
                f"Samples={total_samples}"
            )

            self._update_state(result)
            return result

        except Exception as e:
            logger.error(f"Error en evaluación de rendimiento: {e}")
            import traceback
            traceback.print_exc()
            result = {
                "status": "error",
                "message": f"Error al evaluar rendimiento: {str(e)}",
                "auto_training_triggered": False,
            }
            self._update_state(result)
            return result

    # ----------------------------------------------------------------
    # Disparo programado (llamado por APScheduler)
    # ----------------------------------------------------------------
    def run_scheduled_evaluation(self):
        """
        Ejecuta la evaluación programada y, si es necesario,
        dispara el re-entrenamiento automático.

        NUNCA lanza excepciones — el servicio principal no debe caerse.
        """
        logger.info("[SCHEDULER] Ejecutando evaluación programada de rendimiento...")

        try:
            result = self.evaluate_model_performance()

            if result.get("status") == "degraded":
                logger.warning(
                    f"[SCHEDULER] Rendimiento degradado detectado. "
                    f"Recall={result.get('recall', 'N/A')} "
                    f"< umbral={RECALL_THRESHOLD}"
                )
                # Intentar re-entrenamiento automático
                self._trigger_auto_retrain(result)

            elif result.get("status") == "healthy":
                # Registrar evaluación saludable en churn_training_history
                self._persist_evaluation(result, training_result=None)
                logger.info("[SCHEDULER] Modelo saludable. No se requiere acción.")

            elif result.get("status") == "insufficient_data":
                logger.info(
                    "[SCHEDULER] Datos insuficientes para evaluar. "
                    "Se omite el ciclo."
                )

            # Actualizar siguiente evaluación
            self.next_evaluation_time = (
                datetime.now() + timedelta(hours=MONITOR_INTERVAL_HOURS)
            )

        except Exception as e:
            logger.error(f"[SCHEDULER] Error no controlado en evaluación: {e}")
            import traceback
            traceback.print_exc()
            # NUNCA se propaga la excepción

    # ----------------------------------------------------------------
    # PRIVADO: Obtener datos de feedback desde la BD
    # ----------------------------------------------------------------
    def _get_feedback_data(self) -> list:
        """
        Consulta la BD para obtener predicciones maduras cruzadas
        con el estado real del cliente.

        Retorna lista de tuplas: (predicted_churn, actual_exited)
        """
        cutoff_date = datetime.now() - timedelta(days=MATURATION_DAYS)

        query = """
        SELECT
            cp.is_churn    AS predicted_churn,
            ad.exited      AS actual_exited
        FROM public.churn_predictions cp
        JOIN public.account_details ad
            ON cp.id_customer = ad.id_customer
        WHERE cp.prediction_date <= %s
          AND cp.is_churn IS NOT NULL
          AND ad.exited IS NOT NULL
        ORDER BY cp.prediction_date DESC
        """

        conn = None
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            cur.execute(query, (cutoff_date,))
            rows = cur.fetchall()
            cur.close()
            logger.info(
                f"Feedback data: {len(rows)} registros maduros encontrados "
                f"(predicciones anteriores a {cutoff_date.strftime('%Y-%m-%d')})"
            )
            return rows
        except psycopg2.OperationalError as e:
            logger.error(f"Error de conexión a la BD: {e}")
            raise RuntimeError(
                f"No se pudo conectar a la base de datos para evaluar rendimiento. "
                f"Detalle: {e}"
            )
        except Exception as e:
            logger.error(f"Error consultando feedback: {e}")
            raise
        finally:
            if conn:
                conn.close()

    # ----------------------------------------------------------------
    # PRIVADO: Matriz de confusión
    # ----------------------------------------------------------------
    def _compute_confusion_matrix(self, data: list) -> tuple:
        """
        Calcula TP, FP, FN, TN a partir de (predicted, actual).

        - TP: Modelo predijo churn (True), cliente realmente fugó (True)
        - FP: Modelo predijo churn (True), cliente NO fugó (False)
        - FN: Modelo predijo NO churn (False), cliente SÍ fugó (True)
        - TN: Modelo predijo NO churn (False), cliente NO fugó (False)
        """
        tp = fp = fn = tn = 0

        for predicted_churn, actual_exited in data:
            pred = bool(predicted_churn)
            real = bool(actual_exited)

            if pred and real:
                tp += 1
            elif pred and not real:
                fp += 1
            elif not pred and real:
                fn += 1
            else:
                tn += 1

        logger.info(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        return tp, fp, fn, tn

    # ----------------------------------------------------------------
    # PRIVADO: Métricas de clasificación
    # ----------------------------------------------------------------
    def _compute_metrics(self, tp: int, fp: int, fn: int, tn: int) -> dict:
        """
        Calcula Recall, Precision, F1-Score y Accuracy.
        Usa aritmética simple (sin librerías de ML).
        """
        # Recall = TP / (TP + FN) — ¿Qué % de fugas reales detectamos?
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Precision = TP / (TP + FP) — ¿Qué % de alertas son correctas?
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # F1 = 2 * (P * R) / (P + R) — Media harmónica
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Accuracy = (TP + TN) / Total
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total > 0 else 0.0

        return {
            "recall": recall,
            "precision": precision,
            "f1_score": f1_score,
            "accuracy": accuracy,
        }

    # ----------------------------------------------------------------
    # PRIVADO: Disparar re-entrenamiento automático
    # ----------------------------------------------------------------
    def _trigger_auto_retrain(self, evaluation_result: dict):
        """
        Dispara el re-entrenamiento del modelo y registra los resultados.
        """
        logger.info("=== DISPARANDO RE-ENTRENAMIENTO AUTOMÁTICO (Performance Decay) ===")

        try:
            from fuga.service.auto_training_service import auto_training_service

            # Verificar que no haya un entrenamiento en curso
            if auto_training_service.is_training:
                logger.warning(
                    "Re-entrenamiento omitido: ya hay uno en curso."
                )
                evaluation_result["auto_training_triggered"] = False
                evaluation_result["message"] += (
                    " Re-entrenamiento omitido (ya hay uno en curso)."
                )
                return

            # Ejecutar entrenamiento
            training_result = auto_training_service.train_model()

            if training_result.get("status") == "success":
                evaluation_result["auto_training_triggered"] = True
                evaluation_result["training_run_id"] = training_result.get("run_id")
                evaluation_result["training_metrics"] = training_result.get("metrics")
                logger.info(
                    "✅ Re-entrenamiento automático completado exitosamente. "
                    f"Run ID: {training_result.get('run_id')}"
                )
            else:
                evaluation_result["auto_training_triggered"] = False
                evaluation_result["training_error"] = training_result.get("error")
                logger.error(
                    f"❌ Re-entrenamiento automático falló: "
                    f"{training_result.get('error')}"
                )

            # Persistir evaluación + resultado del entrenamiento
            self._persist_evaluation(evaluation_result, training_result)

        except Exception as e:
            logger.error(f"Error disparando re-entrenamiento: {e}")
            import traceback
            traceback.print_exc()
            evaluation_result["auto_training_triggered"] = False
            evaluation_result["training_error"] = str(e)

    # ----------------------------------------------------------------
    # PRIVADO: Persistir en churn_training_history
    # ----------------------------------------------------------------
    def _persist_evaluation(self, evaluation: dict, training_result: Optional[dict]):
        """
        Registra la evaluación en la tabla churn_training_history (exclusiva de CHURN).

        Usa columnas tipadas para métricas, matriz de confusión y metadata.
        """
        import json

        trigger_reason = (
            "performance_decay"
            if evaluation.get("auto_training_triggered")
            else "scheduled_check"
        )

        # Métricas del nuevo modelo (si hubo re-entrenamiento exitoso)
        new_metrics = {}
        model_version = None
        in_production = False
        if training_result and training_result.get("status") == "success":
            new_metrics = training_result.get("metrics", {})
            run_id = training_result.get("run_id")
            model_version = run_id[:100] if run_id else None
            in_production = True

        # Warnings en JSONB
        warnings_json = {}
        if training_result and training_result.get("status") == "success":
            if new_metrics:
                warnings_json["new_model_metrics"] = new_metrics

        insert_query = """
        INSERT INTO public.churn_training_history
            (training_date, trigger_reason, in_production, model_version,
             accuracy, f1_score, precision_score, recall_score, auc_roc,
             evaluated_samples, true_positives, false_positives,
             true_negatives, false_negatives, recall_threshold, warnings)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        conn = None
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            cur.execute(
                insert_query,
                (
                    datetime.now(),
                    trigger_reason,
                    in_production,
                    model_version,
                    evaluation.get("accuracy"),
                    evaluation.get("f1_score"),
                    evaluation.get("precision"),
                    evaluation.get("recall"),
                    None,  # auc_roc no se calcula en la evaluación del monitor
                    evaluation.get("evaluated_samples"),
                    evaluation.get("true_positives"),
                    evaluation.get("false_positives"),
                    evaluation.get("true_negatives"),
                    evaluation.get("false_negatives"),
                    RECALL_THRESHOLD,
                    json.dumps(warnings_json) if warnings_json else None,
                ),
            )
            conn.commit()
            cur.close()
            logger.info(
                f"✅ Evaluación registrada en churn_training_history "
                f"(trigger: {trigger_reason})"
            )
        except Exception as e:
            logger.error(f"Error persistiendo evaluación: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    # ----------------------------------------------------------------
    # Estado actual (para el endpoint GET /monitor/status)
    # ----------------------------------------------------------------
    def get_status(self) -> dict:
        """Retorna el estado actual del monitor."""
        if self.last_evaluation is None:
            return {
                "status": "no_evaluations",
                "message": "No se ha realizado ninguna evaluación aún.",
                "recall_threshold": RECALL_THRESHOLD,
                "min_samples_required": MIN_FEEDBACK_SAMPLES,
                "maturation_days": MATURATION_DAYS,
                "monitor_enabled": MONITOR_ENABLED,
                "monitor_interval_hours": MONITOR_INTERVAL_HOURS,
                "auto_training_triggered": False,
            }

        result = dict(self.last_evaluation)
        result["monitor_enabled"] = MONITOR_ENABLED
        result["monitor_interval_hours"] = MONITOR_INTERVAL_HOURS
        result["maturation_days"] = MATURATION_DAYS

        if self.last_evaluation_time:
            result["last_evaluation_date"] = self.last_evaluation_time.isoformat()
        if self.next_evaluation_time:
            result["next_evaluation_date"] = self.next_evaluation_time.isoformat()

        return result

    # ----------------------------------------------------------------
    # PRIVADO: Actualizar estado interno
    # ----------------------------------------------------------------
    def _update_state(self, result: dict):
        """Actualiza el estado interno del monitor."""
        self.last_evaluation = result
        self.last_evaluation_time = datetime.now()


# Singleton
performance_monitor = PerformanceMonitorService()
