"""
fraud_service.py — Servicio de predicción de fraude.

Responsabilidad: feature engineering, encode, scale, predict, SHAP explanation.
No gestiona el ciclo de vida del modelo (eso lo hace ModelLoader).
"""
import logging
from typing import Optional

from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from fraude.models_files import model_loader
from fraude.schema.inputs import (
    AuditData,
    BatchFraudInput,
    BatchFraudOutput,
    FraudInput,
    FraudOutput,
    RiskFactor,
)

logger = logging.getLogger(__name__)

# Columnas que deben escalarse antes de pasar al modelo
_COLS_TO_SCALE = ["amt", "city_pop", "age", "distance_km", "hour"]

# Columnas categóricas a codificar
_CATEGORICAL_COLS = ["category", "gender", "job"]

# Adaptador de contrato: Java envía 'M'/'F', el encoder espera 'male'/'female'
_GENDER_NORMALIZE: dict[str, str] = {
    "m": "male",
    "f": "female",
}


def _haversine(lon1, lat1, lon2, lat2):
    """Distancia en km entre dos puntos (Haversine). Trabaja con escalares o arrays."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _wrap_xgb(old_xgb) -> XGBClassifier:
    """
    Re-envuelve un XGBClassifier serializado con una versión antigua de XGBoost
    en una instancia limpia compatible con la versión instalada.

    Usa la API pública `get_booster().save_raw()` / `load_model()` en lugar de
    atributos privados. Esto es estable entre versiones de XGBoost >= 1.6.
    """
    try:
        booster = old_xgb.get_booster()

        # Serializar el booster entrenado a bytes usando formato UBJ (oficial)
        raw_bytes = booster.save_raw("ubj")

        # Crear instancia limpia compatible con la versión instalada
        new_xgb = XGBClassifier(
            eval_metric="logloss",
            device="cpu",
            random_state=42,
            n_jobs=-1,
        )

        # Cargar el booster serializado — API pública, estable y garantizada
        new_xgb.load_model(bytearray(raw_bytes))

        logger.info("✅ XGBoost re-envuelto correctamente mediante save_raw/load_model.")
        return new_xgb

    except Exception as exc:
        logger.warning(
            "No se pudo re-envolver XGBoost (save_raw/load_model): %s. "
            "Usando el modelo original tal cual.",
            exc,
        )
        return old_xgb


class FraudService:
    """
    Servicio de predicción de fraude.

    *Single responsibility*: dada una transacción (o lote), devuelve
    veredicto + score + factores SHAP.
    """

    def __init__(self):
        self._load_components()

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def _load_components(self) -> None:
        """Extrae y prepara los componentes del model_pack."""
        pack = model_loader.get()

        self.scaler = pack["scaler"]
        self.encoders: dict = pack["encoders"]
        self.if_model = pack["model_if"]
        self.explainer = pack.get("explainer")

        # Envolver el XGBClassifier con la API oficial de serialización
        self.xgb_model: XGBClassifier = _wrap_xgb(pack["model_xgb"])

        # Cachear feature_names una sola vez — evita llamar get_booster() en cada predicción
        self.feature_names: list[str] = self.xgb_model.get_booster().feature_names

        # Leer optimal_threshold desde meta_info (calculado por Optuna en entrenamiento)
        meta = pack.get("meta_info", {}) or {}
        self.threshold: float = float(meta.get("optimal_threshold", 0.5))
        if self.threshold == 0.5 and "optimal_threshold" not in meta:
            logger.warning(
                "optimal_threshold no encontrado en meta_info. "
                "Usando 0.5 como fallback — considera re-entrenar el modelo."
            )
        else:
            logger.info("Threshold óptimo del modelo: %.4f", self.threshold)

        logger.info(
            "FraudService listo. Versión: %s | Threshold: %.4f | SHAP: %s",
            model_loader.version,
            self.threshold,
            "disponible" if self.explainer else "no disponible",
        )

    def reload(self) -> dict:
        """
        Recarga el modelo desde DagsHub de forma atómica.
        Si _load_components falla a mitad, el estado del servicio actual
        se preserva intacto (no hay mezcla de componentes de versiones distintas).
        """
        logger.info("🔄 Recargando modelo de fraude en caliente...")
        model_loader.reload()   # Puede lanzar ValueError — si falla, pack anterior activo

        # Construir servicio nuevo en un objeto temporal
        # Si falla aquí, self (el servicio activo) queda sin tocar
        new_svc = FraudService.__new__(FraudService)
        new_svc._load_components()          # Puede lanzar — sin riesgo para self

        # Todo correcto: copiar atributos atómicamente
        self.__dict__.update(new_svc.__dict__)
        logger.info("✅ Modelo recargado correctamente. Versión: %s", model_loader.version)
        return {
            "status": "success",
            "message": f"Modelo recargado. Versión: {model_loader.version}",
            "version": model_loader.version,
        }

    # ------------------------------------------------------------------
    # Predicción
    # ------------------------------------------------------------------

    def predict(self, input_data: FraudInput) -> FraudOutput:
        """Predice fraude para una única transacción."""
        df = pd.DataFrame([input_data.dict()])
        raw_values = df.iloc[0].to_dict()

        df, raw_values, unknown_fields = self._feature_engineering(df, raw_values)
        X_final = self._build_feature_matrix(df)

        proba = float(self.xgb_model.predict_proba(X_final)[0][1])
        veredicto = "ALTO RIESGO" if proba > self.threshold else "LEGÍTIMO"

        risk_factors = self._shap_factors(X_final, raw_values) if self.explainer else []
        if not risk_factors and veredicto == "ALTO RIESGO":
            risk_factors = [
                RiskFactor(
                    feature_name="Modelo General",
                    feature_value="N/A",
                    shap_value=proba,
                    risk_description="Patrón general sospechoso detectado por XGBoost",
                    impact_direction="AUMENTA_RIESGO",
                )
            ]

        audit = AuditData(
            xgboost_score=proba,
            iforest_score=float(X_final["anomaly_score"].iloc[0]),
            base_score=(
                float(self.explainer.expected_value)
                if self.explainer and hasattr(self.explainer, "expected_value")
                else 0.0
            ),
            threshold_used=self.threshold,
            unknown_encoder_values=unknown_fields if unknown_fields else None,
        )

        return FraudOutput(
            transaction_id=input_data.transaction_id,
            veredicto=veredicto,
            score_final=proba,
            detalles_riesgo=risk_factors,
            datos_auditoria=audit,
            recomendacion="Bloquear y Notificar" if veredicto == "ALTO RIESGO" else "Aprobar",
        )

    def predict_batch(self, batch_input: BatchFraudInput) -> BatchFraudOutput:
        """
        Predice fraude para múltiples transacciones de forma vectorizada.
        Construye una sola matriz y llama predict_proba una vez — O(1) overhead
        de modelo vs O(N) del enfoque secuencial anterior.
        """
        transactions = batch_input.transactions
        if not transactions:
            return BatchFraudOutput(
                total_processed=0, total_frauds=0, total_legitimate=0,
                total_errors=0, results=[],
            )

        results = []
        total_frauds = 0
        total_legitimate = 0
        total_errors = 0

        # ------------------------------------------------------------------
        # Ruta vectorizada: un solo DataFrame, un solo predict_proba
        # ------------------------------------------------------------------
        try:
            rows = [t.dict() for t in transactions]
            df_all = pd.DataFrame(rows)

            df_all, _, _ = self._feature_engineering(df_all, {})
            X_all = self._build_feature_matrix(df_all)

            probas = self.xgb_model.predict_proba(X_all)[:, 1]

            # SHAP vectorizado: una sola llamada para todas las filas
            all_shap_values = None
            if self.explainer:
                try:
                    raw_shap = self.explainer.shap_values(X_all)
                    # XGBoost binario devuelve array 2D directo o lista [clase0, clase1]
                    all_shap_values = raw_shap[1] if isinstance(raw_shap, list) else raw_shap
                except Exception as shap_exc:
                    logger.warning("SHAP batch falló, se omitirá explicabilidad: %s", shap_exc)

            for i, (tx, proba) in enumerate(zip(transactions, probas)):
                proba = float(proba)
                veredicto = "ALTO RIESGO" if proba > self.threshold else "LEGÍTIMO"

                # Construir raw_row con los valores legibles para el tooltip de SHAP:
                # - Valores categóricos originales (pre-encoding) vienen de df_all ya
                #   procesado (gender normalizado, etc.) — suficiente para mostrar al usuario
                # - age, hour, distance_km son derivados calculados durante el engineering
                raw_row = transactions[i].dict()     # valores originales del request
                raw_row["age"] = int(df_all["age"].iloc[i])
                raw_row["hour"] = int(df_all["hour"].iloc[i])
                raw_row["distance_km"] = round(float(df_all["distance_km"].iloc[i]), 2)

                # SHAP indexado (ya calculado en batch arriba)
                risk_factors = []
                if all_shap_values is not None:
                    shap_row = all_shap_values[i]
                    risk_factors = self._build_risk_factors_from_shap(
                        X_all.columns.tolist(), shap_row, raw_row
                    )
                if not risk_factors and veredicto == "ALTO RIESGO":
                    risk_factors = [RiskFactor(
                        feature_name="Modelo General", feature_value="N/A",
                        shap_value=proba,
                        risk_description="Patrón general sospechoso detectado por XGBoost",
                        impact_direction="AUMENTA_RIESGO",
                    )]

                audit = AuditData(
                    xgboost_score=proba,
                    iforest_score=float(X_all["anomaly_score"].iloc[i]),
                    base_score=0.0,
                    threshold_used=self.threshold,
                )

                results.append(FraudOutput(
                    transaction_id=tx.transaction_id,
                    veredicto=veredicto,
                    score_final=proba,
                    detalles_riesgo=risk_factors,
                    datos_auditoria=audit,
                    recomendacion=(
                        "Bloquear y Notificar" if veredicto == "ALTO RIESGO" else "Aprobar"
                    ),
                ))

                if veredicto == "ALTO RIESGO":
                    total_frauds += 1
                else:
                    total_legitimate += 1

        except Exception as exc:
            # Si falla la ruta vectorizada (datos incompatibles, etc.) hace fallback
            # transacción a transacción para minimizar impacto al caller.
            logger.error(
                "Falló el path vectorizado de predict_batch (%s). "
                "Reintentando transacción a transacción.",
                exc,
            )
            results = []
            total_frauds = total_legitimate = total_errors = 0
            for tx in transactions:
                try:
                    r = self.predict(tx)
                    results.append(r)
                    if r.veredicto == "ALTO RIESGO":
                        total_frauds += 1
                    else:
                        total_legitimate += 1
                except Exception as tx_exc:
                    logger.warning("Error prediciendo tx %s: %s", tx.transaction_id, tx_exc)
                    results.append(FraudOutput(
                        transaction_id=tx.transaction_id,
                        veredicto="ERROR",
                        score_final=0.0,
                        detalles_riesgo=[],
                        datos_auditoria=AuditData(
                            xgboost_score=0.0, iforest_score=0.0,
                            base_score=0.0, threshold_used=self.threshold,
                        ),
                        recomendacion="Revisar manualmente",
                        error=str(tx_exc),
                    ))
                    total_errors += 1

        return BatchFraudOutput(
            total_processed=len(results),
            total_frauds=total_frauds,
            total_legitimate=total_legitimate,
            total_errors=total_errors,
            results=results,
        )

    # ------------------------------------------------------------------
    # Internos de pipeline
    # ------------------------------------------------------------------

    def _feature_engineering(
        self, df: pd.DataFrame, raw_values: dict
    ) -> tuple[pd.DataFrame, dict, dict]:
        """
        Aplica feature engineering (age, hour, distance_km) y encoding.

        Returns:
            (df_modificado, raw_values_actualizado, dict_de_campos_desconocidos)
        """
        # Fechas y variables derivadas
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
        df["dob"] = pd.to_datetime(df["dob"])
        # relativedelta calcula años completos considerando bisiestos correctamente
        df["age"] = df.apply(
            lambda r: relativedelta(r["trans_date_trans_time"], r["dob"]).years, axis=1
        )
        df["hour"] = df["trans_date_trans_time"].dt.hour
        df["distance_km"] = _haversine(
            df["long"], df["lat"], df["merch_long"], df["merch_lat"]
        )

        if raw_values:
            raw_values["age"] = int(df["age"].iloc[0])
            raw_values["hour"] = int(df["hour"].iloc[0])
            raw_values["distance_km"] = round(float(df["distance_km"].iloc[0]), 2)

        # Normalizar género: adaptar códigos de una letra (M/F) al formato
        # del encoder (male/female). Insensible a mayúsculas.
        if "gender" in df.columns:
            df["gender"] = df["gender"].astype(str).str.lower().map(
                lambda v: _GENDER_NORMALIZE.get(v, v)
            )

        # Encoding de categóricas
        unknown_fields: dict[str, str] = {}
        for col in _CATEGORICAL_COLS:
            encoder = self.encoders[col]
            values = df[col].astype(str).tolist()
            encoded = []
            for val in values:
                if val in encoder.classes_:
                    encoded.append(encoder.transform([val])[0])
                else:
                    default = encoder.classes_[0]
                    logger.warning(
                        "Valor desconocido '%s' en columna '%s'. "
                        "Mapeando a '%s' para no interrumpir la predicción. "
                        "Considera actualizar el encoder.",
                        val, col, default,
                    )
                    unknown_fields[col] = val
                    encoded.append(encoder.transform([default])[0])
            df[col] = encoded

        return df, raw_values, unknown_fields

    def _build_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construye la matriz de features final lista para XGBoost:
        selección de columnas → scaling → anomaly_score.
        Usa self.feature_names cacheado en _load_components().
        """
        base_cols = [c for c in self.feature_names if c != "anomaly_score"]

        X = df[base_cols].copy().astype(float)
        X[_COLS_TO_SCALE] = self.scaler.transform(X[_COLS_TO_SCALE])
        X["anomaly_score"] = self.if_model.decision_function(X[base_cols])

        return X[self.feature_names]

    def _shap_factors(
        self, X: pd.DataFrame, raw_values: dict
    ) -> list[RiskFactor]:
        """
        Calcula SHAP para UNA fila y devuelve los top-5 factores.
        Usado en predict() individual. Para batch, usa _build_risk_factors_from_shap().
        Devuelve lista vacía si el explainer no está disponible o falla.
        """
        try:
            shap_values = self.explainer.shap_values(X)

            if isinstance(shap_values, list):
                shap_row = shap_values[1][0]
            elif len(shap_values.shape) == 2:
                shap_row = shap_values[0]
            else:
                shap_row = shap_values

            return self._build_risk_factors_from_shap(
                X.columns.tolist(), shap_row, raw_values
            )

        except Exception as exc:
            logger.warning("Error calculando SHAP: %s", exc)
            return []

    def _build_risk_factors_from_shap(
        self,
        feature_cols: list[str],
        shap_row: "np.ndarray",
        raw_values: dict,
    ) -> list[RiskFactor]:
        """
        Formatea un array SHAP pre-calculado en los top-5 RiskFactor.
        Reutilizado tanto por predict() individual como por predict_batch().
        """
        top5 = sorted(
            zip(feature_cols, shap_row),
            key=lambda kv: kv[1],
            reverse=True,
        )[:5]

        factors = []
        for feat, shap_val in top5:
            val_original = raw_values.get(feat, "N/A")
            impact = "AUMENTA_RIESGO" if shap_val > 0 else "DISMINUYE_RIESGO"

            if feat == "amt" and shap_val > 0:
                desc = f"El monto de {val_original} es inusualmente alto para este perfil."
            elif feat == "distance_km" and shap_val > 0:
                desc = f"La distancia ({val_original} km) indica una ubicación atípica."
            elif feat == "hour" and shap_val > 0:
                desc = f"La hora de transacción ({val_original}h) es sospechosa."
            else:
                desc = (
                    f"El valor de '{feat}' ({val_original}) "
                    f"impacta el score en {shap_val:.2f}."
                )

            factors.append(RiskFactor(
                feature_name=feat,
                feature_value=str(val_original),
                shap_value=float(shap_val),
                risk_description=desc,
                impact_direction=impact,
            ))
        return factors