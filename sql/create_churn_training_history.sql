-- ============================================================
-- TABLA EXCLUSIVA PARA HISTORIAL DE ENTRENAMIENTO DE CHURN
-- ============================================================
-- Ejecutar en pgAdmin o psql contra la base de datos bankmind
-- ============================================================

-- 1. Crear la tabla
CREATE TABLE IF NOT EXISTS public.churn_training_history (
    id_churn_training   BIGSERIAL PRIMARY KEY,

    -- Cuándo se entrenó
    training_date       TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Razón: 'manual_training', 'performance_decay', 'scheduled_check'
    trigger_reason      VARCHAR(50) NOT NULL,

    -- ¿Este modelo está en producción?
    in_production       BOOLEAN DEFAULT FALSE,

    -- Identificador del modelo (MLflow Run ID u otro)
    model_version       VARCHAR(100),

    -- ══════════════════════════════════════════════
    -- Métricas del modelo entrenado
    -- ══════════════════════════════════════════════
    accuracy            NUMERIC(6,4),
    f1_score            NUMERIC(6,4),
    precision_score     NUMERIC(6,4),
    recall_score        NUMERIC(6,4),
    auc_roc             NUMERIC(6,4),

    -- ══════════════════════════════════════════════
    -- Información del dataset
    -- ══════════════════════════════════════════════
    train_samples       INTEGER,
    test_samples        INTEGER,

    -- ══════════════════════════════════════════════
    -- Resultados de evaluación (Monitor de Rendimiento)
    -- ══════════════════════════════════════════════
    evaluated_samples   INTEGER,
    true_positives      INTEGER,
    false_positives     INTEGER,
    true_negatives      INTEGER,
    false_negatives     INTEGER,
    recall_threshold    NUMERIC(4,3),

    -- ══════════════════════════════════════════════
    -- Metadata adicional (warnings, errores, etc.)
    -- ══════════════════════════════════════════════
    warnings            JSONB,

    created_at          TIMESTAMP DEFAULT NOW()
);

-- 2. Índices
CREATE INDEX IF NOT EXISTS idx_churn_training_date
    ON public.churn_training_history(training_date DESC);

CREATE INDEX IF NOT EXISTS idx_churn_training_production
    ON public.churn_training_history(in_production)
    WHERE in_production = TRUE;

CREATE INDEX IF NOT EXISTS idx_churn_training_trigger
    ON public.churn_training_history(trigger_reason);

-- 3. Comentarios
COMMENT ON TABLE public.churn_training_history
    IS 'Historial de entrenamiento exclusivo del módulo CHURN (fuga de clientes)';

-- ============================================================
-- RELACIÓN CON churn_predictions
-- ============================================================
-- Agregar columna FK en churn_predictions para saber qué
-- entrenamiento produjo el modelo que hizo cada predicción.
-- Es NULLABLE porque las predicciones existentes no tienen
-- este dato.
-- ============================================================

ALTER TABLE public.churn_predictions
    ADD COLUMN IF NOT EXISTS id_churn_training BIGINT;

ALTER TABLE public.churn_predictions
    ADD CONSTRAINT fk_churn_predictions_training
    FOREIGN KEY (id_churn_training)
    REFERENCES public.churn_training_history(id_churn_training)
    ON DELETE SET NULL;

COMMENT ON COLUMN public.churn_predictions.id_churn_training
    IS 'FK al entrenamiento que produjo el modelo usado para esta predicción';

-- ============================================================
-- VERIFICACIÓN
-- ============================================================
SELECT 'churn_training_history CREADA' AS resultado
WHERE EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_name = 'churn_training_history'
);

SELECT 'FK en churn_predictions CREADA' AS resultado
WHERE EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'churn_predictions'
    AND column_name = 'id_churn_training'
);
