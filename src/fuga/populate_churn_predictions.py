"""
populate_churn_predictions.py
=============================================================================
Stratified Priority Sampling — Módulo Fuga de Clientes (BankMind v2)
=============================================================================
Propósito:
    Poblar la tabla `churn_predictions` con predicciones reales del modelo ML
    para ~500 clientes seleccionados estratégicamente, permitiendo que los
    dashboards de "Inteligencia de Riesgo" muestren patrones reales.

Estrategia de selección (3 capas):
    CAPA 1 — VIPs Críticos (~100):
        balance > 100.000 EUR AND (inactivo OR solo 1 producto)
        → Clientes de alto valor con señales de alarma. Siempre presentes.

    CAPA 2 — Perfiles de Riesgo Potencial (~150):
        1 producto OR tenure < 2 años OR credit_score < 600
        → Características históricamente correlacionadas con churn bancario.

    CAPA 3 — Muestra Aleatoria Estratificada (~250):
        Aleatorio del resto, proporcional por país (France/Spain/Germany).
        → Garantiza representatividad geográfica en los gráficos.

Requisitos:
    - PostgreSQL BankMindDB corriendo en localhost:5432
    - Python FastAPI ML corriendo en localhost:8000
    - pip install psycopg2-binary requests

Salida:
    - Inserta predicciones en `churn_predictions`
    - Genera `populate_churn_predictions.sql` (misma carpeta que este script)
      con todos los INSERT reproducibles

Uso:
    python populate_churn_predictions.py
=============================================================================
"""

import psycopg2
import requests
import random
from pathlib import Path
from datetime import datetime, timezone

# ── Configuración ────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "BankMindDB",
    "user":     "postgres",
    "password": "1234",
}

PYTHON_API_URL = "http://localhost:8000/churn/predict"

# Tamaños objetivo por capa
TARGET_LAYER_1 = 100   # VIPs críticos
TARGET_LAYER_2 = 150   # Perfiles de riesgo
TARGET_LAYER_3 = 250   # Estratificado por país

# Proporciones geográficas para Capa 3 (basadas en la distribución real de la BD)
# France: 1935, Spain: 1570, Germany: 1567  →  total 5072 sin USA
GEO_WEIGHTS = {"France": 0.38, "Spain": 0.31, "Germany": 0.31}

# El SQL se genera en la misma carpeta que este script
SQL_OUTPUT_FILE = Path(__file__).parent / "populate_churn_predictions.sql"

# ── Query base: clientes con cuenta y país válido (France/Spain/Germany) ─────
BASE_QUERY = """
    SELECT
        c.id_customer,
        c.age,
        co.country_description   AS geography,
        g.gender_description     AS gender,
        ad.credit_score,
        ad.balance,
        ad.estimated_salary,
        ad.num_of_products,
        ad.tenure,
        CASE WHEN ad.has_cr_card      THEN 1 ELSE 0 END AS has_cr_card,
        CASE WHEN ad.is_active_member THEN 1 ELSE 0 END AS is_active_member,
        CASE WHEN ad.exited           THEN 1 ELSE 0 END AS exited
    FROM customer c
    JOIN account_details ad ON ad.id_customer = c.id_customer
    JOIN country          co ON co.id_country  = c.id_country
    JOIN gender           g  ON g.id_gender    = c.id_gender
    WHERE co.country_description IN ('France', 'Spain', 'Germany')
"""


def fetch_customers(conn, extra_where: str = "", order: str = "", limit: int = 0,
                    exclude_ids: set = None) -> list[dict]:
    """Ejecuta la query base con filtros adicionales y retorna lista de dicts."""
    sql = BASE_QUERY
    if exclude_ids:
        ids_str = ",".join(str(i) for i in exclude_ids)
        sql += f" AND c.id_customer NOT IN ({ids_str})"
    if extra_where:
        sql += f" AND ({extra_where})"
    if order:
        sql += f" ORDER BY {order}"
    if limit:
        sql += f" LIMIT {limit}"

    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_already_predicted(conn) -> set:
    """Retorna los id_customer que ya tienen al menos una predicción."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT id_customer FROM churn_predictions")
        return {row[0] for row in cur.fetchall()}


def call_model(customer: dict) -> dict | None:
    """Llama al Python API y retorna el resultado o None si falla."""
    payload = {
        "CreditScore":     customer["credit_score"],
        "Geography":       customer["geography"],
        "Gender":          customer["gender"],
        "Age":             customer["age"],
        "Tenure":          customer["tenure"],
        "Balance":         float(customer["balance"]),
        "NumOfProducts":   customer["num_of_products"],
        "HasCrCard":       customer["has_cr_card"],
        "IsActiveMember":  customer["is_active_member"],
        "EstimatedSalary": float(customer["estimated_salary"]),
        "Exited":          customer["exited"],
    }
    try:
        resp = requests.post(PYTHON_API_URL, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"    [ERROR] Cliente {customer['id_customer']}: {e}")
        return None


def insert_prediction(conn, customer: dict, result: dict) -> dict:
    """Inserta la predicción en churn_predictions y retorna la fila insertada."""
    churn_prob   = result["churn_probability"]
    is_churn     = bool(result["is_churn"])
    risk_level   = result["risk_level"]
    confidence   = result["prediction_confidence"]
    model_ver    = result.get("model_version", "v1.0.0")
    balance      = float(customer["balance"])
    now          = datetime.now(timezone.utc).replace(tzinfo=None)

    sql = """
        INSERT INTO churn_predictions
            (id_customer, churn_probability, is_churn, risk_level,
             prediction_confidence, customer_value, model_version, prediction_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id_churn_prediction
    """
    with conn.cursor() as cur:
        cur.execute(sql, (
            customer["id_customer"], churn_prob, is_churn, risk_level,
            confidence, balance, model_ver, now
        ))
        new_id = cur.fetchone()[0]
    conn.commit()

    return {
        "id_churn_prediction":   new_id,
        "id_customer":           customer["id_customer"],
        "churn_probability":     churn_prob,
        "is_churn":              is_churn,
        "risk_level":            risk_level,
        "prediction_confidence": confidence,
        "customer_value":        balance,
        "model_version":         model_ver,
        "prediction_date":       now.isoformat(),
    }


def process_layer(conn, name: str, customers: list[dict],
                  sql_inserts: list[str]) -> list[dict]:
    """Procesa una capa: llama al modelo e inserta en BD."""
    print(f"\n{'='*60}")
    print(f"  {name}  ({len(customers)} clientes)")
    print(f"{'='*60}")
    inserted = []
    for i, c in enumerate(customers, 1):
        print(f"  [{i:>3}/{len(customers)}] Cliente {c['id_customer']} "
              f"| {c['geography']:<7} | age={c['age']} "
              f"| balance={float(c['balance']):>10,.0f} "
              f"| prods={c['num_of_products']} "
              f"| score={c['credit_score']}", end="  >>  ")
        result = call_model(c)
        if result is None:
            print("SKIP")
            continue
        row = insert_prediction(conn, c, result)
        print(f"risk={result['risk_level']:<5} prob={result['churn_probability']:.4f}")

        # Genera el INSERT SQL reproducible
        sql_inserts.append(
            f"INSERT INTO churn_predictions "
            f"(id_customer, churn_probability, is_churn, risk_level, "
            f"prediction_confidence, customer_value, model_version, prediction_date) VALUES "
            f"({row['id_customer']}, {row['churn_probability']}, "
            f"{'TRUE' if row['is_churn'] else 'FALSE'}, "
            f"'{row['risk_level']}', {row['prediction_confidence']}, "
            f"{row['customer_value']}, "
            f"'{row['model_version']}', "
            f"'{row['prediction_date']}');"
        )
        inserted.append(row)
    return inserted


def stratified_sample(customers: list[dict], target: int,
                       weights: dict[str, float]) -> list[dict]:
    """Muestra aleatoria estratificada por país."""
    by_country: dict[str, list] = {}
    for c in customers:
        by_country.setdefault(c["geography"], []).append(c)

    sampled = []
    for country, pct in weights.items():
        pool = by_country.get(country, [])
        random.shuffle(pool)
        n = round(target * pct)
        sampled.extend(pool[:n])

    # Si por redondeo quedamos cortos, completamos aleatoriamente
    if len(sampled) < target:
        all_remaining = [c for c in customers if c not in sampled]
        random.shuffle(all_remaining)
        sampled.extend(all_remaining[:target - len(sampled)])

    return sampled[:target]


def write_sql_file(sql_inserts: list[str], stats: dict):
    """Escribe el archivo SQL reproducible."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    header = f"""-- =============================================================================
-- SCRIPT: Poblar churn_predictions — Stratified Priority Sampling
-- Base de datos: BankMindDB (PostgreSQL)
-- Generado: {now_str}
-- Total predicciones insertadas: {stats['total']}
--
-- Distribución:
--   Capa 1 — VIPs Críticos          : {stats['layer1']} clientes
--   Capa 2 — Perfiles de Riesgo     : {stats['layer2']} clientes
--   Capa 3 — Estratificado por País : {stats['layer3']} clientes
--
-- Distribución por nivel de riesgo:
--   Alto  (> 75%): {stats['alto']}
--   Medio (45-75%): {stats['medio']}
--   Bajo  (< 45%): {stats['bajo']}
--
-- NOTA: Este script inserta los resultados exactos del modelo XGBoost
--       ejecutados sobre los datos reales de la BD en la fecha de generación.
--       Si los datos de clientes cambian, regenerar con populate_churn_predictions.py
-- =============================================================================

BEGIN;

"""
    footer = "\nCOMMIT;\n"

    with open(SQL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(sql_inserts))
        f.write(footer)

    print(f"\n  SQL guardado en: {SQL_OUTPUT_FILE}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    random.seed(42)  # Reproducibilidad

    print("\n" + "="*60)
    print("  BankMind v2 — Populate churn_predictions")
    print("  Stratified Priority Sampling (~500 clientes)")
    print("="*60)

    conn = psycopg2.connect(**DB_CONFIG)

    # Clientes que ya tienen predicción → excluir
    already = get_already_predicted(conn)
    print(f"\nClientes ya analizados (se excluiran): {len(already)}")

    sql_inserts: list[str] = []

    # ── CAPA 1: VIPs Críticos ────────────────────────────────────────────────
    layer1_where = (
        "ad.balance > 100000 "
        "AND (ad.is_active_member = false OR ad.num_of_products = 1)"
    )
    layer1_customers = fetch_customers(
        conn,
        extra_where=layer1_where,
        order="ad.balance DESC",
        limit=TARGET_LAYER_1,
        exclude_ids=already,
    )
    layer1_ids = {c["id_customer"] for c in layer1_customers}
    rows1 = process_layer(conn, "CAPA 1 - VIPs Criticos", layer1_customers, sql_inserts)

    # ── CAPA 2: Perfiles de Riesgo ───────────────────────────────────────────
    layer2_where = (
        "(ad.num_of_products = 1 OR ad.tenure < 2 OR ad.credit_score < 600)"
    )
    layer2_candidates = fetch_customers(
        conn,
        extra_where=layer2_where,
        order="ad.credit_score ASC",
        limit=TARGET_LAYER_2 * 3,          # candidatos extra para elegir
        exclude_ids=already | layer1_ids,
    )
    random.shuffle(layer2_candidates)
    layer2_customers = layer2_candidates[:TARGET_LAYER_2]
    layer2_ids = {c["id_customer"] for c in layer2_customers}
    rows2 = process_layer(conn, "CAPA 2 - Perfiles de Riesgo", layer2_customers, sql_inserts)

    # ── CAPA 3: Muestra Aleatoria Estratificada ───────────────────────────────
    layer3_candidates = fetch_customers(
        conn,
        exclude_ids=already | layer1_ids | layer2_ids,
    )
    layer3_customers = stratified_sample(layer3_candidates, TARGET_LAYER_3, GEO_WEIGHTS)
    rows3 = process_layer(conn, "CAPA 3 - Estratificado por Pais", layer3_customers, sql_inserts)

    # ── Resumen final ─────────────────────────────────────────────────────────
    all_rows = rows1 + rows2 + rows3
    alto  = sum(1 for r in all_rows if r["risk_level"] == "Alto")
    medio = sum(1 for r in all_rows if r["risk_level"] == "Medio")
    bajo  = sum(1 for r in all_rows if r["risk_level"] == "Bajo")

    stats = {
        "total":  len(all_rows),
        "layer1": len(rows1),
        "layer2": len(rows2),
        "layer3": len(rows3),
        "alto":   alto,
        "medio":  medio,
        "bajo":   bajo,
    }

    print(f"\n{'='*60}")
    print(f"  RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"  Total insertados : {stats['total']}")
    print(f"  Capa 1 (VIPs)    : {stats['layer1']}")
    print(f"  Capa 2 (Riesgo)  : {stats['layer2']}")
    print(f"  Capa 3 (Estrat.) : {stats['layer3']}")
    print(f"  Riesgo Alto      : {alto}")
    print(f"  Riesgo Medio     : {medio}")
    print(f"  Riesgo Bajo      : {bajo}")

    write_sql_file(sql_inserts, stats)
    conn.close()
    print("\n  Listo! Recarga el dashboard de Inteligencia de Riesgo.\n")


if __name__ == "__main__":
    main()
