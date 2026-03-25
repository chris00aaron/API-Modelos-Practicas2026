# ── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Copiar y instalar dependencias en un entorno virtual para mayor seguridad
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Instalar librerías del sistema necesarias (ej. psycopg2 necesita libpq)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependencias instaladas desde el builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copiar el código fuente de la aplicación
COPY . .

# Exponer el puerto de la API de Inferencia/Modelos
EXPOSE 8000

# Comando de arranque con Uvicorn (modo producción, sin reload)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
