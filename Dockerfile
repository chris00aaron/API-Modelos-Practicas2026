# ============================================================
# BankMind API Predicción — Módulo ATM
# Multi-stage build · Python 3.11.9
# ============================================================

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.11.9-slim AS builder

WORKDIR /build

COPY requirements.txt .

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11.9-slim

LABEL maintainer="BankMind Team"
LABEL description="API de predicción de retiro de efectivo en cajeros automáticos"

WORKDIR /app

# Copiar solo el virtualenv compilado
COPY --from=builder /opt/venv /opt/venv

# Copiar el código fuente
COPY . .

# Activar el virtualenv en el PATH
ENV PATH="/opt/venv/bin:$PATH"

# Variables de entorno por defecto
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOME=/tmp

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/vivo')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]