# src/morosidad/service/__init__.py
from .morosidad_service import predecir_morosidad, predecir_morosidad_batch

__all__ = ["predecir_morosidad", "predecir_morosidad_batch"]

