# src/morosidad/models_files/loader.py
import os
import joblib

# Ruta al archivo del modelo
_MODELO_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# Variable global para el modelo (singleton)
_modelo = None


def cargar_modelo():
    """
    Carga el modelo de morosidad desde el archivo .pkl.
    El modelo se carga una sola vez (singleton pattern).
    
    Returns:
        El modelo cargado, o None si no existe el archivo.
    """
    global _modelo
    
    if _modelo is None:
        if os.path.exists(_MODELO_PATH):
            _modelo = joblib.load(_MODELO_PATH)
            print(f"[OK] Modelo de morosidad cargado desde: {_MODELO_PATH}")
        else:
            print(f"[WARN] Archivo de modelo no encontrado: {_MODELO_PATH}")
            print("   Por favor, agrega tu archivo .pkl en la carpeta models_files/")
    
    return _modelo


def obtener_modelo():
    """
    Obtiene el modelo ya cargado.
    Si no está cargado, intenta cargarlo.
    
    Returns:
        El modelo cargado.
    
    Raises:
        RuntimeError: Si el modelo no está disponible.
    """
    modelo = cargar_modelo()
    if modelo is None:
        raise RuntimeError(
            "El modelo de morosidad no está disponible. "
            "Asegúrate de agregar el archivo modelo_morosidad.pkl "
            "en la carpeta src/morosidad/models_files/"
        )
    return modelo
