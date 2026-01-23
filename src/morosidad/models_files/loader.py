# src/morosidad/models_files/loader.py
import os
import joblib

# Ruta al archivo del modelo
_MODELO_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# Variables globales (singleton)
_modelo = None
_explainer = None
_version = "v1.0"  # Valor por defecto


def cargar_modelo():
    """
    Carga el modelo de morosidad y el explainer SHAP desde el archivo .pkl.
    El archivo debe ser un diccionario con las claves 'modelo' y 'explainer'.
    Se carga una sola vez (singleton pattern).
    
    Returns:
        Tupla (modelo, explainer), o (None, None) si no existe el archivo.
    """
    global _modelo, _explainer, _version
    
    if _modelo is None:
        if os.path.exists(_MODELO_PATH):
            model_pack = joblib.load(_MODELO_PATH)
            
            # Verificar si es un diccionario con las claves esperadas
            if isinstance(model_pack, dict):
                # Claves del script de empaquetado del usuario
                _modelo = model_pack.get('modelo_prediccion')
                _explainer = model_pack.get('shap_explainer')
                
                # Extraer metadatos si existen
                meta = model_pack.get('meta_info', {})
                if isinstance(meta, dict):
                    _version = meta.get('version', "v1.0")
                
                print(f"[OK] Modelo de morosidad cargado desde: {_MODELO_PATH}")
                print(f"[INFO] Versión del modelo: {_version}")
                
                if _explainer:
                    print("[OK] SHAP Explainer cargado correctamente.")
                else:
                    print("[WARN] No se encontró 'shap_explainer' en el archivo .pkl")
            else:
                # Retrocompatibilidad: si es el modelo directamente
                _modelo = model_pack
                print(f"[OK] Modelo de morosidad cargado desde: {_MODELO_PATH}")
                print("[WARN] El archivo no contiene explainer SHAP ni metadatos")
        else:
            print(f"[WARN] Archivo de modelo no encontrado: {_MODELO_PATH}")
            print("   Por favor, agrega tu archivo .pkl en la carpeta models_files/")
    
    return _modelo, _explainer


def obtener_modelo():
    """
    Obtiene el modelo ya cargado.
    Si no está cargado, intenta cargarlo.
    
    Returns:
        El modelo cargado.
    
    Raises:
        RuntimeError: Si el modelo no está disponible.
    """
    modelo, _ = cargar_modelo()
    if modelo is None:
        raise RuntimeError(
            "El modelo de morosidad no está disponible. "
            "Asegúrate de agregar el archivo model.pkl "
            "en la carpeta src/morosidad/models_files/"
        )
    return modelo


def obtener_explainer():
    """
    Obtiene el explainer SHAP ya cargado.
    Si no está cargado, intenta cargarlo.
    
    Returns:
        El explainer SHAP, o None si no está disponible.
    """
    _, explainer = cargar_modelo()
    return explainer


def obtener_version():
    """
    Obtiene la versión del modelo cargado.
    
    Returns:
        String con la versión (ej: "v1.0")
    """
    cargar_modelo()
    return _version


