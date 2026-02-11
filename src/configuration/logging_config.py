from logging.config import dictConfig
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
        # Log para errores de toda la app (Cualquier módulo)
        "file_errors": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOG_DIR / "errors.log"),
            "level": "ERROR", # Solo captura ERROR o superior
            "maxBytes": 5_000_000,
            "backupCount": 5,
            "formatter": "default",
        },
        "file_general": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOG_DIR / "general.log"),
            "maxBytes": 5_000_000,
            "backupCount": 3,
            "formatter": "default",
        },
        # Handlers específicos
        "file_fraude": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOG_DIR / "fraude.log"),
            "maxBytes": 5_000_000,
            "formatter": "default",
        },
        "file_fuga": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOG_DIR / "fuga.log"),
            "maxBytes": 5_000_000,
            "formatter": "default",
        },
        "file_morosidad": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOG_DIR / "morosidad.log"),
            "maxBytes": 5_000_000,
            "formatter": "default",
        },
        "file_retiro_atm": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(LOG_DIR / "retiro_atm.log"),
            "maxBytes": 5_000_000,
            "formatter": "default",
        },
    },
    "loggers": {
        "fraude": {
            "handlers": ["file_fraude", "file_errors"], 
            "level": "INFO", 
            "propagate": False
        },
        "fuga": {
            "handlers": ["file_fuga", "file_errors"], 
            "level": "INFO", 
            "propagate": False
        },
        "morosidad": {
            "handlers": ["file_morosidad", "file_errors"], 
            "level": "INFO", 
            "propagate": False
        },
        "retiro_atm": {
            "handlers": ["file_retiro_atm", "file_errors"], 
            "level": "INFO", 
            "propagate": False
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file_general", "file_errors"],
    },
}

def setup_logging():
    dictConfig(LOGGING_CONFIG)