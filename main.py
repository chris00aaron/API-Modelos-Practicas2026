# main.py
import sys
import os
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

# Agregar src al path para importaciones
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/'))

# Iniciar la aplicación FastAPI
app = FastAPI()

#Codigo base


#Inicializacion del servidor local
if __name__ == "__main__":
    import uvicorn  
    uvicorn.run(app, host="0.0.0.0", port=8000)