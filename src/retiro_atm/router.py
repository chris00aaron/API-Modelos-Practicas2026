from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from retiro_atm.schema.input_retiro_atm import InputDataRetiroAtm
from retiro_atm.schema.output_retiro_atm import OutputDataRetiroAtm
from retiro_atm.service.service_prediction_retiro_atm import ServicioPredicticionRetiroAtm
import sys

router = APIRouter(
    prefix="/api/atm",
    tags=["Predición de Demanda de Efectivo en ATM"]
)

# Instanciamos el servicio una única vez al cargar el módulo
try:
    servicioPrediccionRetiro = ServicioPredicticionRetiroAtm()
except Exception as e:
    print(f"CRITICAL ERROR: No se pudo inicializar el servicio de predicción de retiros: {e}")
    sys.exit(1)


#Codigo base
@router.post("/predecir",tags=["Predicción del Retiro de Efectivo en ATM"])
async def predecir_temperatura(input_data: InputDataRetiroAtm ) -> OutputDataRetiroAtm:
    """
    Endpoint para predecir el monto ha retirar en un solo dia en un ATM.
    """
    try:
        return servicioPrediccionRetiro.predecir_retiro(input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.post("/v1/withdrawal", tags=["Predicción del Retiro de Efectivo en ATM"])
async def predecir_retiro(
    input_data: list[InputDataRetiroAtm]
) -> list[OutputDataRetiroAtm]:
    try:
        for i in input_data:
            print(f"{i}")

        return await run_in_threadpool(
            servicioPrediccionRetiro.predecir_retiro_lote,
            input_data
        )
    except Exception:
        print(f"FALLA {Exception}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")
