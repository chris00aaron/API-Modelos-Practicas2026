from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from retiro_atm.schema.input_retiro_atm import InputDataRetiroAtm, InputDataRetiroAtmExtraporaneo
from retiro_atm.schema.output_retiro_atm import OutputDataRetiroAtm, OutputDataRetiroAtmExtraporanea
from retiro_atm.service.service_prediction_retiro_atm import ServicioPrediccionRetiroAtm
from retiro_atm.service.atm_model_provider import ModeloActualizandoseError
import sys
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/atm",
    tags=["Predición de Demanda de Efectivo en ATM"]
)

# Instanciamos el servicio una única vez al cargar el módulo
try:
    servicioPrediccionRetiro = ServicioPrediccionRetiroAtm()
except Exception as e:
    logger.error(f"CRITICAL ERROR: No se pudo inicializar el servicio de predicción de retiros: {e}")
    sys.exit(1)

#Codigo base
@router.post("/predecir",
    summary="Predecir el monto ha retirar en un solo dia en un ATM.",
    description="Endpoint para predecir el monto ha retirar en un solo dia en un ATM."
)
async def predecir_temperatura(input_data: InputDataRetiroAtm ) -> OutputDataRetiroAtm:
    """
    Endpoint para predecir el monto ha retirar en un solo dia en un ATM.
    """
    try:
        return servicioPrediccionRetiro.predecir_retiro(input_data)
    except ModeloActualizandoseError as e:
        logger.warning(f"Modelo en actualización => {e}")
        raise HTTPException(status_code=503, detail="El modelo está siendo actualizado. Por favor, intente nuevamente más tarde.")
    
    except Exception as e:
        logger.error(f"Error en predecir_retiro: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.post("/v1/withdrawal",
    summary="Predecir el monto ha retirar en un solo dia en un ATM.",
    description="Endpoint para predecir el monto ha retirar en un solo dia en un ATM.")
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
    except ModeloActualizandoseError as e:
        logger.warning(f"Modelo en actualización => {e}")
        raise HTTPException(status_code=503, detail="El modelo está siendo actualizado. Por favor, intente nuevamente más tarde.")
    
    except Exception as e:
        logger.error(f"Error en predecir_retiro_lote: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")
    
    
@router.post("/v1/withdrawal/off-time",
    summary="Predecir el monto ha retirar en un solo dia en un ATM.",
    description="Endpoint para predecir el monto ha retirar en un solo dia en un ATM.")
async def predecir_retiro_extraporaneo(
    input_data: list[InputDataRetiroAtmExtraporaneo]
) -> list[OutputDataRetiroAtmExtraporanea]:
    try:
        for i in input_data:
            print(f"{i}")

        return await run_in_threadpool(
            servicioPrediccionRetiro.predecir_retiro_lote_extraporaneo,
            input_data
        )
    except ModeloActualizandoseError as e:
        logger.warning(f"Modelo en actualización => {e}")
        raise HTTPException(status_code=503, detail="El modelo está siendo actualizado. Por favor, intente nuevamente más tarde.")
    
    except Exception as e:
        logger.error(f"Error en predecir_retiro_lote: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")
    

@router.post("/v1/withdrawal/update",
    summary="Actualizar el modelo predictivo desde DagsHub.",
    description="Endpoint para actualizar el modelo predictivo desde DagsHub.")
async def actualizar_modelo_atm(background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(servicioPrediccionRetiro.provider.recargar_modelo)
        return {"mensaje": "Actualización iniciada. El nuevo modelo estará disponible en unos momentos."}
    except Exception as e:
        logger.error(f"Error en actualizar modelo: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@router.get("/v1/withdrawal/info",
    summary="Obtener información del modelo predictivo.",
    description="Endpoint para obtener información del modelo predictivo.")
async def obtener_info_modelo():
    try:
        return servicioPrediccionRetiro.provider.info_modelo()
    except Exception as e:
        logger.error(f"Error en obtener información del modelo: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")