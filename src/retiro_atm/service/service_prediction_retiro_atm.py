import numpy as np
import logging

from .atm_model_provider import AtmModelProvider, ModeloActualizandoseError
from src.retiro_atm.schema import InputDataRetiroAtm, OutputDataRetiroAtm

logger = logging.getLogger(__name__)

class ServicioPrediccionRetiroAtm:
    def __init__(self):
        self.provider = AtmModelProvider()

    def predecir_retiro(self, input_data: InputDataRetiroAtm) -> OutputDataRetiroAtm:
        """Realiza una predicción individual consumiendo el modelo en RAM."""
        # El provider decide si entrega el modelo o lanza la excepción de 'actualizando'
        modelo = self.provider.obtener_modelo()
        
        x = self._build_features(input_data)
        y_pred_log = modelo.predict(x)
        prediccion_final = float(np.expm1(y_pred_log)[0])

        return OutputDataRetiroAtm(atm=input_data.atm, retiro=prediccion_final)


    def predecir_retiro_lote(self, inputs: list[InputDataRetiroAtm]) -> list[OutputDataRetiroAtm]:
        """Realiza predicciones en lote de forma eficiente."""
        modelo = self.provider.obtener_modelo()
        logger.info(f"Procesando lote de {len(inputs)} registros.")

        # Optimizamos la creación de la matriz
        X = np.array([self._build_features(inp).ravel() for inp in inputs])

        y_pred_log = modelo.predict(X)
        y_pred_final = np.expm1(y_pred_log)

        return [
            OutputDataRetiroAtm(atm=inputs[i].atm, retiro=float(y_pred_final[i]))
            for i in range(len(inputs))
        ]

    def _build_features(self, data: InputDataRetiroAtm) -> np.ndarray:
        return np.array([[ 
            data.diaSemana, data.tendencia_lags, data.lag1, data.lag5,
            data.lag11, data.caida_reciente, data.retiros_finde_anterior,
            data.retiros_domingo_anterior, data.ratio_finde_vs_semana,
            data.domingo_bajo, data.ubicacion, data.ambiente
        ]])