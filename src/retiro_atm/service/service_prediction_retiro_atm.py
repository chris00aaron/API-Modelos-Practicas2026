from xgboost import XGBRegressor
from joblib import load
import numpy
from src.retiro_atm.schema import InputDataRetiroAtm
from src.retiro_atm.schema import OutputDataRetiroAtm


class ServicioPredicticionRetiroAtm():
    __path_model: str = r"src/retiro_atm/models_files/retiro_atm_model.joblib"
    __model : XGBRegressor

    def __init__(self):
        self.__model = load(self.__path_model)
    
    def predecir_retiro(self,input:InputDataRetiroAtm) -> OutputDataRetiroAtm:
        x = numpy.array([[
            input.dia_semana,
            input.quincena,
            input.semana_mes,
            input.dia_mes,
            input.lag1,
            input.lag5,
            input.lag7,
            input.lag11,
            input.tendencia_lags,
            input.esFeriado,
            input.caida_reciente,
            input.volatilidad_reciente,
            input.media_movil_3d,
            input.retiros_finde_anterior,
            input.lunes_post_finde_bajo,
            input.domingo_bajo,
            input.ubicacion,  
            input.ambiente
        ]])

        #Obtenemosla predicción del modelo 
        y_pred_log = self.__model.predict(x)
        y_pred_final = numpy.expm1(y_pred_log) # Volver a la escala de pesos/dólares
        
        #Casteamos el valor deseado a predecir
        prediccion_retiro = float(y_pred_final[0])
        return OutputDataRetiroAtm(retiro=prediccion_retiro)