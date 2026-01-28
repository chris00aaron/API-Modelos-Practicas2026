import os
from xgboost import XGBRegressor
from joblib import load
import numpy
from src.retiro_atm.schema import InputDataRetiroAtm
from src.retiro_atm.schema import OutputDataRetiroAtm


class ServicioPredicticionRetiroAtm():
    __model : XGBRegressor

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path_model = os.path.join(base_dir, "models_files", "retiro_atm_model.joblib")
        self.__model = load(path_model)
    
    def predecir_retiro(self,input:InputDataRetiroAtm) -> OutputDataRetiroAtm:
        x = numpy.array([[
            input.diaSemana,
            input.tendencia_lags,
            input.lag1,
            input.lag5,
            input.lag11,
            input.caida_reciente,
            input.retiros_finde_anterior,
            input.retiros_domingo_anterior,
            input.ratio_finde_vs_semana,
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