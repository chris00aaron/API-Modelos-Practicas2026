from pydantic import BaseModel

class InputDataRetiroAtm(BaseModel):
    dia_semana: int
    quincena: int
    semana_mes: int
    dia_mes: float
    lag1: float
    lag5 :float
    lag7 : float
    lag11 : float
    tendencia_lags : float
    esFeriado : int
    caida_reciente : int  
    volatilidad_reciente : float
    media_movil_3d : float
    retiros_finde_anterior : float
    lunes_post_finde_bajo : int
    domingo_bajo : int
    ubicacion : int  
    ambiente : int