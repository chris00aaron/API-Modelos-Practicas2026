from pydantic import BaseModel

class OutputDataRetiroAtm(BaseModel):
    atm: int
    retiro : float


class OutputDataRetiroAtmExtraporanea(BaseModel):
    atm: int
    retiro : float
    prediction_date: str