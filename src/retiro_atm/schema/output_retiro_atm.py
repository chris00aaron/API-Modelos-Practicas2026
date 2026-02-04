from pydantic import BaseModel

class OutputDataRetiroAtm(BaseModel):
    atm: int
    retiro : float