from pydantic import BaseModel, Field

class ChurnInput(BaseModel):
    CreditScore: int = Field(..., description="Puntaje crediticio", example=600)
    Geography: str = Field(..., description="País (France, Spain, Germany)", example="France")
    Gender: str = Field(..., description="Género (Male, Female)", example="Male")
    Age: int = Field(..., description="Edad del cliente", example=40)
    Tenure: int = Field(..., description="Años siendo cliente", example=3)
    Balance: float = Field(..., description="Saldo en cuenta", example=60000.0)
    NumOfProducts: int = Field(..., description="Número de productos", example=2)
    HasCrCard: int = Field(..., description="Tiene tarjeta (1=Sí, 0=No)", example=1)
    IsActiveMember: int = Field(..., description="Es activo (1=Sí, 0=No)", example=1)
    EstimatedSalary: float = Field(..., description="Salario estimado", example=50000.0)

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 40,
                "Balance": 60000.0,
                "CreditScore": 600,
                "EstimatedSalary": 50000.0,
                "Gender": "Male",
                "Geography": "France",
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "NumOfProducts": 2,
                "Tenure": 3
            }
        }