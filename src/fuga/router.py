from fastapi import APIRouter, HTTPException
from fuga.service.churn_service import churn_service
from fuga.schema.inputs import ChurnInput

router = APIRouter(
    prefix="/churn",
    tags=["Churn Prediction"]
)

@router.post("/predict")
def predict_churn(data: ChurnInput):
    input_data = data.model_dump()
    try:
        result = churn_service.predict(input_data)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
