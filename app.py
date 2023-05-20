from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from typing import List
import pandas as pd

app = FastAPI()


model_name = "Stroke Prediction"


class StrokePrediction(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

    class Config:
        schema_extra = {
            "example": {
                "gender": "Female/Male/Other",
                "age": "25.0/any_float_number",
                "hypertension": "1/0",
                "heart_disease": "1/0",
                "ever_married": "Yes/No",
                "work_type": "Govt_job/Self-employed/Private/children/Never_worked",
                "residence_type": "Rural/Urban",
                "avg_glucose_level": "85.0/any_float_number",
                "bmi": "25.0/any_float_number",
                "smoking_status": "never smoked/Unknown/smokes/formerly smoked",
            }
        }


class Output(BaseModel):
    prediction: int


load_model = pickle.load(open("stroke.pkl", "rb"))


@app.get("/")
async def model_info():
    return {"name": model_name}


@app.get("/health")
async def service_health():
    return {"ok"}


@app.post("/predict", response_model=Output)
async def get_prediction(inputs: List[StrokePrediction]):
    for input in inputs:
        df = pd.DataFrame([input.dict()])
        prediction = load_model.predict(df)[0]
    return {"prediction": prediction}
