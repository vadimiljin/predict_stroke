import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from fastapi import FastAPI
from pydantic import BaseModel


class Patient(BaseModel):
    age: int
    gender: str
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str


app = FastAPI()

model = joblib.load("stroke.pkl")


def predict(model, data):
    data = pd.DataFrame(data.dict(), index=[0])

    # Handle NaN values with SimpleImputer
    imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    label = model.predict(data)[0]
    spam_prob = model.predict_proba(data)

    return {"label": int(label), "probability": float(spam_prob[0][1].round(3))}


@app.post("/predict/")
async def stroke_prediction_query(patient: Patient):
    return predict(model, patient)
