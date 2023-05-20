import requests

url = "http://localhost:8000/predict/"

data = {
    "age": 50,
    "gender": "Male",
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "residence_type": "Urban",
    "avg_glucose_level": 75.0,
    "bmi": 25.0,
    "smoking_status": "formerly smoked",
}

response = requests.post(url, json=data)
print(response.json())
