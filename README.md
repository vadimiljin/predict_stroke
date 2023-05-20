# Stroke Risk Identification: A Machine Learning Approach

## **Objective**: Unveiling Stroke Risks 

In this project, I aim to uncover individual characteristics that may increase the risk of stroke. The main goal is to fine-tune a machine learning model to provide precise predictions of stroke risk.

## **Methods Used**:

To attain this goal, I employed a wide range of strategies and techniques:

- **Exploratory Data Analysis (EDA)**: EDA serves as the foundational step, providing an initial understanding of the data and highlighting key features for model building.

- **Data Visualization**: Graphical representation of the data aids in understanding data distributions and extracting insights.

- **Statistical Inference**: I used statistical tests to identify any significant differences between individuals who experienced a stroke and those who did not.

- **Feature Engineering**: This involves creating or modifying features to enhance the performance of the machine learning model.

- **Machine Learning Models**: A variety of models such as Logistic Regression, SVC, k-NN, Decision Tree, and Random Forest were implemented.

- **Pipeline Creation**: Pipelines were incorporated to simplify the workflow and minimize errors.

- **Hyperparameter Tuning**: Depending on the model, GridSearchCV or RandomizedSearchCV was used to optimize hyperparameters.

- **Resampling Techniques**: To tackle the imbalance in the dataset, undersampling and oversampling techniques were used.

- **Gradient Boosting Models**: Advanced models like XGBoost and CatBoost were used.

- **Feature Importance Evaluation**: I used permutation_importance and the SHAP (SHapley Additive exPlanations) library to ascertain feature importance.

## **Findings and Results**:

After extensive experimentation, the **XGBoost** model was found to deliver the best performance for stroke prediction. The optimal parameters included:

- Subsample = 0.5
- N_estimators = 100
- Max_depth = 3
- Learning_rate = 0.01
- Colsample_bytree = 0.7
- Colsample_bylevel = 0.6
- Scale_pos_weight = 20

Performance metrics:

- Positive class: Precision - 14.13%, Recall - 85.48%
- Negative class: Precision - 99.17%, Recall - 77.02%
- Overall Accuracy: 77.38%.

The most impactful features in predicting stroke were identified as age, average glucose level, unmarried status, unknown smoking status, hypertension, and heart disease.

## **Practical Implications**:

Imbalance in the data (with a negative-to-positive class ratio of 22.442) is a common issue with medical datasets, including this one.

In practice, it's crucial to have a model that ensures accurate diagnosis, thereby preventing potentially life-threatening oversights. The model I developed aids in achieving this, minimizing the likelihood of false negatives while keeping false positives to a manageable level. 

This also aligns with a business perspective, as a correctly diagnosed patient implies optimized hospital stays and lower costs due to fewer unnecessary medical tests.

## **API Utilization**:

To make the stroke prediction model easily accessible and usable, I have implemented an API using FastAPI. 

### Installation:

You can run the API locally by following these steps:

1. Clone the repository:
   ```
   git clone https://github.com/vadimiljin/predict_stroke.git
   ```

2. Navigate into the directory:
   ```
   cd app
   ```
3. Install the requirements:
   ```
   pip install -r requirements.txt
   ```
4. Run the API:
   ```
   uvicorn main:app --reload
   ```

Now, the API is running on your local machine at `http://localhost:8000`.

### Sample API Request:

CURL

```bash
curl -X POST -H "Content-Type: application/json" -d '{
    "age": 50,
    "gender": "Male",
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "residence_type": "Urban",
    "avg_glucose_level": 75.0,
    "bmi": 25.0,
    "smoking_status": "formerly smoked"
}' http://localhost:8000/predict/
```

PYTHON

```python
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
```

### Sample API Response:

```json
{'label': 0, 'probability': 0.44200000166893005}
```
- The `label` key represents the predicted label or class. In this case, the value of 0 suggests that the model predicted a negative outcome or a low likelihood of stroke.
- The `probability` key provides the probability associated with the predicted label. The value of 0.44200000166893005 indicates the estimated probability of the predicted outcome. In this case, it suggests that there is a 44.2% probability of the negative outcome (no stroke) according to my model.