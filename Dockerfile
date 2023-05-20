FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9
COPY ./app /app
WORKDIR /app
RUN pip install scikit-learn==1.0.2 joblib==1.2.0 xgboost==0.90 pandas==1.4.2 mlxtend==0.21.0 pydantic==1.10.2