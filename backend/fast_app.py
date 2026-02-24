from fastapi import FastAPI
from src.predict_pipeline import PredictPipeline
import pandas as pd

app = FastAPI()

predict_pipeline = PredictPipeline()

@app.get("/")
def home():
    return {"message": "API is working"}

@app.post("/predict")
def get_prediction(data: dict):

    # Convert JSON to DataFrame
    features = pd.DataFrame([data])

    result = predict_pipeline.predict(features)

    return {"prediction": float(result[0])}