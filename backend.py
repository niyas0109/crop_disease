import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="ML Backend API")

# Train and load model in memory
data = load_iris()
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(data.data, data.target)
target_names = ["Setosa", "Versicolor", "Virginica"]


class PredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict")
def predict(data: PredictionInput):
    features = np.array(
        [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    )
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    return {
        "prediction_label": target_names[prediction],
        "confidence": float(probabilities[prediction]),
    }
