from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel


# Создаём FastAPI приложение
app = FastAPI(
    title="ML Prediction API",
    description="API для предсказания прочности при растяжении (strength) композиционных материалов в зависимости от шага нашивки (patch_spacing) и поверхностной плотности (surface_density)",
    version="1.0.0",
)


# Путь к модели
MODEL_PATH = Path(__file__).parent.parent / "model" / "my_model.joblib"


# Загружаем модель при запуске приложения
model = joblib.load(MODEL_PATH)


# Описываем входные данные
class PredictionRequest(BaseModel):
    patch_spacing: float
    surface_density: float


# Главная страница
@app.get("/")
def root():
    return {
        "message": "ML Prediction API is running"
    }


# Endpoint для предсказания
@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = model.predict([
        [
            request.patch_spacing,
            request.surface_density,
        ]
    ])

    return {
        "prediction": float(prediction[0])
    }