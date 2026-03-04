from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request,
            location: str,
            temperature: float,
            humidity: float,
            wind_speed: float,
            precipitation: float,
            cloud_cover: float,
            pressure: float):

    data = CustomData(
        location=location,
        temperature=temperature,
        humidity=humidity,
        wind_speed=wind_speed,
        precipitation=precipitation,
        cloud_cover=cloud_cover,
        pressure=pressure
    )

    pred_df = data.get_data_as_dataframe()

    pipeline = PredictPipeline()
    result = pipeline.predict(pred_df)

    prediction = "Rain Expected" if result[0] == 1 else "No Rain"

    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "results": prediction
        }
    )