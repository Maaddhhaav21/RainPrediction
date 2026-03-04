import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        location: str,
        temperature: float,
        humidity: float,
        wind_speed: float,
        precipitation: float,
        cloud_cover: float,
        pressure: float,
    ):

        self.location = location
        self.temperature = temperature
        self.humidity = humidity
        self.wind_speed = wind_speed
        self.precipitation = precipitation
        self.cloud_cover = cloud_cover
        self.pressure = pressure

    def get_data_as_dataframe(self):

        try:

            custom_data_input_dict = {
                "location": [self.location],
                "temperature": [self.temperature],
                "humidity": [self.humidity],
                "wind_speed": [self.wind_speed],
                "precipitation": [self.precipitation],
                "cloud_cover": [self.cloud_cover],
                "pressure": [self.pressure],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)