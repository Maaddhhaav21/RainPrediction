from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    location = request.form.get('location')
    temperature = float(request.form.get('temperature'))
    humidity = float(request.form.get('humidity'))
    wind_speed = float(request.form.get('wind_speed'))
    precipitation = float(request.form.get('precipitation'))
    cloud_cover = float(request.form.get('cloud_cover'))
    pressure = float(request.form.get('pressure'))

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

    predict_pipeline = PredictPipeline()
    result = predict_pipeline.predict(pred_df)

    prediction = "🌧 Rain Expected" if result[0] == 1 else "☀ No Rain Expected"

    return render_template(
        'home.html',
        results=prediction
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)