import os
from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('home.html')

    else:
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
        print("Input Features:")
        print(pred_df)

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)

        print("Raw model prediction:", result)
        print("Prediction type:", type(result[0]))

        prediction = "🌧 Rain Expected" if int(result[0]) == 1 else "☀ No Rain Expected"
        return render_template(
            'home.html',
            results=prediction
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
