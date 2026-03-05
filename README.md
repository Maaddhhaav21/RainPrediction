# 🌧 Extreme Rainfall Prediction using Machine Learning (2025)

An **end-to-end Machine Learning system** that predicts **extreme rainfall events** using atmospheric parameters such as **temperature, humidity, pressure, wind speed, precipitation, and cloud cover**.

The project includes **data preprocessing, model training, model evaluation, and a production-ready web application** for real-time predictions.

The system can help support **meteorological analysis, early warning systems, and climate monitoring applications.**

---

# 📌 Project Overview

Extreme rainfall events can cause:

- Flooding
- Infrastructure damage
- Transportation disruption
- Agricultural loss

This project uses **Machine Learning classification models** to predict whether **rainfall will occur tomorrow** based on weather parameters.

The system performs:

- Data ingestion
- Data preprocessing
- Feature transformation
- Model training
- Model selection
- Model deployment

---

# 🚀 Features

- End-to-end ML pipeline
- Automated data preprocessing
- Feature engineering and scaling
- Multiple model comparison
- Hyperparameter tuning
- Model persistence
- Web interface for predictions
- Production-ready architecture
- Cloud deployment support

---

# 🧠 Machine Learning Models Used

The system evaluates multiple classification algorithms:

- Random Forest
- Gradient Boosting
- XGBoost
- CatBoost
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- AdaBoost

The **best performing model is automatically selected** based on classification accuracy.

---

# 📊 Dataset Features

The model predicts rainfall using the following atmospheric parameters:

| Feature       | Description                      |
| ------------- | -------------------------------- |
| Location      | City or region                   |
| Temperature   | Temperature in Fahrenheit (°F)   |
| Humidity      | Atmospheric humidity (%)         |
| Wind Speed    | Wind speed (km/h)                |
| Precipitation | Current precipitation level (mm) |
| Cloud Cover   | Cloud cover percentage (%)       |
| Pressure      | Atmospheric pressure (hPa)       |

Target Variable:

```
Rain Tomorrow
0 → No Rain
1 → Rain Expected
```

---

# 🏗 Project Architecture

```
RainPrediction/
│
├── artifacts/                # Saved models and preprocessing objects
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── train.csv
│   └── test.csv
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transform.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── templates/
│   └── home.html
│
├── app.py                    # Flask web application
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/extreme-rainfall-prediction.git
cd extreme-rainfall-prediction
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

Mac/Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🏃 Running the Application

Start the Flask server:

```bash
python app.py
```

Open your browser:

```
http://127.0.0.1:5000
```

Enter weather parameters and get rainfall predictions instantly.

---

# 🌐 Deployment (Render)

This application can be deployed on **Render** for public access.

### Step 1 — Push Project to GitHub

```
git add .
git commit -m "Initial commit"
git push origin main
```

### Step 2 — Create Render Web Service

1. Go to **https://render.com**
2. Click **New Web Service**
3. Connect your GitHub repository
4. Configure:

```
Environment: Python
Build Command: pip install -r requirements.txt
Start Command: python app.py
```

### Step 3 — Deploy

Render will automatically:

- Install dependencies
- Start the Flask server
- Provide a public URL

Example:

```
https://extreme-rainfall-prediction.onrender.com
```

---

# 📈 Example Prediction

Input:

| Feature       | Value   |
| ------------- | ------- |
| Location      | Mumbai  |
| Temperature   | 60°F    |
| Humidity      | 98%     |
| Wind Speed    | 28 km/h |
| Precipitation | 15 mm   |
| Cloud Cover   | 95%     |
| Pressure      | 980 hPa |

Prediction:

```
🌧 Rain Expected
```

---

# 🧪 Tech Stack

- Python
- Scikit-learn
- XGBoost
- CatBoost
- Pandas
- NumPy
- Flask
- HTML/CSS
- Render (Deployment)

---

# 📌 Future Improvements

- Real-time weather API integration
- Rainfall probability prediction
- Deep learning models
- Interactive dashboard
- Geographic rainfall visualization

---

# 👨‍💻 Author

**Madhav Manoj**
