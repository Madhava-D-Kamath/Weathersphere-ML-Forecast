# 🌎 WeatherSphere AI — Predictive Meteorological Intelligence

<div align="center">

### A state-of-the-art Machine Learning weather forecasting platform featuring localized time-series sequence model training and live interactive telemetry dashboards.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📖 Table of Contents

- [Project Description](#-project-description)
- [System Architecture](#️-system-architecture)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation Guide](#-installation-guide)
- [How to Run](#-how-to-run)
- [Live API References](#-live-api-references)
- [Testing & Validation](#-testing--validation)
- [Roadmap & Contributing](#-roadmap--contributing)
- [License](#-license)

---

## 📝 Project Description

**WeatherSphere AI** is an advanced meteorological analytics dashboard that demonstrates the power of localized, sequence-aware Machine Learning models in predicting temporal atmospheric shifts. While standard physical weather models require massive supercomputing grids, WeatherSphere AI consumes 5 years of local climate history from the Open-Meteo API on the fly, applies robust time-series feature engineering, trains baseline (Random Forest) and deep sequence-based (PyTorch LSTM, GRU, Bi-LSTM) networks, and overlays live visual telemetry.

The platform provides a high-fidelity, dark-themed dashboard built with Streamlit, enabling meteorologists, developers, and users to explore local climates, run predictive model comparisons, and inspect live atmospheric alerts.

---

## 🗺️ System Architecture

The core data flow of WeatherSphere AI integrates data ingestion, preprocessing, modeling, and real-time visualization:

```
[Open-Meteo REST API] ──(Ingest)──> [data/raw_weather.csv]
                                            │
                                    (Feature Eng.)
                                            │
                                            v
[Streamlit GUI app.py] <──(Run)── [data/processed_weather.csv]
          │                                 │
     (Model Zoo)                       (80/20 Split)
          │                                 │
          v                                 v
[models/lstm_model.pth] <──(Train)──> [models/random_forest_model.joblib]
```

---

## 🌟 Key Features

### 🔮 1. Custom Deep Learning Model Studio
* Interactively configure hidden layers, learning rates, batch sizes, and training epochs directly from the dashboard.
* Watch live PyTorch Mean Squared Error (MSE) loss curves plot in real time as the neural network learns.

### 🗺️ 2. Interactive Map Visualizations
* **Live Sidebar Weather Radar**: A clean, visual-only live map widget showing local radar reflections centered on coordinates.
* **Interactive Main Map**: Under the **Maps** tab, a 550px interactive maps frame equipped with a horizontal selector to toggle between:
  * 📡 **Weather Radar** (`radar`)
  * 🛰️ **Satellite Map** (`satellite`)
  * 💨 **Wind Velocity** (`wind`)
  * 🌡️ **Temperature Layer** (`temp`)
  * 🌧️ **Rain / Accumulation** (`rain`)

### 🕒 3. Actual 24-Hour & 10-Day Extended Forecasts
* Replaces simulated curves with dynamic, real-time hourly temperature paths and precipitation bars using double-axis Plotly charts.
* Highlights a 10-day forecast with daily high/low envelopes translated automatically from World Meteorological Organization (WMO) codes.

### ⚠️ 4. Dynamic Biometeorological Alert Panel
* Analyzes current temperature limits, barometric pressure offsets, and wind velocities.
* Automatically triggers severe health warning banners (Heat Warnings, Cold/Vasoconstriction Risks, Joint Pain Pressure drops) detailing medical guidelines.

### 📄 5. Automated PDF Report Builder
* Bundles all model scores, comparisons, pipeline upgrades, and impact reviews into a clean, print-ready document.

---

## 💻 Tech Stack

| Technology | Layer | Role / Purpose |
|---|---|---|
| **Python 3.14 / 3.9+** | Core Runtime | Base computational language |
| **Streamlit 1.35.0+** | Front-End & UI | Glassmorphism dashboard frame |
| **PyTorch 2.0.0+** | Deep Learning | Sequence LSTM, GRU, and Bi-LSTM models |
| **Scikit-Learn 1.2.0+** | Tabular ML | Tabular Random Forest regressor baseline |
| **Plotly 5.14.0+** | Data Viz | Double-axis curves and hover tooltip charts |
| **Pandas / NumPy** | Data Prep | Time-series aggregations, lags, and rolling averages |
| **ReportLab / FPDF2** | Document compiler | Restored project report PDF compiler |

---

## 📂 Project Structure

```text
weathersphere-ai/
├── data/
│   ├── raw_weather.csv           # Ingested 5-year daily weather dataset
│   └── processed_weather.csv     # Transformed dataset with lags, rolling averages, and season encodings
├── docs/
│   └── architecture_design.md    # System Architecture & Design Specification
├── models/
│   ├── random_forest_model.joblib # Saved Random Forest baseline model
│   ├── lstm_model.pth            # Saved PyTorch LSTM model weights
│   ├── lstm_scaler.joblib        # Fitted features and target MinMaxScaler state objects
│   ├── best_model.txt            # Text file storing the name of the best performing model
│   ├── active_location.txt       # Active city/location name context
│   └── active_coordinates.txt    # Coordinates of current location context
├── outputs/
│   ├── metrics_comparison.csv    # Evaluated test metrics for both models
│   ├── actual_vs_predicted.png   # Overlaid test set forecast visualization
│   ├── rf_feature_importance.png # Top feature relative importances from Random Forest
│   └── latest_forecast.json      # Current weather status & next-day model prediction metadata
├── src/
│   ├── data_ingestion.py         # API fetching module for 5-year archives
│   ├── feature_engineering.py    # Lag, rolling statistics, and seasonal encodings
│   ├── train_random_forest.py    # Random Forest training & evaluation
│   ├── train_lstm.py             # PyTorch LSTM, GRU, and Bi-LSTM model definition & training
│   ├── evaluate.py               # Combined model validation and graph output generator
│   └── pipeline.py               # Orchestrator script for the training workflow
├── app.py                        # Premium glassmorphism Streamlit dashboard
├── generate_report_pdf.py        # PDF technical report builder script
├── requirements.txt              # Project package dependencies
└── README.md                     # Documentation (this file)
```

---

## ⚙️ Installation Guide

### 1. Set Up the Virtual Environment
Activate the workspace-wide virtual environment in your terminal:

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows CMD:**
```cmd
.\.venv\Scripts\activate.bat
```

### 2. Install Project Dependencies
Install the required packages in your active environment:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1. Run the End-to-End ML Pipeline
To fetch weather historical datasets, execute feature engineering, train models, compare metrics, and generate evaluation plots:
```bash
python src/pipeline.py
```

### 2. Launch the Interactive Dashboard
Boot up the Streamlit server:
```bash
streamlit run app.py
```

### 3. Compile the PDF Project Report
Compile and export the project report document directly to the root workspace:
```bash
python generate_report_pdf.py
```

---

## 🔌 Live API References

The platform communicates with the following external endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `https://geocoding-api.open-meteo.com/v1/search` | `GET` | Translates city searches into coordinates |
| `https://api.open-meteo.com/v1/forecast` | `GET` | Fetches live hourly/daily weather forecasts |
| `https://archive-api.open-meteo.com/v1/archive` | `GET` | Fetches 5-year daily climate datasets |

---

## 🧪 Testing & Validation

* **Syntax Verification**: Validate code syntax before staging:
  ```bash
  python -m py_compile app.py
  ```
* **Pipeline Integration Validation**: Run the evaluation suite:
  ```bash
  python src/evaluate.py
  ```

---

## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for details.