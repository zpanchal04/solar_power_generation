# ☀️ SunCast — Solar Power Generation Forecasting Dashboard

SunCast is an interactive **Streamlit-powered dashboard** that predicts **solar power generation (MWh)** using machine-learning models and real-time environmental parameters.  
It provides a complete workflow including:

✅ Historical data analysis  
✅ Model performance comparison  
✅ Real-time prediction using 20+ weather & solar geometry features  

This project is ideal for data science, renewable energy engineering, and machine-learning deployment.

---

## 🚀 Features

### ✅ 1. Machine Learning Model Integration
- Supports XGBoost, Gradient Boosting, Linear Regression, Ridge Regression  
- Loads saved ML pipelines (`.pkl` files)  
- Falls back to a mock model when real pipelines are unavailable  

### ✅ 2. Real-Time Solar Prediction Tool
Accepts 20+ critical environmental parameters:
- Irradiance (GHI)  
- Temperature  
- Humidity  
- Cloud layers (low/mid/high)  
- Wind speeds (10m, 80m, 900mb)  
- Wind gusts  
- Atmospheric pressure  
- Snowfall & precipitation  
- Solar zenith, azimuth, and incidence angle  

### ✅ 3. Model Performance Dashboard
- RMSE & R² comparison  
- KPI cards for best model insight  
- Interactive Plotly bar charts  
- Actual vs. predicted historical visualization  

### ✅ 4. Clean & Modern Interface
- Streamlit UI enhanced with custom CSS  
- Sidebar navigation  
- Balloon animation on prediction  
- Mobile-friendly layout  

---

## 🧠 Tech Stack

| Category | Tools |
|---------|-------|
| Language | Python |
| Dashboard | Streamlit |
| ML Models | XGBoost, GradientBoosting, Linear Regression, Ridge |
| Visualization | Plotly |
| Data Handling | Pandas, NumPy |
| Model Loading | Joblib |

---

## 📂 Project Structure
📦 solar_power_generation
├── solar.py # Main Streamlit dashboard
├── Solar.ipynb # Notebook for data prep & modeling
├── spg.csv # Historical solar dataset
├── gbr_pipeline.pkl # Gradient Boosting model
├── rf_pipeline.pkl # Random Forest model
├── xgb_pipeline.pkl # XGBoost model
├── linear_regression_pipeline.pkl # Linear regression model
├── ridge_pipeline.pkl # Ridge regression model
├── solar power generation analytics.pbix # Power BI dashboard
├── README.md # Documentation
└── .ipynb_checkpoints/ # Notebook temp files





