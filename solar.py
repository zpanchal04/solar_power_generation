import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import time # Used to simulate a long-running prediction process

# --- Mock Classes and Data Simulation ---
# Since we cannot load the actual .pkl files, we create mock classes 
# that mimic the behavior of a trained model and a data scaler.

class MockModel:
    """Simulates a trained ML model (e.g., XGBoost Regressor)."""
    def predict(self, data):
        """
        Mock prediction function requires 20 features to match the expected structure.
        """
        # Check if the correct number of features (20) is passed
        EXPECTED_FEATURES = 20
        if data.shape[1] != EXPECTED_FEATURES:
             st.error(f"Mock Model Error: Expected {EXPECTED_FEATURES} features, but received {data.shape[1]}. Check feature DataFrame construction.")
             return np.array([0.0])
             
        # Use key inputs to bias the mock prediction
        irradiance_score = data['shortwave_radiation_backwards_sfc'].iloc[0] / 1200
        temp_score = data['temperature_2_m_above_gnd'].iloc[0] / 45
        
        # Simulated MWh generation, highly dependent on GHI and temperature
        base_power = 1500 + (irradiance_score * 1000) + (temp_score * 500)
        return np.array([base_power + np.random.normal(0, 50)])

class MockScaler:
    """Simulates a fitted data scaler (e.g., StandardScaler)."""
    def transform(self, data):
        # In a real app, this would preprocess the input features.
        return data

@st.cache_resource
def load_resources():
    """
    Loads the actual CSV data for analysis and attempts to load the model.
    """
    MODEL_FILE = 'xgb_pipeline.pkl'
    DATA_FILE = 'spg.csv'
    
    # Initialize variables to Mock objects to ensure they are always bound
    model = MockModel()
    scaler = MockScaler() 

    # --- 1. Load Model ---
    try:
        model = joblib.load(MODEL_FILE)
        scaler = MockScaler() 
        st.success(f"Successfully loaded model from {MODEL_FILE} .")
    except FileNotFoundError:
        st.warning(f"Could not find '{MODEL_FILE}'. Using mock objects instead of real model/scaler.")
    except Exception as e:
        st.error(f"Error loading model: {e}. Using mock objects.")

    # --- 2. Load and Prepare Actual Data (spg.csv) ---
    try:
        data_df = pd.read_csv(DATA_FILE)
        
        # Assume data is hourly and convert target (kW -> MWh) for display consistency (1 MWh = 1000 kWh)
        data_df['Actual Power (MWh)'] = data_df['generated_power_kw'] / 1000 
        
        # Create a sequential date/time index for plotting
        data_df['Date'] = pd.date_range(start='2024-01-01 00:00:00', periods=len(data_df), freq='H')
        
        # Create a mock 'Predicted Power' column for the analysis view since the model is often mocked
        data_df['Predicted Power (MWh)'] = data_df['Actual Power (MWh)'] * np.random.uniform(0.9, 1.05, len(data_df))
        
        # Rename key columns for display purposes
        data_df = data_df.rename(columns={
            'shortwave_radiation_backwards_sfc': 'Irradiance (W/m²)', 
            'temperature_2_m_above_gnd': 'Temperature (°C)'
        })
        
        st.info(f"Successfully loaded {len(data_df)} rows of data from {DATA_FILE} for Historical Analysis.")

    except FileNotFoundError:
        st.error(f"Could not find required data file '{DATA_FILE}'. Using synthesized mock data for analysis.")
        
        # Revert to synthesized mock data if CSV is missing
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data_df = pd.DataFrame({
            'Date': dates,
            'Irradiance (W/m²)': np.random.randint(200, 1200, 100),
            'Temperature (°C)': np.random.uniform(15, 35, 100).round(1),
            'Actual Power (MWh)': (np.sin(np.arange(100) / 15) + 2) * 1000 + np.random.normal(0, 100, 100),
            'Predicted Power (MWh)': (np.sin(np.arange(100) / 15) + 2) * 1000 + np.random.normal(0, 150, 100)
        })
    except Exception as e:
        st.error(f"Error processing data file: {e}. Using synthesized mock data.")
        # Revert to synthesized mock data if processing fails
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        data_df = pd.DataFrame({
            'Date': dates,
            'Irradiance (W/m²)': np.random.randint(200, 1200, 100),
            'Temperature (°C)': np.random.uniform(15, 35, 100).round(1),
            'Actual Power (MWh)': (np.sin(np.arange(100) / 15) + 2) * 1000 + np.random.normal(0, 100, 100),
            'Predicted Power (MWh)': (np.sin(np.arange(100) / 15) + 2) * 1000 + np.random.normal(0, 150, 100)
        })


    # Simulate Model Performance Metrics
    metrics = {
        'Linear Regression': {'RMSE': 350.5, 'R²': 0.85},
        'Gradient Boosting': {'RMSE': 125.2, 'R²': 0.96},
        'XGBoost': {'RMSE': 98.7, 'R²': 0.98}
    }
    
    return model, scaler, data_df, metrics

# --- Utility Functions ---

def run_prediction(features_df, model, scaler):
    """
    Handles the prediction logic.
    
    The features_df MUST contain the 20 features expected by the model/pipeline 
    with their exact required column names and order.
    """
    try:
        # The pipeline handles scaling internally.
        prediction = model.predict(features_df)[0]
        
        return prediction
    except Exception as e:
        # This will catch feature mismatch errors, but if the mock model is running, it should work.
        st.error(f"Prediction failed: {e}")
        return None

# --- Dashboard Layout and Pages ---

# 1. set_page_config MUST be the very first Streamlit command.
st.set_page_config(
    page_title="SunCast: Advanced Solar Prediction Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom Styling (st.markdown) is also a Streamlit command and must follow config.
st.markdown(
    """
    <style>
    .reportview-container .main {
        color: #1f2937;
        background-color: #f3f4f6;
    }
    h1 {
        color: #f59e0b;
        font-weight: 700;
        font-size: 2.5rem;
    }
    .stButton>button {
        background-color: #fbbf24;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #d97706;
        color: white;
        transform: translateY(-2px);
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        color: #ef4444; /* Red color for impact */
    }
    .metric-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #4b5563;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 3. Load all resources (cached) AFTER set_page_config
model, scaler, data_df, performance_metrics = load_resources()


st.title("☀️ SunCast: Solar Power Generation Forecast")
st.subheader("Leveraging Machine Learning for Renewable Energy Optimization")

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Model Overview & Metrics", "🔮 Interactive Prediction Tool"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by OpenAI GPT-4 | Data Science Enthusiasts") 

# --- PAGE 1: Model Overview & Metrics ---
if page == "📊 Model Overview & Metrics":
    st.header("1. Model Performance Snapshot")
    st.markdown("A comparison of different models trained for the solar prediction task.")

    # Convert metrics dictionary to DataFrame for easy display
    metrics_df = pd.DataFrame(performance_metrics).T
    metrics_df = metrics_df.reset_index().rename(columns={'index': 'Model'})
    
    # Highlight the best performing model (lowest RMSE)
    best_model = metrics_df.loc[metrics_df['RMSE'].idxmin()]

    # Layout with columns for key performance indicators (KPIs)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>Best Model</div>
                <div class='metric-value' style='color:#0d9488;'>{best_model['Model']}</div>
            </div>
            """, unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>Best RMSE</div>
                <div class='metric-value' style='color:#ef4444;'>{best_model['RMSE']:.1f}</div>
            </div>
            """, unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>Best R-squared</div>
                <div class='metric-value' style='color:#2563eb;'>{best_model['R²']:.2f}</div>
            </div>
            """, unsafe_allow_html=True
        )
        
    with col4:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>Active Features</div>
                <div class='metric-value' style='color:#9333ea;'>20</div>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("Detailed Model Comparison")
    
    # Plotly Bar Chart for RMSE comparison
    fig_rmse = px.bar(
        metrics_df,
        x='Model',
        y='RMSE',
        color='RMSE',
        color_continuous_scale=px.colors.sequential.Inferno_r,
        title="Model Root Mean Squared Error (RMSE)",
        height=400
    )
    fig_rmse.update_layout(xaxis_title="", yaxis_title="RMSE (MWh)", plot_bgcolor='white')
    st.plotly_chart(fig_rmse, use_container_width=True)
    
    # Display the metrics data table
    st.markdown("#### Performance Data Table")
    st.dataframe(metrics_df.set_index('Model').style.highlight_min(subset=['RMSE'], color='lightgreen').highlight_max(subset=['R²'], color='lightblue'))

# --- PAGE 2: Interactive Prediction Tool ---
elif page == "🔮 Interactive Prediction Tool":
    st.header("2. Real-Time Solar Power Prediction")
    st.markdown("Input the 20 environmental and solar geometry parameters below to get a power generation forecast.")
    
    # --- Input Form ---
    with st.form(key='prediction_form'):
        
        st.subheader("Group 1: Key Radiation and Surface Conditions")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Feature 1: shortwave_radiation_backwards_sfc (Irradiance)
            shortwave_radiation_backwards_sfc = st.slider(
                "GHI [W/m²]", 
                min_value=0, max_value=1200, value=750, step=10, key='f1'
            )
        with col2:
            # Feature 2: temperature_2_m_above_gnd (Air Temperature)
            temperature_2_m_above_gnd = st.slider(
                "Air Temp [°C]", 
                min_value=-10.0, max_value=45.0, value=25.0, step=0.5, key='f2'
            )
        with col3:
            # Feature 3: relative_humidity_2_m_above_gnd (Humidity)
            relative_humidity_2_m_above_gnd = st.number_input(
                "Relative Humidity (%)", 
                min_value=0, max_value=100, value=60, step=1, key='f3'
            )
        with col4:
            # Feature 4: total_precipitation_sfc (New: Precipitation)
            total_precipitation_sfc = st.number_input(
                "Total Precipitation [kg/m²]", 
                min_value=0.0, max_value=20.0, value=0.0, step=0.1, key='f4'
            )
            
        st.markdown("---")
        st.subheader("Group 2: Wind and Surface Pressure")
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            # Feature 5: wind_speed_10_m_above_gnd (Updated name)
            wind_speed_10_m_above_gnd = st.slider(
                "Wind Speed (10m) [m/s]",
                min_value=0.0, max_value=30.0, value=5.0, step=0.1, key='f5'
            )
        with col6:
            # Feature 6: wind_direction_10_m_above_gnd (New)
            wind_direction_10_m_above_gnd = st.slider(
                "Wind Direction (10m) [deg]",
                min_value=0.0, max_value=360.0, value=180.0, step=1.0, key='f6'
            )
        with col7:
            # Feature 7: wind_gust_10_m_above_gnd (New)
            wind_gust_10_m_above_gnd = st.slider(
                "Wind Gust (10m) [m/s]",
                min_value=0.0, max_value=40.0, value=10.0, step=0.5, key='f7'
            )
        with col8:
            # Feature 8: mean_sea_level_pressure_MSL
            mean_sea_level_pressure_MSL = st.number_input(
                "MSL Pressure [hPa]",
                min_value=900.0, max_value=1100.0, value=1013.0, step=1.0, key='f8'
            )

        # --- Advanced Inputs (Group 3 & 4) in an Expander ---
        with st.expander("Advanced Weather & Geometry Inputs (12 features)"):
            
            st.markdown("#### Group 3: Cloud Layers and Snow")
            col9, col10, col11, col12, col13 = st.columns(5)
            
            with col9:
                # Feature 9: total_cloud_cover_sfc (New)
                total_cloud_cover_sfc = st.slider(
                    "Total Cloud Cover (SFC) [%]",
                    min_value=0.0, max_value=100.0, value=30.0, step=1.0, key='f9'
                )
            with col10:
                # Feature 10: high_cloud_cover_high_cld_lay
                high_cloud_cover_high_cld_lay = st.slider(
                    "High Cloud Cover (0.0 to 1.0)",
                    min_value=0.0, max_value=1.0, value=0.2, step=0.05, key='f10'
                )
            with col11:
                # Feature 11: medium_cloud_cover_mid_cld_lay (Updated name)
                medium_cloud_cover_mid_cld_lay = st.slider(
                    "Mid Cloud Cover (0.0 to 1.0)",
                    min_value=0.0, max_value=1.0, value=0.1, step=0.05, key='f11'
                )
            with col12:
                # Feature 12: low_cloud_cover_low_cld_lay
                low_cloud_cover_low_cld_lay = st.slider(
                    "Low Cloud Cover (0.0 to 1.0)",
                    min_value=0.0, max_value=1.0, value=0.05, step=0.05, key='f12'
                )
            with col13:
                # Feature 13: snowfall_amount_sfc
                snowfall_amount_sfc = st.number_input(
                    "Snowfall [kg/m²]",
                    min_value=0.0, max_value=50.0, value=0.0, step=0.1, key='f13'
                )
                
            st.markdown("#### Group 4: Solar Geometry and Upper Atmosphere Wind")
            col14, col15, col16, col17, col18, col19, col20 = st.columns(7)

            with col14:
                # Feature 14: angle_of_incidence
                angle_of_incidence = st.slider(
                    "Angle of Incidence [deg]",
                    min_value=0.0, max_value=90.0, value=30.0, step=0.5, key='f14'
                )
            with col15:
                # Feature 15: zenith (New)
                zenith = st.slider(
                    "Solar Zenith Angle [deg]",
                    min_value=0.0, max_value=90.0, value=30.0, step=0.5, key='f15'
                )
            with col16:
                # Feature 16: azimuth
                azimuth = st.slider(
                    "Solar Azimuth [deg]",
                    min_value=0.0, max_value=360.0, value=180.0, step=1.0, key='f16'
                )
            with col17:
                # Feature 17: wind_speed_80_m_above_gnd (New)
                wind_speed_80_m_above_gnd = st.slider(
                    "Wind Speed (80m) [m/s]",
                    min_value=0.0, max_value=35.0, value=10.0, step=0.1, key='f17'
                )
            with col18:
                # Feature 18: wind_direction_80_m_above_gnd (New)
                wind_direction_80_m_above_gnd = st.slider(
                    "Wind Dir (80m) [deg]",
                    min_value=0.0, max_value=360.0, value=180.0, step=1.0, key='f18'
                )
            with col19:
                # Feature 19: wind_speed_900_mb (New)
                wind_speed_900_mb = st.slider(
                    "Wind Speed (900mb) [m/s]",
                    min_value=0.0, max_value=40.0, value=15.0, step=0.5, key='f19'
                )
            with col20:
                # Feature 20: wind_direction_900_mb (New)
                wind_direction_900_mb = st.slider(
                    "Wind Dir (900mb) [deg]",
                    min_value=0.0, max_value=360.0, value=180.0, step=1.0, key='f20'
                )

        st.markdown("---")
        # Submit button for the form
        submit_button = st.form_submit_button(label='Predict Power Output')
        
    # --- Prediction Output ---
    if submit_button:
        # Assemble ALL 20 features into a DataFrame, using the EXACT names and ensuring all are present.
        # The feature order here is based on the CSV header for optimal compatibility with the loaded pipeline.
        features = pd.DataFrame({
            'temperature_2_m_above_gnd': [temperature_2_m_above_gnd],
            'relative_humidity_2_m_above_gnd': [relative_humidity_2_m_above_gnd],
            'mean_sea_level_pressure_MSL': [mean_sea_level_pressure_MSL],
            'total_precipitation_sfc': [total_precipitation_sfc],
            'snowfall_amount_sfc': [snowfall_amount_sfc],
            'total_cloud_cover_sfc': [total_cloud_cover_sfc],
            'high_cloud_cover_high_cld_lay': [high_cloud_cover_high_cld_lay],
            'medium_cloud_cover_mid_cld_lay': [medium_cloud_cover_mid_cld_lay],
            'low_cloud_cover_low_cld_lay': [low_cloud_cover_low_cld_lay],
            'shortwave_radiation_backwards_sfc': [shortwave_radiation_backwards_sfc],
            'wind_speed_10_m_above_gnd': [wind_speed_10_m_above_gnd],
            'wind_direction_10_m_above_gnd': [wind_direction_10_m_above_gnd],
            'wind_speed_80_m_above_gnd': [wind_speed_80_m_above_gnd],
            'wind_direction_80_m_above_gnd': [wind_direction_80_m_above_gnd],
            'wind_speed_900_mb': [wind_speed_900_mb],
            'wind_direction_900_mb': [wind_direction_900_mb],
            'wind_gust_10_m_above_gnd': [wind_gust_10_m_above_gnd],
            'angle_of_incidence': [angle_of_incidence],
            'zenith': [zenith],
            'azimuth': [azimuth],
        })
        
        # Show a spinner while the "prediction" runs
        with st.spinner('Calculating solar generation...'):
            time.sleep(1.5) # Simulate API/computation time
            predicted_power = run_prediction(features, model, scaler)

        if predicted_power is not None:
            st.balloons()
            st.success("✅ Prediction Complete!")
            
            # Display result in a large, prominent box
            st.markdown(
                f"""
                <div style="background-color: #e0f2f1; padding: 2rem; border-radius: 1rem; text-align: center; margin-top: 1.5rem; border: 2px solid #009688;">
                    <h2 style="color: #009688; margin-bottom: 0.5rem;">Forecasted Power Output</h2>
                    <p class='metric-value' style='font-size: 4rem; color: #009688;'>{predicted_power:.2f}</p>
                    <p style="font-size: 1.5rem; color: #4b5563; font-weight: bold;">Megawatt-hours (MWh)</p>
                </div>
                """, unsafe_allow_html=True
            )
