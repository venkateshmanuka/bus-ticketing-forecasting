import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load pre-trained models
sarima_model_path = 'sarima_model.pkl'
arima_model_path = 'arima_model.pkl'
lstm_model_path = 'lstm_model.h5'

sarima_model = joblib.load(sarima_model_path)
arima_model = joblib.load(arima_model_path)
lstm_model = load_model(lstm_model_path)

# Streamlit UI setup
st.title("Bus Ticket Revenue Forecasting App")
st.write("Forecast bus ticket revenue using SARIMA, ARIMA, and LSTM models.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read dataset
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.write(df.head())

    # Data preprocessing
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.set_index('Date', inplace=True)

    # Select columns for forecasting
    st.write("Select columns for revenue forecasting:")
    selected_columns = st.multiselect("Columns", df.columns.tolist(), default=['Revenue Generated (INR)'])

    if selected_columns:
        df_selected = df[selected_columns]
        st.write("Selected Data:")
        st.write(df_selected.head())

        # Resample data by day and handle missing values
        df_selected = df_selected.resample("D").sum().fillna(0)

        # Scale data for LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_selected)

        # Model selection
        model_type = st.selectbox("Select Model:", ("SARIMA", "ARIMA", "LSTM"))

        forecast_steps = st.slider("Select number of days to forecast:", 1, 60, 7)

        # Forecasting
        st.write(f"Forecast for the next {forecast_steps} days using {model_type} model:")

        if model_type == "SARIMA":
            # SARIMAX forecasting using forecast() method
            start = len(df_selected)
            forecast_result = sarima_model.forecast(steps=forecast_steps)
            forecast_values = forecast_result

        elif model_type == "ARIMA":
            # ARIMA forecasting using forecast() method
            forecast_result = arima_model.forecast(steps=forecast_steps)
            forecast_values = forecast_result

        elif model_type == "LSTM":
            # LSTM forecasting
            lstm_input = scaled_data[-60:].reshape(1, 60, 1)
            lstm_forecast = []

            for _ in range(forecast_steps):
                prediction = lstm_model.predict(lstm_input)[0, 0]   
                lstm_forecast.append(prediction)
                lstm_input = lstm_input[:, 1:, :]
                lstm_input = np.concatenate([lstm_input, np.array([[[prediction]]])], axis=1)

            # Reverse scaling
            forecast_values = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1)).flatten()

        # Forecasted Dates
        forecast_dates = pd.date_range(start=df_selected.index[-1], periods=forecast_steps + 1, freq="D")[1:]

        # Display forecast
        st.write("Forecasted Revenue:", forecast_values)
        st.write("Forecasted Dates:", forecast_dates)

        # Plot results
        st.write("Forecast Plot:")
        plt.figure(figsize=(12, 6))
        plt.plot(df_selected.index, df_selected[selected_columns[0]], label="Actual Revenue")
        plt.plot(forecast_dates, forecast_values, label=f"{model_type} Forecast", linestyle="--")
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Revenue Generated (INR)")
        plt.title(f"Bus Ticket Revenue Forecast using {model_type}")
        st.pyplot(plt)

else:
    st.write("Please upload a CSV file to proceed.")
