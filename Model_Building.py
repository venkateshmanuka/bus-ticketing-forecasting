import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError

# Load dataset
df = pd.read_csv(r"C:\Users\DELL\Downloads\Data_Set\Data Set (5)\Operational_Bus_data - Operational_Bus_data.csv")

# Preprocessing steps
numeric_features = df.select_dtypes(exclude=['object']).columns
categorical_features = ['Bus Route No.', 'From', 'To', 'Way', 'Main Station']

# Numeric pipeline
num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', MinMaxScaler())
])

# Categorical pipeline
categ_pipeline = Pipeline([
    ('encoding', OneHotEncoder(sparse_output=False))
])

# Full preprocessing pipeline
preprocess_pipeline = ColumnTransformer([
    ('numerical', num_pipeline, numeric_features),
    ('categorical', categ_pipeline, categorical_features)
])

# Apply transformation and check the column names
preprocessed = preprocess_pipeline.fit(df)
df1 = pd.DataFrame(preprocessed.transform(df), columns=preprocessed.get_feature_names_out())

# Save the preprocessing pipeline
joblib.dump(preprocess_pipeline, 'preprocessed_pipeline.pkl')

# Now let's check column names to determine the target column reference
print("Transformed DataFrame Columns:", df1.columns)

# Define the correct target column name after transformation
# For example, if 'Revenue Generated (INR)' is numeric, it will remain the same after transformation
target_col = 'numerical__Revenue Generated (INR)'  # Corrected based on preprocessing

# Define features and target
X = df1.drop(columns=[target_col])  # Drop the target column from features
Y = df1[target_col]  # Use the correct transformed target column

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# ARIMA Model
p, d, q = 1, 1, 1
arima_model = ARIMA(Y_train, order=(p, d, q))
arima_fitted = arima_model.fit()
print("ARIMA Model Summary:")
print(arima_fitted.summary())

# Forecast using ARIMA
forecast_arima = arima_fitted.forecast(steps=len(Y_test))

# SARIMA Model
sarima_model = SARIMAX(Y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fitted = sarima_model.fit()
print("SARIMA Model Summary:")
print(sarima_fitted.summary())

# Forecast using SARIMA
forecast_sarima = sarima_fitted.forecast(steps=len(Y_test))

# LSTM Model
def create_lstm_model():
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(1, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError())
    return model

# Reshape data for LSTM
Y_train_lstm = np.array(Y_train).reshape(-1, 1)
Y_test_lstm = np.array(Y_test).reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler()
Y_train_scaled = scaler.fit_transform(Y_train_lstm)
Y_test_scaled = scaler.transform(Y_test_lstm)

# Reshape input for LSTM (samples, time steps, features)
X_train_lstm = Y_train_scaled.reshape((Y_train_scaled.shape[0], 1, 1))

# Train LSTM
lstm_model = create_lstm_model()
lstm_model.fit(X_train_lstm, Y_train_scaled, epochs=50, verbose=1, batch_size=32)

# Forecast using LSTM
X_test_lstm = Y_test_scaled.reshape((Y_test_scaled.shape[0], 1, 1))
lstm_predictions_scaled = lstm_model.predict(X_test_lstm)
lstm_forecast = scaler.inverse_transform(lstm_predictions_scaled)

# Model Evaluation
mae_arima = mean_absolute_error(Y_test, forecast_arima)
mse_arima = mean_squared_error(Y_test, forecast_arima)
rmse_arima = np.sqrt(mse_arima)

mae_sarima = mean_absolute_error(Y_test, forecast_sarima)
mse_sarima = mean_squared_error(Y_test, forecast_sarima)
rmse_sarima = np.sqrt(mse_sarima)

mae_lstm = mean_absolute_error(Y_test, lstm_forecast)
mse_lstm = mean_squared_error(Y_test, lstm_forecast)
rmse_lstm = np.sqrt(mse_lstm)

# Print evaluation metrics
print("ARIMA Evaluation Metrics:")
print(f"MAE: {mae_arima}, MSE: {mse_arima}, RMSE: {rmse_arima}")

print("SARIMA Evaluation Metrics:")
print(f"MAE: {mae_sarima}, MSE: {mse_sarima}, RMSE: {rmse_sarima}")

print("LSTM Evaluation Metrics:")
print(f"MAE: {mae_lstm}, MSE: {mse_lstm}, RMSE: {rmse_lstm}")

# Visualization
plt.figure(figsize=(15, 6))
plt.plot(Y_test.values, label='Actual Revenue')
plt.plot(forecast_arima, label='ARIMA Forecast')
plt.plot(forecast_sarima, label='SARIMA Forecast')
plt.plot(lstm_forecast, label='LSTM Forecast')
plt.legend()
plt.title('Revenue Forecasting')
plt.show()

# Save models
joblib.dump(arima_fitted, 'arima_model.pkl')
joblib.dump(sarima_fitted, 'sarima_model.pkl')
lstm_model.save('lstm_model.h5')  # Save as .h5 format for Keras models

# Save the entire preprocessing pipeline
joblib.dump(preprocess_pipeline, 'preprocessed_pipeline.pkl')
print("Preprocessing pipeline saved successfully.")
