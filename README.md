# 🚍 Bus Ticketing Demand Optimization and Forecasting

## 🚀 Project Overview
This project optimizes and forecasts **bus ticket demand** using machine learning models like **ARIMA, SARIMA, and LSTM**.  
The goal is to **improve scheduling, minimize operational costs, and increase revenue**.

---

## 📊 Dataset Description
The dataset contains **14,278 rows and 13 columns**, including:
- `Date` - Trip date
- `Bus Route No.` - Unique route identifier
- `From` & `To` - Source and destination
- `Trips per Day` - Number of trips per day
- `Frequency (mins)` - Interval between buses
- `Tickets Sold` - Number of tickets booked (**Important Feature**)
- `Revenue Generated (INR)` - Total revenue per trip (**Target Variable**)

---

## 📌 Key Steps
### ✅ **1. Data Preprocessing**
- Handled **missing values, outliers, and duplicates**
- Feature engineering for trend analysis
- Normalized and encoded categorical features

### ✅ **2. Model Development**
- **ARIMA** (AutoRegressive Integrated Moving Average)
- **SARIMA** (Seasonal ARIMA)
- **LSTM** (Long Short-Term Memory Neural Networks)

### ✅ **3. Deployment**
- **Flask API (`bus_optimization.py`)** for predictions
- **Streamlit Dashboard (`Bus_deployment.py`)** for visualization

---

## ⚙️ Installation & Setup
```bash
# Clone the repository
git clone https://github.com/venkateshmanuka/bus-ticketing-forecasting.git

# Navigate to the project directory
cd bus-ticketing-forecasting

# Install required dependencies
pip install -r requirements.txt
