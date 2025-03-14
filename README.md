# 🚍 Bus Ticketing Demand Optimization and Forecasting

## 📌 Project Overview

This project focuses on optimizing and forecasting bus ticket demand using machine learning models like **ARIMA, SARIMA, and LSTM**. The goal is to improve scheduling, minimize operational costs, and increase revenue by accurately predicting ticket sales and passenger demand. The model leverages historical bus operations data and advanced time-series forecasting techniques.

---

## 🔍 Business Problem

The **bus service industry** faces several challenges:
- Inefficient resource allocation leading to overbooked or underutilized trips.
- Customer dissatisfaction due to **misaligned schedules** and **unpredictable demand** patterns.
- Revenue loss and increased operational costs.
- Poor customer experience due to frequent cancellations and routes operating with minimal occupancy.

This project aims to address these challenges by developing an accurate **demand forecasting system** to optimize bus schedules and ticket pricing.

---

## 🎯 Business Objectives
- **Maximize prediction accuracy** for ticket sales and passenger demand.
- **Minimize operational costs** by optimizing trip schedules and resource allocation.
- **Enhance customer experience** by reducing trip cancellations and improving availability.

### ⚠️ Constraints
- Ensure **full compliance** with data privacy regulations.
- Maintain **interpretability and usability** of the predictive model for actionable insights.

---

## 📊 Dataset Description

The dataset used for this project contains **14,278 rows and 13 columns**, including:

| Column Name            | Description                                      |
|------------------------|--------------------------------------------------|
| **Date**              | Trip date                                       |
| **Bus Route No.**     | Unique route identifier                         |
| **From & To**         | Source and destination                          |
| **Trips per Day**     | Number of trips per day                         |
| **Frequency (mins)**  | Interval between buses                          |
| **Tickets Sold**      | Number of tickets booked (**Important Feature**) |
| **Revenue Generated** | Total revenue per trip (**Target Variable**)     |

- **Total Rows:** 14,278
- **Total Columns:** 13
- **Dataset Format:** CSV

---

## ⚙️ Project Architecture

This project follows the **CRISP-ML(Q)** methodology:

1. **Business & Data Understanding** - Define problem & collect data.
2. **Data Preparation** - Preprocessing, handling missing values, feature engineering.
3. **Model Building** - ARIMA, SARIMA, LSTM models.
4. **Model Evaluation** - Assess performance using RMSE, MAE, etc.
5. **Model Deployment** - Flask API, Streamlit Dashboard.
6. **Monitoring & Maintenance** - Track model performance over time.

---

## 🔬 Key Steps

### ✅ 1. Data Preprocessing
- Handled **missing values**, **outliers**, and **duplicates**.
- Applied **feature engineering** for trend analysis.
- **Normalized** and **encoded** categorical features.

### ✅ 2. Model Development
- **ARIMA (AutoRegressive Integrated Moving Average)** - Suitable for short-term forecasting.
- **SARIMA (Seasonal ARIMA)** - Handles seasonality in ticket sales.
- **LSTM (Long Short-Term Memory)** - Deep learning model for sequential data prediction.

### ✅ 3. Model Evaluation
- **Mean Absolute Percentage Error (MAPE)** used for evaluation.
- **LSTM outperformed ARIMA & SARIMA** in accuracy and error minimization.

### ✅ 4. Deployment
- **Flask API (`bus_optimization.py`)** for real-time forecasting.
- **Streamlit Dashboard (`Bus_deployment.py`)** for interactive visualization.

---

## 💻 Technical Stack

### **Programming Language**
- Python

### **Libraries & Frameworks**
- **Data Manipulation**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Statistical Analysis & Forecasting**: Statsmodels (for ARIMA, SARIMA)
- **Machine Learning**: TensorFlow, Keras (for LSTM)
- **Data Preprocessing**: Scikit-learn
- **Outlier Handling**: Feature-engine (Winsorization)
- **Deployment**: Flask, Streamlit

### **Hardware Requirements**
- **Processor**: Intel Core i5 or higher / AMD Ryzen 5
- **RAM**: 16 GB
- **Storage**: 256 GB SSD or higher
- **GPU (Optional for LSTM)**: NVIDIA GTX 1050 or higher

### **Software Requirements**
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python Version**: 3.9 or higher
- **IDE/Editor**: Spyder, Jupyter Notebook, VS Code

---

## 📊 Model Comparison & Accuracy

| Model  | MAE (Mean Absolute Error) | MSE (Mean Squared Error) | RMSE (Root Mean Squared Error) |
|--------|---------------------------|--------------------------|--------------------------------|
| **LSTM**  | **6.86E-06 (Best)** | **9.12E-10 (Lowest)** | **3.02E-05 (Best)** |
| **SARIMA** | 0.031 | 0.0054 | 0.0733 |
| **ARIMA**  | 0.0313 | 0.0054 | 0.0733 |

### 🔥 **Best Model: LSTM**
- **Lower errors across all metrics.**
- **Captures time-series patterns more effectively.**

---

## 🚀 Deployment Strategy

- **Local Deployment:** Runs using `streamlit run Bus_deployment.py`
- **Cloud Deployment (Future Scope):** Can be hosted on AWS, GCP, or Azure.
- **Interactive Dashboard:** Users upload data, select a model, and view predictions.
- **Forecasting Range:** Predicts ticket demand for **1 to 60 days**.

---

## 🔮 Future Scope

🔹 **Improve Accuracy** – Fine-tune models for better predictions.
🔹 **Real-Time Data** – Connect to live ticketing systems.
🔹 **Cloud Deployment** – Host on AWS, Azure, or Google Cloud.
🔹 **More Factors** – Include weather, holidays, and fuel prices.
🔹 **Better Dashboard** – Use Power BI or Tableau for insights.
🔹 **Auto Model Selection** – Dynamically choose the best model.
🔹 **Mobile Access** – Develop a mobile-friendly version.

---

## 📌 How to Run the Project

### 1️⃣ Clone the Repository
```bash
 git clone https://github.com/yourusername/bus-ticketing-forecasting.git
 cd bus-ticketing-forecasting
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit Dashboard
```bash
streamlit run Bus_deployment.py
```

---

## 🛠 Contributors
- **Venkatesh Manuka** 

🔗 **Connect on LinkedIn**: [Venkatesh Manuka](https://www.linkedin.com/in/venkatesh-manuka/)

---

## 📩 Queries?
Feel free to **raise an issue** or **contact me** via GitHub! 🚀

