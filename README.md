# 🏙️ CityPulse AI - Real-Time Urban Alert System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![API](https://img.shields.io/badge/API-WhatsApp%20Integration-brightgreen)

## 📌 Project Overview
**CityPulse AI** is an intelligent, live-working prototype designed to enhance urban safety and awareness. By leveraging Machine Learning and real-time data processing, this backend system monitors critical city metrics and triggers automated emergency alerts directly to citizens via the WhatsApp API. 

## 🚀 Core Features
* **Multi-Domain Predictive Modeling:** Dedicated machine learning models to analyze and predict urban conditions:
  * 🚦 **Traffic Forecasting** (`traffic_model.py`)
  * 🚨 **Crime Hotspot Detection** (`crime_model.py`)
  * ☁️ **Air Quality Index (AQI) Monitoring** (`aqi_model.py`)
  * 🌪️ **Disaster Prediction** (`train_disaster.py`)
* **Real-Time Automated Alerts:** Categorizes incidents based on severity and dispatches instant WhatsApp notifications to ensure critical information reaches users without delay.
* **Robust Backend:** Centralized routing and logic handling using a Python/Flask architecture (`app.py`).

## 🛠️ Tech Stack
* **Language:** Python
* **Backend Framework:** Flask 
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Communication:** WhatsApp API Integration

## 📂 Project Structure
```text
📦 CityPulse-AI-Alert-System
 ┣ 📜 app.py                # Main application logic and API routing
 ┣ 📜 aqi_model.py          # Script for Air Quality Index predictions
 ┣ 📜 crime_model.py        # Script for crime data analysis and alerts
 ┣ 📜 traffic_model.py      # Script for traffic congestion forecasting
 ┣ 📜 train_disaster.py     # Script for disaster prediction modeling
 ┗ 📜 .gitignore            # Git ignore configurations (excluding large .pkl files)
