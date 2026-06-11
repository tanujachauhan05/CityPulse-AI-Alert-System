from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app) 

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/data", methods=["GET"])
def data_snapshot():
    # Lightweight endpoint for simple dashboards.
    return jsonify({"aqi": "—", "traffic": "—", "crime": "—", "alert": "—"})

# --- LOAD ALL 4 MODELS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

print("Starting CityPulse AI Server...")
try:
    print("Loading traffic_model.pkl...")
    traffic_model = joblib.load(os.path.join(MODELS_DIR, 'traffic_model.pkl'))
    print("Loading road_encoder.pkl...")
    road_encoder = joblib.load(os.path.join(MODELS_DIR, 'road_encoder.pkl'))
    
    print("Loading disaster_model.pkl...")
    disaster_model = joblib.load(os.path.join(MODELS_DIR, 'disaster_model.pkl'))
    
    print("Loading aqi_model.pkl...")
    aqi_model = joblib.load(os.path.join(MODELS_DIR, 'aqi_model.pkl'))
    print("Loading season_encoder.pkl...")
    season_encoder = joblib.load(os.path.join(MODELS_DIR, 'season_encoder.pkl'))
    
    print("Loading crime_model.pkl...")
    crime_model = joblib.load(os.path.join(MODELS_DIR, 'crime_model.pkl'))
    print("Loading area_encoder.pkl...")
    area_encoder = joblib.load(os.path.join(MODELS_DIR, 'area_encoder.pkl'))
    print("✅ All 4 Models Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise SystemExit(1)

# --- 1. TRAFFIC API ---
@app.route('/predict/traffic', methods=['POST'])
def predict_traffic():
    data = request.json
    try: road_encoded = road_encoder.transform([data['road_type']])[0]
    except: road_encoded = 0 
        
    input_df = pd.DataFrame([{
        'hour': int(data['hour']), 'day_of_week': int(data['day_of_week']),
        'is_weekend': int(data['is_weekend']), 'rainfall': float(data['rainfall']),
        'temperature': float(data['temperature']), 'road_type': road_encoded
    }])
    score = min(max(int(traffic_model.predict(input_df)[0] * 100), 0), 100) 
    level = "SEVERE" if score > 80 else "HIGH" if score > 60 else "MODERATE" if score > 40 else "LOW"
    return jsonify({'score': score, 'level': level})

# --- 2. DISASTER API ---
@app.route('/predict/disaster', methods=['POST'])
def predict_disaster():
    data = request.json
    drainage_map = {'Overloaded (>100%)': 1.0, 'Critical (80-100%)': 2.0, 'High (60-80%)': 3.0, 'Normal (<60%)': 4.0}
    elev_map = {'Low-lying (flood prone)': 2.0, 'Medium elevation': 15.0, 'High elevation': 50.0}
    
    input_df = pd.DataFrame([{
        'rainfall': float(data['rainfall']), 'river_level': float(data['river_level']),
        'wind_speed': float(data['wind_speed']), 
        'drainage_capacity': drainage_map.get(data['drainage_capacity'], 3.0),
        'elevation': elev_map.get(data['elevation'], 15.0)
    }])
    prediction_label = disaster_model.predict(input_df)[0]
    score = 95 if prediction_label == 'Emergency' else 25
    return jsonify({'label': str(prediction_label).upper(), 'score': score})

# --- 3. AQI API ---
@app.route('/predict/aqi', methods=['POST'])
def predict_aqi():
    data = request.json
    try: season_encoded = season_encoder.transform([data['season']])[0]
    except: season_encoded = 0
    
    input_df = pd.DataFrame([{
        'pm2_5': float(data['pm2_5']), 'pm10': float(data['pm10']),
        'no2': float(data['no2']), 'so2': float(data['so2']),
        'co': float(data['co']), 'humidity': float(data['humidity']),
        'wind_speed': float(data['wind_speed']), 'season': season_encoded
    }])
    
    aqi_val = int(aqi_model.predict(input_df)[0])
    score = min(int((aqi_val / 500) * 100), 100) # Convert AQI to 0-100% for progress bar
    level = "HAZARDOUS" if aqi_val > 300 else "POOR" if aqi_val > 150 else "MODERATE"
    return jsonify({'score': score, 'level': level, 'raw_aqi': aqi_val})

# --- 4. CRIME API ---
@app.route('/predict/crime', methods=['POST'])
def predict_crime():
    data = request.json
    try: area_encoded = area_encoder.transform([data['area']])[0]
    except: area_encoded = 0
        
    input_df = pd.DataFrame([{
        'area': area_encoded, 'hour': int(data['hour']),
        'day': int(data['day']), 'lighting_index': float(data['lighting_index'])
    }])
    
    score = min(max(int(crime_model.predict(input_df)[0]), 0), 100)
    level = "HIGH RISK" if score > 70 else "MEDIUM RISK" if score > 40 else "LOW RISK"
    return jsonify({'score': score, 'level': level})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", "5001"))
    app.run(debug=True, port=port)