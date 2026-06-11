import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

print("Loading AQI dataset...")
df = pd.read_csv('../data/aqi_data.csv')

# Features matching your dashboard inputs
# Note: dashboard sends pm2.5, pm10, no2, so2, co, humidity, wind_speed, season
df.rename(columns={'pm2.5': 'pm2_5'}, inplace=True) # Ensure clean column names
features = ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'humidity', 'wind_speed', 'season']
target = 'final_aqi'
df = df[features + [target]].dropna()

# Encode Season (Text to Numbers)
le_season = LabelEncoder()
df['season'] = le_season.fit_transform(df['season'])

X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training AQI Model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Save Model and Encoder to the models folder
joblib.dump(model, "../models/aqi_model.pkl")
joblib.dump(le_season, "../models/season_encoder.pkl")
print("✅ aqi_model.pkl and season_encoder.pkl saved!")