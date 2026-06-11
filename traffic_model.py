import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

print("Loading Traffic dataset...")
df = pd.read_csv('../data/traffic_data.csv') # Assuming data is in the data folder

# Features matching your dashboard inputs
features = ['hour', 'day_of_week', 'is_weekend', 'rainfall', 'temperature', 'road_type']
target = 'congestion_index'
df = df[features + [target]].dropna()

# Encode Road Type (Text to Numbers)
le_road = LabelEncoder()
df['road_type'] = le_road.fit_transform(df['road_type'])

X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Traffic Model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) 
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy (R-squared): {accuracy * 100:.2f}%")

# Save Model and Encoder to the models folder
joblib.dump(model, "../models/traffic_model.pkl")
joblib.dump(le_road, "../models/road_encoder.pkl")
print("✅ traffic_model.pkl and road_encoder.pkl saved!")