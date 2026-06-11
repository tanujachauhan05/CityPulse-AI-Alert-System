import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

print("Loading Crime dataset...")
df = pd.read_csv('../data/crime_data.csv')

# Create a logical 0-100 Risk Score for the AI to learn
# Late at night + low lighting + high past crime = High Risk Score
df['risk_score'] = (
    (df['past_crime_frequency'] / df['past_crime_frequency'].max() * 40) + 
    ((10 - df['lighting_index']) * 3) + 
    (np.where((df['hour'] >= 22) | (df['hour'] <= 4), 30, 0))
)
df['risk_score'] = df['risk_score'].clip(0, 100) # Keep between 0 and 100

# Features matching your dashboard inputs
features = ['area', 'hour', 'day', 'lighting_index']
target = 'risk_score'
df = df[features + [target]].dropna()

# Encode Area (Text to Numbers)
le_area = LabelEncoder()
df['area'] = le_area.fit_transform(df['area'])

X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Crime Model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Save Model and Encoder to the models folder
joblib.dump(model, "../models/crime_model.pkl")
joblib.dump(le_area, "../models/area_encoder.pkl")
print("✅ crime_model.pkl and area_encoder.pkl saved!")