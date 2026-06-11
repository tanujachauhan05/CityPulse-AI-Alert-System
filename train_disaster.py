import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("Loading Disaster dataset...")
df = pd.read_csv('../data/disaster_data.csv')

# Features matching your dashboard inputs
features = ['rainfall', 'river_level', 'wind_speed', 'drainage_capacity', 'elevation']
X = df[features].dropna()
y = df.loc[X.index, 'disaster_risk_label'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Disaster Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Save Model to the models folder
joblib.dump(model, "../models/disaster_model.pkl")
print("✅ disaster_model.pkl saved!")