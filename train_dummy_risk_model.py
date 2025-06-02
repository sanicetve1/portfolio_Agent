import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Dummy data
X = pd.DataFrame([
    {"sector_Technology": 1, "sector_Natural Resources": 0, "sector_Banking": 0, "sector_F&B": 0, "sector_Healthcare": 0, "sector_Utilities": 0, "sector_Other": 0},
    {"sector_Technology": 0, "sector_Natural Resources": 1, "sector_Banking": 0, "sector_F&B": 0, "sector_Healthcare": 0, "sector_Utilities": 0, "sector_Other": 0},
    {"sector_Technology": 0, "sector_Natural Resources": 0, "sector_Banking": 1, "sector_F&B": 0, "sector_Healthcare": 0, "sector_Utilities": 0, "sector_Other": 0},
])
y = [0, 1, 2]  # Corresponds to: Low, Medium, High

# Train model
clf = RandomForestClassifier()
clf.fit(X, y)

# Save locally with matching environment
output_path = os.path.join("app", "risk_model.pkl")
joblib.dump(clf, output_path)

print(f"✅ Model saved to {output_path}")
