
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Paths
CSV_PATH = "../data/portfolio_data.csv"
MODEL_PATH = "../model/risk_model.pkl"

# Load dataset
df = pd.read_csv(CSV_PATH)

# Encode risk levels to numeric
le = LabelEncoder()
df['risk_label'] = le.fit_transform(df['risk_level'])

# Features and target
X = df.drop(columns=['risk_level', 'risk_label', 'customer'])


y = df['risk_label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(clf, MODEL_PATH)

print("✅ Model retrained and saved to:", MODEL_PATH)
