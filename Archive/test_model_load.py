import joblib

# Load the model
model = joblib.load("app/risk_model.pkl")

# Try predicting with dummy data
input_vector = [[1, 0, 0, 0, 0, 0, 0]]  # Only Technology sector = 1
prediction = model.predict(input_vector)

print(f"✅ Model prediction output: {prediction}")
