from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = FastAPI()

# --- Data Models ---
class Stock(BaseModel):
    symbol: str
    quantity: int
    buy_price: float

class PortfolioRequest(BaseModel):
    portfolio: list[Stock]

class GPTRequest(BaseModel):
    portfolio: list[Stock]
    risk_rating: str
    esg_preference: bool
    sectors: list[str]

# --- Load ML Model ---
model_path = os.path.join(os.path.dirname(__file__), "risk_model.pkl")
model = joblib.load(model_path)

# --- ML Risk Prediction Endpoint ---
@app.post("/predict-risk")
def predict_risk(data: PortfolioRequest):
    print("📥 Incoming request payload:", data)
    try:
        df = pd.DataFrame([s.dict() for s in data.portfolio])
        print("📊 Converted DataFrame:")
        print(df)

        if df.empty:
            raise ValueError("Empty portfolio")

        # Dummy sector encoding
        sectors = ["Technology", "Natural Resources", "Banking", "F&B", "Healthcare", "Utilities", "Other"]
        sector_weights = {f"sector_{s}": 0 for s in sectors}
        sector_weights["sector_Technology"] = 1  # Placeholder logic

        expected_columns = [f"sector_{s}" for s in sectors]
        row = [sector_weights.get(col, 0) for col in expected_columns]
        final_df = pd.DataFrame([row], columns=expected_columns)

        prediction = model.predict(final_df)[0]
        rating_map = {0: "Low", 1: "Medium", 2: "High"}
        risk_rating = rating_map.get(int(prediction), "Unknown")

        return {
            "risk_rating": risk_rating,
            "risk_level": int(prediction),
            "avg_volatility": round((df['buy_price'].std() or 0) / df['buy_price'].mean(), 3) if df['buy_price'].mean() else 0.0,
            "sector_weights": final_df.iloc[0].to_dict()
        }

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# --- GPT Analysis Endpoint ---
@app.post("/analyze-portfolio-gpt")
def analyze_portfolio_gpt(data: GPTRequest):
    try:
        prompt = f"""
You are a portfolio analyst. The user has the following portfolio and preferences:

Portfolio:
{json.dumps([s.dict() for s in data.portfolio], indent=2)}

Risk Rating: {data.risk_rating}
ESG Preference: {"Yes" if data.esg_preference else "No"}
Preferred Sectors: {", ".join(data.sectors) if data.sectors else "None"}

Analyze the portfolio and recommend any adjustments to align with the user's goals.
Only suggest real stock symbols. No placeholders.
"""

        print("🧠 GPT Prompt:\n", prompt)

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )

        print("🧠 GPT Raw Response:\n", response)
        result = response.choices[0].message.content
        return {"summary": result}

    except Exception as e:
        print("❌ GPT analysis error:", e)
        raise HTTPException(status_code=500, detail=str(e))
