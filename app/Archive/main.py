
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import joblib
import os
import yfinance as yf
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = "../model/risk_model.pkl"
risk_model = joblib.load(MODEL_PATH)

SECTOR_MAP = {
    "AAPL": "Technology",
    "TSLA": "Technology",
    "MSFT": "Technology",
    "JPM": "Banking",
    "XOM": "Natural Resources",
    "KO": "F&B",
    "JNJ": "Healthcare",
    "NEE": "Utilities"
}

SECTORS = [
    "Technology",
    "Natural Resources",
    "Banking",
    "F&B",
    "Healthcare",
    "Utilities",
    "Other"
]

class StockItem(BaseModel):
    symbol: str
    quantity: int
    buy_price: float

class PortfolioRequest(BaseModel):
    risk_rating: str = "Medium"
    sectors: List[str] = []
    esg_preference: bool = False
    portfolio: List[StockItem]

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_sector_weights(portfolio: List[StockItem]) -> Dict[str, float]:
    sector_value = {sector: 0 for sector in SECTORS}
    total_value = 0
    for item in portfolio:
        try:
            current_price = yf.Ticker(item.symbol).history(period="1d")["Close"].iloc[-1]
            value = item.quantity * current_price
            sector = SECTOR_MAP.get(item.symbol, "Other")
            sector_value[sector] += value
            total_value += value
        except Exception:
            pass
    return {
        f"sector_{s}": round(sector_value[s] / total_value, 2) if total_value > 0 else 0.0
        for s in SECTORS
    }

def get_average_volatility(portfolio: List[StockItem]) -> float:
    vol_sum = 0
    count = 0
    for item in portfolio:
        try:
            hist = yf.Ticker(item.symbol).history(period="6mo")
            daily_returns = hist["Close"].pct_change().dropna()
            if not daily_returns.empty:
                vol_sum += daily_returns.std()
                count += 1
        except Exception:
            pass
    return round(vol_sum / count, 3) if count > 0 else 0

@app.post("/predict-risk")
async def predict_risk(request: PortfolioRequest):
    sector_weights = get_sector_weights(request.portfolio)
    avg_vol = get_average_volatility(request.portfolio)
    input_vector = [avg_vol] + [sector_weights.get(f"sector_{s}", 0.0) for s in SECTORS]
    prediction = risk_model.predict([input_vector])[0]
    return {
        "risk_level": int(prediction),
        "avg_volatility": float(avg_vol),
        "sector_weights": {k: float(v) for k, v in sector_weights.items()}
    }

@app.post("/analyze-portfolio-gpt")
async def analyze_portfolio_gpt(request: PortfolioRequest):
    lines = []
    for item in request.portfolio:
        current_price = yf.Ticker(item.symbol).history(period="1d")["Close"].iloc[-1]
        lines.append(
            f"- {item.symbol}: {item.quantity} shares bought at ${item.buy_price}, current price ${round(current_price, 2)}"
        )

    esg_text = "The user prefers ESG-compliant investments.\n" if request.esg_preference else ""
    sector_list = ", ".join(request.sectors)
    sector_text = f"Preferred sectors: {sector_list}\n" if sector_list else ""

    prompt = (
        f"The customer has a {request.risk_rating} risk tolerance.\n"
        f"{esg_text}"
        f"{sector_text}"
        "Here is their current stock portfolio:\n\n" + "\n".join(lines) +
        "\n\nSuggest buy/sell/hold actions and possible new allocations in JSON format followed by reasoning."
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a financial assistant. Respond with a list of JSON suggestions followed by natural language summary."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    summary = response.choices[0].message.content
    return {"summary": summary}
