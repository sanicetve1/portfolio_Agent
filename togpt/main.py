from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.utils import get_stock_price
from openai import OpenAI
import os

app = FastAPI()
#openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai = OpenAI(api_key="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")


class StockItem(BaseModel):
    symbol: str
    quantity: int
    buy_price: float


class PortfolioRequest(BaseModel):
    risk_rating: str
    portfolio: List[StockItem]

@app.post("/analyze-portfolio-gpt")
async def analyze_portfolio_gpt(request: PortfolioRequest):
    lines = []
    for item in request.portfolio:
        current_price = get_stock_price(item.symbol)
        lines.append(
            f"- {item.symbol}: {item.quantity} shares bought at ${item.buy_price}, current price ${current_price}"
        )

    prompt = (
        f"The customer has a {request.risk_rating} risk tolerance. "
        f"Here is their current stock portfolio:\n\n"
        + "\n".join(lines)
        + "\n\nProvide buy/sell/hold recommendations for each stock. "
        "Also suggest 2 additional stocks appropriate for this risk profile."
    )

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a financial assistant who gives risk-aware portfolio suggestions."},
            {"role": "user", "content": prompt}
        ]
    )

    return {"summary": response.choices[0].message.content}
