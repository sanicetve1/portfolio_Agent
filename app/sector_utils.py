# sector_utils.py
import yfinance as yf

# Optional fallback mapping if API fails or sector missing
fallback_sector_map = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "TSLA": "Automotive",
    "GOOGL": "Technology",
    "JPM": "Banking",
    "XOM": "Natural Resources",
    "AMZN": "Retail",
    "UNH": "Healthcare",
    "NVDA": "Technology",
    "WMT": "Retail",
}

def get_sector(symbol: str) -> str:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        sector = info.get("sector")
        if sector:
            return sector
    except:
        pass
    return fallback_sector_map.get(symbol.upper(), "Other")
