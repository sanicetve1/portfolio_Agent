import yfinance as yf

def get_stock_price(symbol: str) -> float:
    try:
        ticker = yf.Ticker(symbol)
        price = ticker.info.get("regularMarketPrice")

        if price is None:
            price = ticker.fast_info.get("lastPrice")

        if price is None:
            price = ticker.info.get("previousClose")

        return round(price, 2) if price else 0

    except Exception as e:
        print(f"[ERROR] Could not fetch price for {symbol}: {e}")
        return 0
