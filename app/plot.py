
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

def get_price_forecast_plot(symbols):
    try:
        end = datetime.today()
        start = end - timedelta(days=180)
        plt.figure(figsize=(10, 6))

        for symbol in symbols:
            stock_data = yf.download(symbol, start=start, end=end)
            if not stock_data.empty:
                plt.plot(stock_data['Close'], label=symbol)

        plt.title("6-Month Price Trend of Suggested Stocks")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close()
        return encoded
    except Exception as e:
        print("Error generating forecast plot:", e)
        return ""
