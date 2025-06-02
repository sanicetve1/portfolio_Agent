import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

def get_price_forecast_plot(symbols):
    end = datetime.today()
    start = end - timedelta(days=365)

    plt.figure(figsize=(10, 6))

    for symbol in symbols:
        data = yf.download(symbol, start=start, end=end)
        if not data.empty:
            data['Close'].plot(label=symbol)

    plt.title("Past 12-Month Stock Price Trends")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    # Convert to base64
    return base64.b64encode(buf.read()).decode('utf-8')
