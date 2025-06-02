import streamlit as st
import requests
import json

st.title("Portfolio Risk & Recommendation with GPT")

st.markdown("Enter your portfolio as JSON (symbol, quantity, buy_price):")

example_json = """
[
  {"symbol": "AAPL", "quantity": 10, "buy_price": 150},
  {"symbol": "TSLA", "quantity": 5, "buy_price": 600}
]
"""

portfolio_input = st.text_area("Portfolio JSON", example_json, height=150)

if st.button("Analyze Portfolio"):
    try:
        portfolio_data = json.loads(portfolio_input)

        response = requests.post(
            "http://localhost:8000/analyze-portfolio-gpt",
            json=portfolio_data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            st.subheader("GPT Summary & Recommendation")
            st.markdown(result["summary"])
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"Error: {e}")


from app.plot import get_price_forecast_plot

# Simulate suggested symbols or extract them from GPT response
suggested_symbols = ["AAPL", "MSFT"]  # Replace with actual GPT suggestions if needed

chart = get_price_forecast_plot(suggested_symbols)
st.subheader("📈 Suggested Stock Trends (Past Year)")
st.image(f"data:image/png;base64,{chart}")
