import streamlit as st
import requests
import json
import re
import pandas as pd
from app.plot import get_price_forecast_plot
from app.pdf_parser import extract_text_from_pdf, parse_portfolio_from_text

st.title("Portfolio Risk & Recommendation with GPT")

# --- PDF Upload Section ---
st.markdown("### 📄 Upload Portfolio PDF (Optional)")
pdf_file = st.file_uploader("Upload a portfolio PDF", type=["pdf"])

portfolio_data = []

if pdf_file is not None:
    extracted_text = extract_text_from_pdf(pdf_file)
    portfolio_data = parse_portfolio_from_text(extracted_text)
    st.success("✅ Portfolio extracted from PDF.")
    st.json(portfolio_data)
else:
    st.markdown("### Or Enter Your Portfolio Manually (JSON)")
    example_json = '''[
      {"symbol": "AAPL", "quantity": 10, "buy_price": 150},
      {"symbol": "TSLA", "quantity": 5, "buy_price": 600}
    ]'''
    portfolio_input = st.text_area("Portfolio JSON", example_json, height=150)
    if portfolio_input:
        try:
            portfolio_data = json.loads(portfolio_input)
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

# --- Preferences Section ---
st.markdown("### Portfolio Preferences")
col1, col2 = st.columns(2)

with col1:
    risk_rating = st.selectbox("Risk Rating", ["Low", "Medium", "High"])
    esg_preference = st.checkbox("Prefer ESG-compliant?")

with col2:
    sectors = st.multiselect(
        "Preferred Sectors",
        ["Technology", "Natural Resources", "Banking", "F&B", "Healthcare", "Utilities"]
    )

# --- Analyze Button ---
if st.button("Analyze Portfolio") and portfolio_data:
    try:
        # Risk Prediction
        risk_response = requests.post(
            "http://localhost:8000/predict-risk",
            json={"portfolio": portfolio_data},
            timeout=120
        )

        if risk_response.status_code == 200:
            risk_result = risk_response.json()
            st.subheader("🔎 Portfolio Risk Classification")
            st.markdown(f"**Risk Level:** `{risk_result['risk_level']}`")
            st.markdown(f"**Avg Volatility:** `{risk_result['avg_volatility']}`")
            st.markdown("**Sector Weights:**")
            st.json(risk_result["sector_weights"])
        else:
            st.warning("⚠️ Risk prediction failed.")

        # GPT Analysis
        response = requests.post(
            "http://localhost:8000/analyze-portfolio-gpt",
            json={
                "risk_rating": risk_rating,
                "sectors": sectors,
                "esg_preference": esg_preference,
                "portfolio": portfolio_data
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            full_output = result["summary"]

            json_text = ""
            suggested_symbols = []

            try:
                json_match = re.search(r"{[\\s\\S]*?}\\s*(?=\\n|$)", full_output)
                if json_match:
                    json_text = json_match.group(0)
                    suggestions = json.loads(json_text)

                    st.json(suggestions)

                    if "currentPortfolio" not in suggestions:
                        current_keys = [k for k, v in suggestions.items() if isinstance(v, dict) and "action" in v]
                        suggestions["currentPortfolio"] = {k: suggestions[k] for k in current_keys}

                    if "newAllocations" not in suggestions and "new_allocations" in suggestions:
                        suggestions["newAllocations"] = suggestions["new_allocations"]

                    current_df = pd.DataFrame.from_dict(suggestions["currentPortfolio"], orient="index")
                    current_df.index.name = "Symbol"
                    st.subheader("📋 Current Portfolio Analysis")
                    st.dataframe(current_df)

                    new_allocs = suggestions.get("newAllocations") or suggestions.get("new_allocations") or []
                    new_df = pd.DataFrame(new_allocs)

                    st.subheader("📊 New Allocation Suggestions")
                    st.dataframe(new_df)

                    # Extract symbols if possible
                    if "ticker" in new_df.columns:
                        suggested_symbols = new_df["ticker"].dropna().tolist()
                    elif "stock" in new_df.columns:
                        suggested_symbols = new_df["stock"].dropna().tolist()
                    elif "symbol" in new_df.columns:
                        suggested_symbols = new_df["symbol"].dropna().tolist()
                    else:
                        suggested_symbols = []

                else:
                    st.warning("⚠️ No JSON suggestions found.")
            except Exception as e:
                st.warning(f"⚠️ Failed to parse GPT suggestions: {e}")
                suggested_symbols = []

            summary_text = full_output.replace(json_text, "").strip()
            if summary_text:
                st.subheader("🧠 GPT Summary & Recommendation")
                st.markdown(summary_text)

            if suggested_symbols:
                chart = get_price_forecast_plot(suggested_symbols)
                st.subheader("📈 Suggested Stocks — 6-Month Trend")
                st.image(f"data:image/png;base64,{chart}")
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"❌ Error analyzing portfolio: {e}")