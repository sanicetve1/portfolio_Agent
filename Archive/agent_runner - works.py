# agent_runner.py - Streamlit-based AI Portfolio Agent (Upgraded)

import streamlit as st
import json
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from app.plot import get_price_forecast_plot
from app.utils import get_stock_price
from app.pdf_parser import extract_text_from_pdf, parse_portfolio_from_text
import requests
import re
import pandas as pd

st.set_page_config(page_title="AI Portfolio Agent", layout="wide")
st.title("🤖 Autonomous Portfolio Agent")

# --- Input Section ---
st.markdown("### Upload Portfolio (PDF or JSON)")
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

portfolio_data = []
if pdf_file:
    text = extract_text_from_pdf(pdf_file)
    portfolio_data = parse_portfolio_from_text(text)
    st.success("✅ Portfolio extracted from PDF")
    st.json(portfolio_data)
else:
    example = '''[
      {"symbol": "AAPL", "quantity": 10, "buy_price": 150},
      {"symbol": "TSLA", "quantity": 5, "buy_price": 600}
    ]'''
    user_input = st.text_area("Paste Portfolio JSON", example, height=150)
    if user_input:
        try:
            portfolio_data = json.loads(user_input)
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

# --- Preferences ---
risk_rating = st.selectbox("Risk Rating", ["Low", "Medium", "High"])
esg_preference = st.checkbox("Prefer ESG-compliant?")
sectors = st.multiselect("Preferred Sectors", [
    "Technology", "Natural Resources", "Banking", "F&B", "Healthcare", "Utilities"
])

# --- Tool Definitions ---
def predict_risk_tool(portfolio):
    response = requests.post("http://localhost:8000/predict-risk", json={"portfolio": portfolio})
    return response.json() if response.ok else {"error": "Risk prediction failed."}

def gpt_analysis_tool(prompt_str):
    try:
        portfolio_match = re.search(r"Portfolio:\s+(.*?)\s+Risk Rating:", prompt_str, re.DOTALL)
        rating_match = re.search(r"Risk Rating:\s+(.*?)\s+ESG Preference:", prompt_str)
        esg_match = re.search(r"ESG Preference:\s+(.*?)\s+Preferred Sectors:", prompt_str)
        sectors_match = re.search(r"Preferred Sectors:\s+(.*)", prompt_str)

        portfolio = json.loads(portfolio_match.group(1).strip()) if portfolio_match else []
        rating = rating_match.group(1).strip() if rating_match else "Medium"
        esg = esg_match.group(1).strip().lower() == "yes" if esg_match else False
        sectors = [s.strip() for s in sectors_match.group(1).split(',')] if sectors_match else []

        payload = {
            "portfolio": portfolio,
            "risk_rating": rating,
            "esg_preference": esg,
            "sectors": sectors
        }

        response = requests.post("http://localhost:8000/analyze-portfolio-gpt", json=payload)
        return response.json() if response.ok else {"error": "GPT analysis failed."}
    except Exception as e:
        return {"error": f"Parsing error: {e}"}


def plot_tool(symbols):
    return get_price_forecast_plot(symbols)

# --- Agent Execution ---
def run_agent():
    if not portfolio_data:
        st.warning("Please provide portfolio data.")
        return

    # Define tools with input/output expected
    tools = [
        Tool(name="PredictRisk", func=lambda x: predict_risk_tool(x), description="Predicts portfolio risk from given data"),
        Tool(name="GPTAnalysis", func=gpt_analysis_tool, description="Analyzes portfolio and gives GPT investment advice"),
        Tool(name="PlotPrices", func=lambda x: plot_tool(x), description="Plots future trends for given stock symbols")
    ]

    llm = ChatOpenAI(temperature=0.3)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )

    # Call the agent with structured context
    context = {
        "portfolio": portfolio_data,
        "rating": risk_rating,
        "esg": esg_preference,
        "sectors": sectors
    }

    prompt = f"""
    The user has provided the following portfolio data and preferences:

    Portfolio:
    {json.dumps(portfolio_data, indent=2)}

    Risk Rating: {risk_rating}
    ESG Preference: {'Yes' if esg_preference else 'No'}
    Preferred Sectors: {', '.join(sectors) if sectors else 'None'}

    Please analyze the risk, suggest any changes, and plot trends.
    """

    with st.expander("🧪 Agent Execution Log"):
        st.code(prompt, language="markdown")

    result = agent.run(prompt)
    st.success("✅ Agent completed its analysis.")
    st.markdown("### 🧾 Agent Output")
    st.write(result)

    # 🔍 Optional: Extract and display base64 chart if mentioned
    if "AAPL" in result or "TSLA" in result:
        try:
            chart = get_price_forecast_plot(["AAPL", "TSLA"])
            st.image(f"data:image/png;base64,{chart}", caption="6-Month Forecast")
        except Exception as e:
            st.warning(f"⚠️ Could not render chart: {e}")

# --- Button ---
if st.button("🧠 Run AI Agent"):
    run_agent()
