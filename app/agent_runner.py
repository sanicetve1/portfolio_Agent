# === Import Dependencies ===
import streamlit as st
import json
import uuid
import matplotlib.pyplot as plt
import io
import base64
from collections import Counter
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from plot import get_price_forecast_plot
from utils import get_stock_price
from pdf_parser import extract_text_from_pdf, parse_portfolio_from_text
from chroma_store import save_to_vector_db
from historical_suggestions import render_historical_suggestions
from chroma_store import search_similar_portfolios


import requests
import re
import pandas as pd

# === Streamlit App Configuration ===
st.set_page_config(page_title="AI Portfolio Agent", layout="wide")
st.title("🤖 Autonomous Portfolio Agent")

# === Portfolio Input & Preferences Section ===
with st.container():
    input_col1, input_col2 = st.columns(2)

    # Upload PDF or paste JSON
    with input_col1:
        st.markdown("### Upload Portfolio (PDF or JSON)")
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

    portfolio_data = []
    if pdf_file:
        text = extract_text_from_pdf(pdf_file)
        portfolio_data = parse_portfolio_from_text(text)
        st.success("✅ Portfolio extracted from PDF")
        st.json(portfolio_data)
    else:
        with input_col2:
            example = """[
  {\"symbol\": \"AAPL\", \"quantity\": 10, \"buy_price\": 150},
  {\"symbol\": \"TSLA\", \"quantity\": 5, \"buy_price\": 600}
]"""
            user_input = st.text_area("Paste Portfolio JSON", example, height=150)
            if user_input:
                try:
                    portfolio_data = json.loads(user_input)
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")

    # Risk preferences and sector selections
    with input_col1:
        gpt_risk_source = st.radio("Use which risk rating for GPT suggestions?", ["User-selected", "ML-predicted"], index=1)
        risk_rating = st.selectbox("Risk Rating", ["", "Low", "Medium", "High"],
                                   format_func=lambda x: x if x else "Select Risk Rating", key="risk_rating")
        esg_preference = st.checkbox("Prefer ESG-compliant?", key="esg")
    with input_col2:
        sectors = st.multiselect("Preferred Sectors",
                                 ["Technology", "Natural Resources", "Banking", "F&B", "Healthcare", "Utilities"],
                                 key="sectors")

# === Tool Definitions ===
# Tool 1: Risk Prediction via ML model

def predict_risk_tool(input_data):
    try:
        if isinstance(input_data, dict) and "portfolio" in input_data:
            portfolio = input_data["portfolio"]
        elif isinstance(input_data, list):
            portfolio = input_data
        elif isinstance(input_data, str):
            st.warning("⚠️ Input was a string. Using current user portfolio instead.")
            portfolio = portfolio_data
        else:
            raise ValueError("Invalid input format")

        response = requests.post("https://portfolio-agent.onrender.com/predict-risk", json={"portfolio": portfolio})
        return response.json() if response.ok else {"error": "Risk prediction failed."}
    except Exception as e:
        st.error(f"❌ Risk prediction exception: {e}")
        return {"error": str(e)}

# Tool 2: GPT Portfolio Analysis

def gpt_analysis_tool(input_data):
    try:
        payload = {
            "portfolio": input_data.get("portfolio", []),
            "risk_rating": input_data.get("risk_rating", "Medium"),
            "esg_preference": input_data.get("esg_preference", False),
            "sectors": input_data.get("sectors", [])
        }

        response = requests.post("https://portfolio-agent.onrender.com/analyze-portfolio-gpt", json=payload)
        return response.json() if response.ok else {"error": "GPT analysis failed."}
    except Exception as e:
        return {"error": f"Parsing error: {e}"}

# Tool 3: Generate plot for stock symbols

def plot_tool(symbols):
    validated = []
    for sym in symbols:
        try:
            price = get_stock_price(sym)
            if price:
                validated.append(sym)
        except:
            continue
    return get_price_forecast_plot(validated)

# Wrapper to log tool inputs/outputs

def log_tool_call(tool_name, input_data, func):
    st.session_state.tool_trace.append({"tool": tool_name, "input": input_data})
    output = func(input_data)
    st.session_state.tool_trace[-1]["output"] = output
    return output

# Pie chart generator for GPT-suggested stock distribution

def generate_pie_chart(summary_text):
    tickers = re.findall(r'\b[A-Z]{2,5}\b', summary_text)
    known_exclusions = {"ESG", "ETF", "USA", "YES", "BUY", "SELL", "HOLD", "RISK"}
    tickers = [t for t in tickers if t not in known_exclusions]
    if not tickers:
        return None

    counts = Counter(tickers)
    labels = list(counts.keys())
    sizes = list(counts.values())

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return encoded

# === Initial ML Risk Prediction (before agent) ===
ml_risk_result = predict_risk_tool(portfolio_data)
predicted_risk = ml_risk_result.get("risk_rating", risk_rating if risk_rating else "Medium")

if ml_risk_result:
    with st.expander("📉 Risk Rating Comparison"):
        col1, col2 = st.columns(2)
        col1.metric("User Risk Rating", risk_rating if risk_rating else "Not selected")
        risk_color = {"Low": "🟢 Low", "Medium": "🟡 Medium", "High": "🔴 High"}
        col1.markdown(f"**Risk Color:** {risk_color.get(risk_rating, '⚪ Unknown')}")
        col2.metric("ML-Predicted Risk", predicted_risk)
        pred_color = {"Low": "🟢 Low", "Medium": "🟡 Medium", "High": "🔴 High"}
        col2.markdown(f"**Prediction Color:** {pred_color.get(predicted_risk, '⚪ Unknown')}")

# === Agent Execution Trigger ===
def run_agent():
    if not portfolio_data:
        st.warning("Please provide portfolio data.")
        return

    MAX_PORTFOLIO_ITEMS = 10
    portfolio_trimmed = portfolio_data[:MAX_PORTFOLIO_ITEMS]
    portfolio_str = json.dumps(portfolio_trimmed, indent=2)

    if len(portfolio_data) > MAX_PORTFOLIO_ITEMS:
        st.warning(f"🔹 Portfolio too large. Using first {MAX_PORTFOLIO_ITEMS} items for analysis.")

    tools = [
        Tool(name="PredictRisk", func=lambda x: log_tool_call("PredictRisk", x, predict_risk_tool),
             description="Predicts portfolio risk"),
        Tool(name="GPTAnalysis", func=lambda x: log_tool_call("GPTAnalysis", x, gpt_analysis_tool),
             description="GPT-based investment advice"),
        Tool(name="PlotPrices", func=lambda x: log_tool_call("PlotPrices", x, plot_tool),
             description="Price trend plotter")
    ]

    llm = ChatOpenAI(temperature=0.3)
    agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

    input_data = {
        "portfolio": portfolio_trimmed,
        "risk_rating": predicted_risk if gpt_risk_source == 'ML-predicted' else risk_rating,
        "esg_preference": esg_preference,
        "sectors": sectors
    }

    st.session_state.tool_trace = []
    for tool in tools:
        if tool.name == "PredictRisk":
            predict_result = tool.func(input_data)
            break
    else:
        predict_result = {"error": "PredictRisk tool not found"}

    st.session_state.tool_trace.append({"tool": "PredictRisk", "input": input_data, "output": predict_result})

    for tool in tools:
        if tool.name == "GPTAnalysis":
            gpt_result = tool.func(input_data)
            break
    else:
        gpt_result = {"error": "GPTAnalysis tool not found"}

    st.session_state.tool_trace.append({"tool": "GPTAnalysis", "input": input_data, "output": gpt_result})
    #render_historical_suggestions(search_similar_portfolios, portfolio_data)

    # === Results Display ===
    st.success("✅ Agent completed its analysis.")
    st.markdown("### 📟 Agent Output")

    col1, col2 = st.columns(2)

    # Display ML Prediction output
    with col1:
        st.subheader("📈 ML Risk Prediction")
        ml_output = next((t for t in st.session_state.tool_trace if t["tool"] == "PredictRisk"), None)
        if ml_output and isinstance(ml_output["output"], dict):
            ml_data = ml_output["output"]
            st.markdown(f"**Risk Rating:** {ml_data.get('risk_rating', 'Unknown')}")
            st.markdown(f"**Risk Level:** {ml_data.get('risk_level', 'N/A')}")
            st.markdown(f"**Average Volatility:** {ml_data.get('avg_volatility', 'N/A')}")

            sector_weights = ml_data.get("sector_weights", {})
            if sector_weights:
                st.markdown("**Sector Weights:**")
                for sector, weight in sector_weights.items():
                    st.markdown(f"- {sector}: {weight}")
        else:
            st.info("ℹ️ ML model was not called.")

    # Display GPT analysis
    with col2:
        st.subheader("🧠 GPT Suggestions")
        st.markdown("**🔍 Live GPT Summary**")
        gpt_output = next((t for t in st.session_state.tool_trace if t["tool"] == "GPTAnalysis"), None)
        if gpt_output and isinstance(gpt_output["output"], dict) and "summary" in gpt_output["output"]:
            summary_text = gpt_output["output"]["summary"]
            st.markdown(summary_text)
            save_to_vector_db(portfolio_data, summary_text)
            pie = generate_pie_chart(summary_text)
            if pie:
                st.image(f"data:image/png;base64,{pie}", caption="📊 Suggested Stock Allocation")

            # Show chart for symbols in current GPT output
            suggested_symbols = list(set(re.findall(r'\b[A-Z]{1,5}\b', summary_text)))
            known_exclusions = {"ETF", "ESG", "USA", "BUY", "SELL", "HOLD", "RISK"}
            symbols_to_plot = [s for s in suggested_symbols if s not in known_exclusions]

            if symbols_to_plot:
                try:
                    chart = get_price_forecast_plot(symbols_to_plot)
                    st.image(f"data:image/png;base64,{chart}", caption="6-Month Forecast")
                except Exception as e:
                    st.warning(f"⚠️ Could not render chart: {e}")
        else:
            st.info("ℹ️ GPT was not called or returned no summary.")

    # Show tool trace for debugging or transparency
    with st.expander("🛠️ Tool Trace"), st.container():
        for trace in st.session_state.tool_trace:
            st.markdown(f"**{trace['tool']}**")
            st.code(f"Input: {trace['input']}")
            st.code(f"Output: {trace['output']}")

# === Run Agent Button ===
if st.button("🧠 Run AI Agent"):
    run_agent()
