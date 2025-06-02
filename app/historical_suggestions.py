# historical_suggestions.py
import streamlit as st

def render_historical_suggestions(search_fn, portfolio_data):
    try:
        similar = search_fn(portfolio_data)
        seen = set()
        unique_summaries = []
        for m in similar.get("metadatas", [[]])[0]:
            summary = m.get("summary", "").strip()
            if summary and summary not in seen:
                seen.add(summary)
                unique_summaries.append(summary)

        if unique_summaries:
            with st.expander("🧠 Similar Historical Suggestions"):
                for i, summary in enumerate(unique_summaries):
                    st.markdown(f"**#{i + 1}**")
                    st.markdown(summary)

    except Exception as e:
        st.warning(f"⚠️ Could not load historical suggestions: {e}")
