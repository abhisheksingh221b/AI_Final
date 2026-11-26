"""Gemini-powered parking recommendations and business analysis."""
import streamlit as st

from _setup import check_ready, db
from gemini_assistant import (advise_revenue_strategy, chat_about_data, discover_insights,
                              explain_price, recommend_parking, weekly_report)

st.set_page_config(page_title="ParkedIn: AI assistant", page_icon="✨", layout="wide")
st.title("Parking AI assistant")
st.caption("Gemini and OpenAI turn current and historical parking data into concise guidance. Outputs are advisory.")
check_ready(st)


def api_keys():
    """Read optional Streamlit secrets; environment variables are handled by the service."""
    try:
        return st.secrets.get("GEMINI_API_KEY"), st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None, None


def show_result(call):
    try:
        result = call()
        for key, value in result.items():
            label = key.replace("_", " ").title()
            if isinstance(value, list):
                st.markdown(f"**{label}**")
                if value:
                    for item in value:
                        st.markdown(f"- {item}")
                else:
                    st.write("None identified.")
            else:
                st.markdown(f"**{label}:** {value}")
    except Exception as exc:
        st.error(str(exc))


recommend_tab, explain_tab, report_tab, chat_tab, insights_tab, strategy_tab = st.tabs([
    "Find parking", "Explain a price", "Weekly report", "Ask about data", "Discover insights", "Revenue strategy"
])

with recommend_tab:
    st.subheader("Parking recommendation")
    st.caption("Enter your location to compare current distance, occupancy, and suggested hourly price.")
    lots = db.lots()
    center_lat = float(lots.latitude.median()) if not lots.empty else 0.0
    center_lon = float(lots.longitude.median()) if not lots.empty else 0.0
    with st.form("recommend_form"):
        c1, c2 = st.columns(2)
        latitude = c1.number_input("Your latitude", value=center_lat, format="%.6f")
        longitude = c2.number_input("Your longitude", value=center_lon, format="%.6f")
        submit = st.form_submit_button("Recommend a lot", type="primary")
    if submit:
        show_result(lambda: recommend_parking(latitude, longitude, *api_keys()))

with explain_tab:
    st.subheader("Dynamic pricing explanation")
    current = db.latest_status()
    if current.empty:
        st.info("Start the simulation to see current prices.")
    else:
        lot_id = st.selectbox("Parking lot", current.lot_id.tolist(), key="explain_lot")
        if st.button("Explain suggested price"):
            show_result(lambda: explain_price(lot_id, *api_keys()))

with report_tab:
    st.subheader("Weekly business report")
    st.caption("Summarizes the most recent seven days of available replay data.")
    if st.button("Generate weekly report", type="primary"):
        show_result(lambda: weekly_report(*api_keys()))

with chat_tab:
    st.subheader("Ask about parking data")
    st.caption("Answers are grounded in up to 30 days of recorded simulation readings and prices.")
    with st.form("chat_form"):
        question = st.text_input("Question", placeholder="Which lot had the highest average occupancy?")
        ask = st.form_submit_button("Ask Gemini", type="primary")
    if ask:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            show_result(lambda: chat_about_data(question.strip(), *api_keys()))

with insights_tab:
    st.subheader("Insight discovery")
    st.caption("Looks for hourly patterns and unusual lot-day occupancy using an IQR rule.")
    if st.button("Analyze recent history", type="primary"):
        show_result(lambda: discover_insights(*api_keys()))

with strategy_tab:
    st.subheader("Revenue strategy advisor")
    st.caption("Suggestions are hypotheses to test; the pricing elasticity is assumed by the simulator.")
    if st.button("Suggest business actions", type="primary"):
        show_result(lambda: advise_revenue_strategy(*api_keys()))
