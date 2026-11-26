"""Lot details: occupancy with the next 3 hours of forecast, price history, and why this price."""
import pandas as pd
import streamlit as st

from _setup import check_ready, db, refresh_seconds

st.set_page_config(page_title="ParkedIn: lot details", layout="wide")
st.title("Lot details")
check_ready(st)
lot = st.selectbox("Parking lot", db.lots()["lot_id"])


@st.fragment(run_every=refresh_seconds())
def lot_view(lot_id):
    hist = db.history(lot_id)
    if hist.empty:
        st.info("No data yet. Press Start on the home page.")
        return
    hist["ts"] = pd.to_datetime(hist["ts"])
    last = hist.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price for next slot", f"Rs {last['price']:.0f} / hour")
    c2.metric("Demand level", last["level"])
    c3.metric("Occupancy now", f"{last['occ_rate']:.0%}")
    c4.metric("Expected at new price", f"{last['expected_occ']:.0%}")

    recent = hist.tail(54)                                   # about the last 3 days
    fc = db.latest_forecast(lot_id)
    fc_times = [last["ts"] + pd.Timedelta(minutes=30 * h) for h in fc["slot_ahead"]]
    chart = pd.concat([
        pd.DataFrame({"Actual occupancy": recent["occ_rate"].values}, index=recent["ts"]),
        pd.DataFrame({"Forecast (next 3 hours)": [last["occ_rate"], *fc["predicted"]]},
                     index=[last["ts"], *fc_times]),
    ])
    st.subheader("Occupancy and forecast")
    st.caption("Forecast slots past 4:30 pm fall overnight on the chart; the data has no night readings.")
    st.line_chart(chart, y_label="Occupancy rate")

    st.subheader("Price")
    st.line_chart(recent.set_index("ts")["price"], y_label="Rs per hour")

    st.subheader("Why this price")
    status = db.latest_status().set_index("lot_id")
    st.write(status["reason"].get(lot_id, "-"))


lot_view(lot)
