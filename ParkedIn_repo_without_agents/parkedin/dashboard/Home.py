"""ParkedIn dashboard, home page: live map and prices.

Run:  streamlit run dashboard/Home.py   (with  python simulation.py  running in another terminal)
"""
import streamlit as st

from _setup import COLOURS, check_ready, db, refresh_seconds

st.set_page_config(page_title="ParkedIn", page_icon="P", layout="wide")
st.title("ParkedIn: live parking prices")
st.caption("Every few seconds the system replays one 30-minute slot of the test days: it forecasts "
           "demand, labels it, prices each car park and checks its own forecasts.")
check_ready(st)

c1, c2, _ = st.columns([1, 1, 4])
if c1.button("Start simulation", type="primary", use_container_width=True):
    db.set_control("running", 1)
if c2.button("Stop simulation", use_container_width=True):
    db.set_control("running", 0)


@st.fragment(run_every=refresh_seconds())
def live_view():
    df = db.latest_status()
    state = "**Running.** " if db.get_control("running") == "1" else "**Paused.** "
    if df.empty:
        st.write(state + "Press Start to begin.")
        return
    st.write(state + f"Simulated time: {df['ts'].iloc[0]}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Average price", f"Rs {df['price'].mean():.0f} / hour")
    k2.metric("Average occupancy now", f"{df['occ_rate'].mean():.0%}")
    k3.metric("Lots at peak demand", int((df["level"] == "peak").sum()))
    k4.metric("Special day", "Yes" if df["is_special_day"].max() == 1 else "No")

    df["colour"] = df["level"].map(COLOURS)
    st.map(df, latitude="latitude", longitude="longitude", color="colour", size=40, zoom=14)
    st.caption("Dot colour = demand level: green low, olive medium, orange high, red peak.")

    def bar(label):
        return st.column_config.ProgressColumn(label, format="percent", min_value=0, max_value=1)

    st.dataframe(
        df[["lot_id", "price", "level", "occ_rate", "expected_occ", "queue_length"]],
        hide_index=True, use_container_width=True, height=(len(df) + 1) * 35 + 3,
        column_config={"lot_id": "Lot", "price": "Price (Rs/h)", "level": "Demand level",
                       "occ_rate": bar("Occupancy now"),
                       "expected_occ": bar("Expected at new price"), "queue_length": "Queue"})


live_view()
