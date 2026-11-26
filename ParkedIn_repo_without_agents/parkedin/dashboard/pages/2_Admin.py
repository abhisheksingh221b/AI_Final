"""Admin: results vs a fixed price, alerts, live settings, price limits and model results."""
import json

import pandas as pd
import streamlit as st

from _setup import ROOT, check_ready, db, load_config, refresh_seconds

st.set_page_config(page_title="ParkedIn: admin", layout="wide")
st.title("Admin")
check_ready(st)
cfg = load_config()


@st.fragment(run_every=refresh_seconds())
def live_results():
    st.subheader(f"ParkedIn vs a fixed Rs {cfg['fixed_price']} price (simulated)")
    t = db.metric_totals().set_index("name")
    if "revenue_dynamic" not in t.index:
        st.info("No results yet. Press Start on the home page.")
        return
    rd, rf = t.loc["revenue_dynamic", "total"], t.loc["revenue_fixed", "total"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", f"Rs {rd:,.0f}", f"{100 * (rd / rf - 1):+.1f}% vs fixed")
    c2.metric("Average occupancy", f"{t.loc['occupancy_dynamic', 'mean']:.0%}",
              f"fixed price: {t.loc['occupancy_fixed', 'mean']:.0%}", delta_color="off")
    c3.metric("Lot-slots 95%+ full", f"{t.loc['full_lots_dynamic', 'total']:.0f}",
              f"fixed price: {t.loc['full_lots_fixed', 'total']:.0f}", delta_color="off")
    rolling = db.metric_series("rolling_mae")
    c4.metric("Forecast error (rolling MAE)",
              f"{rolling['value'].iloc[-1]:.3f}" if len(rolling) else "warming up")

    revenue = pd.DataFrame({
        "ParkedIn": db.metric_series("revenue_dynamic").set_index("ts")["value"].cumsum(),
        "Fixed price": db.metric_series("revenue_fixed").set_index("ts")["value"].cumsum()})
    revenue.index = pd.to_datetime(revenue.index)
    st.line_chart(revenue, y_label="Cumulative revenue (Rs)")

    st.subheader("Alerts")
    alerts = db.recent_alerts()
    if alerts.empty:
        st.write("No alerts.")
    else:
        st.dataframe(alerts, hide_index=True, use_container_width=True)


live_results()

st.divider()
st.subheader("Simulation controls")
c1, c2, c3 = st.columns(3)
speed = c1.slider("Seconds per simulated slot", 1, 10, int(float(db.get_control("tick_seconds", 5))))
over_w = c2.slider("Penalty for over-full lots", 0.5, 5.0,
                   float(db.get_control("over_weight", cfg["over_weight"])), 0.5)
under_w = c3.slider("Penalty for empty spaces (lower = higher prices)", 0.5, 5.0,
                    float(db.get_control("under_weight", cfg["under_weight"])), 0.5)
b1, b2 = st.columns([1, 5])
if b1.button("Apply settings"):
    db.set_control("tick_seconds", speed)
    db.set_control("over_weight", over_w)
    db.set_control("under_weight", under_w)
    st.success("Saved. The simulation uses them from the next slot.")
if b2.button("Reset simulation (start the replay again)"):
    db.reset_simulation()
    st.success("Reset. Press Start on the home page.")

st.subheader("Price limits per lot")
edited = st.data_editor(db.lots()[["lot_id", "capacity", "min_price", "max_price"]],
                        hide_index=True, disabled=["lot_id", "capacity"], use_container_width=True)
if st.button("Save price limits"):
    step = cfg["price_step"]
    bad = ((edited["min_price"] >= edited["max_price"])
           | (edited["min_price"] % step != 0) | (edited["max_price"] % step != 0))
    if bad.any():
        st.error(f"Each min must be below its max, and both must be multiples of Rs {step}.")
    else:
        for r in edited.itertuples():
            db.set_limits(r.lot_id, r.min_price, r.max_price)
        st.success("Saved. Prices use the new limits from the next slot.")

st.divider()
st.subheader("Offline model results (from train.py and evaluate_pricing.py)")
metrics_file = ROOT / "models" / "metrics.json"
if metrics_file.exists():
    m = json.loads(metrics_file.read_text())
    st.write("Forecast error on the test days (MAE of occupancy rate, lower is better)")
    mae = pd.DataFrame(m["forecast_mae"]).rename_axis("Slots ahead")
    st.dataframe(mae.style.format("{:.4f}").highlight_min(axis=1), use_container_width=True)
    st.write("What drives the 1-slot-ahead XGBoost forecast (feature importance)")
    st.bar_chart(pd.Series(m["feature_importance"]).sort_values(ascending=False))
    st.write(f"K-Means silhouette score (4 levels): {m['kmeans_silhouette_k4']:.3f}")
pricing_file = ROOT / "models" / "pricing_results.csv"
if pricing_file.exists():
    st.write("Pricing strategies over the whole test period, at three assumed price sensitivities")
    st.dataframe(pd.read_csv(pricing_file), hide_index=True, use_container_width=True)
