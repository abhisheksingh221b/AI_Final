"""Simulation: replays the test days one 30-minute slot at a time, as if they were live.

For every slot it runs the same steps, in order:
  1. Check last slot's forecasts against what actually happened (and retrain on drift).
  2. Forecast each car park's occupancy for the next 3 hours.
  3. Label each car park's demand level.
  4. Choose each car park's price.
  5. Save everything to the database, which the dashboard reads.

Run:  python simulation.py      then press Start on the dashboard.
"""
import time
import traceback

import numpy as np

import db
import demand_levels
import forecasting
from common import FORECAST_MODEL_PATH, PATTERN_MODEL_PATH, load_config
from data import load_features, lots_table, rows_up_to, slot_rows, test_slots
from monitoring import ForecastMonitor
from pricing import expected_occupancy, set_prices


def safe(name, fn, fallback):
    """Run one step; if it crashes, log an alert and keep going with the fallback."""
    try:
        return fn()
    except Exception as exc:                       # noqa: BLE001
        traceback.print_exc()
        db.add_alert("-", f"{name} failed this slot ({exc}); kept last good values")
        return fallback


class Simulation:
    def __init__(self):
        self.cfg = load_config()
        self.df = load_features()
        self.slots = test_slots(self.df)
        saved = forecasting.load_models()
        self.models, self.n_trees, self.test_mae = saved["models"], saved["n_trees"], saved["test_mae"]
        self.level_model = demand_levels.load_model()
        db.init(lots_table(self.df), self.cfg)
        self.reset_state()

    def reset_state(self):
        """Forget everything from the last replay (at start-up and after Reset on the dashboard)."""
        self.monitor = ForecastMonitor(self.test_mae, self.cfg)
        self.price_memory = {}         # each lot's last chosen price (prices move at most Rs 10 a slot)
        self.last_prices = {}          # prices actually set in the previous slot
        self.last_forecasts = {}

    def business_metrics(self, batch):
        """Score last slot's prices against what happened, vs a fixed price.
        Demand in the data = demand at the base price; the assumed driver reaction
        turns it into the occupancy we would have seen at our price."""
        rows = batch[batch["SystemCodeNumber"].isin(list(self.last_prices))]
        if rows.empty:
            return []
        demand, cap = rows["occ_rate"].to_numpy(), rows["Capacity"].to_numpy()
        price = np.array([self.last_prices[lot] for lot in rows["SystemCodeNumber"]])
        fixed = float(self.cfg["fixed_price"])
        out = []
        for tag, p in (("dynamic", price), ("fixed", fixed)):
            occ = expected_occupancy(demand, p, self.cfg)
            out += [("ALL", f"revenue_{tag}", float((p * occ * cap * 0.5).sum())),   # 0.5 h per slot
                    ("ALL", f"occupancy_{tag}", float(occ.mean())),
                    ("ALL", f"full_lots_{tag}", float((occ >= 0.95).sum())),
                    ("ALL", f"target_gap_{tag}",
                     float(np.abs(occ - self.cfg["target_occupancy"]).mean()))]
        return out + [("ALL", "avg_price", float(price.mean()))]

    def step(self):
        """Run one slot. Returns False when the replay is over."""
        index = int(db.get_control("replay_index", "0"))
        if index >= len(self.slots):
            return False
        slot = self.slots[index]
        batch = slot_rows(self.df, slot)
        limits = db.get_limits()
        for key in ("over_weight", "under_weight"):          # live changes from the Admin page
            self.cfg[key] = float(db.get_control(key, self.cfg[key]))
        horizon = self.cfg["forecast_horizon"]

        # 1. Check last slot's forecasts against the new readings
        metrics, alerts, retrain = safe("Forecast check", lambda: self.monitor.check(
            batch, self.last_forecasts, self.last_prices, limits), ([], [], False))
        metrics += self.business_metrics(batch)
        if retrain:
            safe("Retraining", lambda: forecasting.retrain(
                self.models, self.n_trees, rows_up_to(self.df, slot), horizon), None)
            self.monitor.reset_after_retrain()
            alerts.append("Forecast models retrained on all data seen so far")

        # 2. Forecast, 3. demand level, 4. prices
        forecasts = safe("Forecasting", lambda: forecasting.predict(self.models, batch, horizon),
                         self.last_forecasts)
        levels = safe("Demand levels", lambda: demand_levels.label(self.level_model, batch),
                      {lot: "medium" for lot in batch["SystemCodeNumber"]})
        decisions = safe("Pricing", lambda: set_prices(forecasts, levels, limits,
                                                       self.price_memory, self.cfg), {})

        # 5. Save
        db.save_tick(slot, batch, forecasts, decisions, metrics, alerts)
        self.last_forecasts = forecasts
        self.last_prices = {lot: d["price"] for lot, d in decisions.items()}
        db.set_control("replay_index", index + 1)
        return True

    def run_forever(self):
        print(f"Ready: {len(self.slots)} slots to replay. Press Start on the dashboard.")
        while True:
            if db.get_control("reset_requested") == "1":
                self.reset_state()
                db.set_control("reset_requested", 0)
                print("Simulation reset.")
            if db.get_control("running") != "1":
                time.sleep(0.5)                              # idle: check the buttons often
                continue
            if self.step():
                print(f"slot {db.get_control('replay_index')} / {len(self.slots)}")
            else:
                db.set_control("running", 0)
                print("Replay finished. Press Reset on the Admin page to run it again.")
            time.sleep(float(db.get_control("tick_seconds", self.cfg["tick_seconds"])))


if __name__ == "__main__":
    if not (FORECAST_MODEL_PATH.exists() and PATTERN_MODEL_PATH.exists()):
        raise SystemExit("Trained models not found. Run  python train.py  first.")
    Simulation().run_forever()
