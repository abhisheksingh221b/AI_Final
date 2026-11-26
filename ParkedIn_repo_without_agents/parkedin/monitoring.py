"""Forecast monitoring: tracks the forecast error, raises alerts, and asks for retraining on drift."""
from collections import defaultdict, deque

import numpy as np


class ForecastMonitor:
    """Keeps the recent forecast errors and the slots-at-maximum-price count for each lot."""

    def __init__(self, test_mae, cfg):
        self.test_mae, self.cfg = test_mae, cfg
        self.errors = defaultdict(lambda: deque(maxlen=cfg["monitor_window"]))  # per lot
        self.at_max = defaultdict(int)          # slots in a row at the maximum price
        self.drifting = False

    def check(self, batch, last_forecasts, last_prices, limits):
        """Returns (metrics, alerts, retrain_now)."""
        metrics, alerts, retrain = [], [], False

        # 1. Error of last tick's 1-slot-ahead forecast, per lot
        for lot, actual in zip(batch["SystemCodeNumber"], batch["occ_rate"]):
            if lot in last_forecasts:
                err = abs(actual - last_forecasts[lot][0])
                self.errors[lot].append(err)
                metrics.append((lot, "abs_error", float(err)))

        # 2. Rolling error vs the test error: drift?
        full = [np.mean(e) for e in self.errors.values() if len(e) == e.maxlen]
        if full:
            rolling = float(np.mean(full))
            metrics.append(("ALL", "rolling_mae", rolling))
            too_high = rolling > self.cfg["drift_factor"] * self.test_mae
            if too_high and not self.drifting:
                alerts.append(f"Forecast error has drifted: rolling MAE {rolling:.3f} "
                              f"vs test MAE {self.test_mae:.3f}")
                retrain = bool(self.cfg["auto_retrain"])
            self.drifting = too_high

        # 3. A lot stuck at its maximum price for 4 hours (8 slots)
        for lot, price in last_prices.items():
            top = limits.get(lot, (0, self.cfg["max_price"]))[1]
            self.at_max[lot] = self.at_max[lot] + 1 if price >= top else 0
            if self.at_max[lot] == 8:
                alerts.append(f"{lot} has been at its maximum price (Rs {top:.0f}) for 4 hours")
        return metrics, alerts, retrain

    def reset_after_retrain(self):
        self.errors.clear()
        self.drifting = False
