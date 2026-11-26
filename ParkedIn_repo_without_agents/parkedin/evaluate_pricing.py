"""Compare pricing strategies over the whole test period, for the report.

Run:  python evaluate_pricing.py      (after train.py; under a minute)

Strategies: fixed price, rules only, rules + grid search (ParkedIn), each at three
assumed price sensitivities (0.2, 0.4, 0.6). Writes models/pricing_results.csv.
"""
import numpy as np
import pandas as pd

import demand_levels
import forecasting
from common import MODELS_DIR, load_config
from data import load_features, slot_rows, test_slots
from pricing import expected_occupancy, rule_price, set_prices


def replay(strategy, cfg, batches, forecasts, levels):
    memory = {}                                      # each lot's last chosen price
    last, revenue, gaps, full, prices = {}, 0.0, [], 0, []
    for batch, fc, lv in zip(batches, forecasts, levels):
        if last:                                     # score last tick's prices on this slot
            p = np.array([last[lot] for lot in batch["SystemCodeNumber"]])
            occ = expected_occupancy(batch["occ_rate"].to_numpy(), p, cfg)
            revenue += float((p * occ * batch["Capacity"].to_numpy() * 0.5).sum())
            gaps.append(float(np.abs(occ - cfg["target_occupancy"]).mean()))
            full += int((occ >= 0.95).sum())
            prices.extend(p)
        if strategy == "Fixed price":
            last = {lot: float(cfg["fixed_price"]) for lot in fc}
        elif strategy == "Rules only":
            last = {lot: rule_price(v[0], lv[lot], cfg, cfg["min_price"], cfg["max_price"])
                    for lot, v in fc.items()}
        else:
            last = {lot: d["price"] for lot, d in set_prices(fc, lv, {}, memory, cfg).items()}
    return {"revenue_rs": round(revenue), "avg_gap_from_85pct": round(float(np.mean(gaps)), 3),
            "lot_slots_95pct_full": full, "avg_price_rs": round(float(np.mean(prices)), 1)}


def main():
    base = load_config()
    df = load_features()
    models, level_model = forecasting.load_models()["models"], demand_levels.load_model()
    batches = [slot_rows(df, s) for s in test_slots(df)]
    forecasts = [forecasting.predict(models, b, base["forecast_horizon"]) for b in batches]
    levels = [demand_levels.label(level_model, b) for b in batches]

    rows = [{"elasticity": e, "strategy": s,
             **replay(s, dict(base, elasticity=e), batches, forecasts, levels)}
            for e in (0.2, 0.4, 0.6) for s in ("Fixed price", "Rules only", "Rules + grid search")]
    table = pd.DataFrame(rows)
    fixed = table[table["strategy"] == "Fixed price"].set_index("elasticity")["revenue_rs"]
    table["revenue_vs_fixed_pct"] = (100 * (table["revenue_rs"] / table["elasticity"].map(fixed) - 1)).round(1)
    table.to_csv(MODELS_DIR / "pricing_results.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
