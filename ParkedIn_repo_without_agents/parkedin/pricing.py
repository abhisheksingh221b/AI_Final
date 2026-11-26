"""Pricing: chooses each lot's price for the next slot.

1. Rule-based start: base price x demand-level multiplier x (1 + forecast - 85% target).
2. Grid search: try the allowed prices (at most Rs 10 away from the lot's last price,
   inside its min/max limits) and keep the one with the best score:
       score = revenue - over_weight x (too full)^2 - under_weight x (too empty)^2
"""
import numpy as np

MULTIPLIER = {"low": 0.8, "medium": 1.0, "high": 1.2, "peak": 1.4}


def expected_occupancy(demand, price, cfg):
    """Assumed driver reaction: occupancy falls as the price rises above the base price."""
    ratio = np.asarray(price, dtype=float) / cfg["base_price"]
    return np.clip(demand * ratio ** -cfg["elasticity"], 0, 1)


def rule_price(forecast, level, cfg, lo, hi):
    p = cfg["base_price"] * MULTIPLIER[level] * (1 + forecast - cfg["target_occupancy"])
    p = round(p / cfg["price_step"]) * cfg["price_step"]          # Rs 10 steps
    return float(np.clip(p, lo, hi))


def choose_price(lot, forecast, level, lo, hi, last_prices, cfg):
    """Best allowed price for one lot. Updates last_prices[lot] with the choice."""
    step, target = cfg["price_step"], cfg["target_occupancy"]
    start = rule_price(forecast, level, cfg, lo, hi)
    last = last_prices.get(lot, start)
    prices = np.arange(lo, hi + 1, step, dtype=float)
    near = prices[np.abs(prices - last) <= step]
    if near.size:                                      # empty only if limits just changed
        prices = near
    occ = expected_occupancy(forecast, prices, cfg)
    score = (prices * occ / cfg["max_price"]                            # revenue, scaled 0..1
             - cfg["over_weight"] * np.maximum(0, occ - target) ** 2
             - cfg["under_weight"] * np.maximum(0, target - occ) ** 2)
    best = int(score.argmax())                         # ties go to the cheaper price
    last_prices[lot] = float(prices[best])
    return {"price": float(prices[best]), "expected_occ": float(occ[best]), "level": level,
            "reason": f"forecast {forecast:.0%} full, {level} demand; "
                      f"rule price Rs {start:.0f}, best score at Rs {prices[best]:.0f}"}


def set_prices(forecasts, levels, limits, last_prices, cfg):
    """{lot: {"price", "expected_occ", "level", "reason"}} for every lot."""
    default = (cfg["min_price"], cfg["max_price"])
    return {lot: choose_price(lot, fc[0], levels[lot], *limits.get(lot, default), last_prices, cfg)
            for lot, fc in forecasts.items()}
