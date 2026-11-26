"""Train every model once, compare them on the test days, and save the results.

Run:  python train.py      (about 1 to 3 minutes)
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import silhouette_score
from statsmodels.tsa.arima.model import ARIMA

from data import FEATURES, build_features
from demand_levels import COLS, train_kmeans
from forecasting import fit_xgb, make_target
from common import FORECAST_MODEL_PATH, METRICS_PATH, MODELS_DIR, PATTERN_MODEL_PATH, load_config

warnings.filterwarnings("ignore")


def arima_mae(df, horizon):
    """ARIMA baseline: fit once per lot on train+val days, then forecast from every
    3rd test slot using only the data seen up to that slot."""
    pairs = {h: [] for h in range(1, horizon + 1)}
    for _, g in df.groupby("SystemCodeNumber"):
        y = g["occ_rate"].to_numpy()
        n = int((g["split"] != "test").sum())
        model = ARIMA(y[:n], order=(2, 0, 1)).fit()
        for t in range(n, len(y) - 1, 3):
            fc = model.apply(y[:t]).forecast(steps=horizon)
            for h in pairs:
                if t + h - 1 < len(y):
                    pairs[h].append((y[t + h - 1], fc[h - 1]))
    return {h: float(mae(*zip(*p))) for h, p in pairs.items()}


def main():
    H = load_config()["forecast_horizon"]
    MODELS_DIR.mkdir(exist_ok=True)

    print("1/4 Building features ...")
    df = build_features()
    train, val, test = (df["split"] == s for s in ("train", "val", "test"))
    complete = df[FEATURES].notna().all(axis=1)

    print("2/4 Forecast models: naive, Random Forest, XGBoost, ARIMA ...")
    results = {"naive": {}, "arima": {}, "random_forest": {}, "xgboost": {}}
    final_models, n_trees = {}, {}
    for h in range(1, H + 1):
        y = make_target(df, h)
        te = test & y.notna() & complete
        results["naive"][h] = mae(y[te], df.loc[te, "occ_rate"])          # "no change" forecast

        ok = train & y.notna() & complete
        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=42)
        rf.fit(df.loc[ok, FEATURES], y[ok])
        results["random_forest"][h] = mae(y[te], rf.predict(df.loc[te, FEATURES]))

        xgb = fit_xgb(df, h, train, val)
        n_trees[h] = xgb.best_iteration + 1
        results["xgboost"][h] = mae(y[te], np.clip(xgb.predict(df.loc[te, FEATURES]), 0, 1))
        if h == 1:
            importance = dict(zip(FEATURES, xgb.feature_importances_.astype(float)))
        # the model the live system uses: retrained on train + validation days
        final_models[h] = fit_xgb(df, h, train | val, n_estimators=n_trees[h])
    results["arima"] = arima_mae(df, H)
    joblib.dump({"models": final_models, "n_trees": n_trees,
                 "test_mae": results["xgboost"][1]}, FORECAST_MODEL_PATH)

    print("3/4 K-Means demand levels ...")
    rows = df.loc[train, COLS].dropna()
    elbow = {k: float(train_kmeans(rows, k)["kmeans"].inertia_) for k in range(2, 7)}
    km = train_kmeans(rows, 4)
    joblib.dump(km, PATTERN_MODEL_PATH)
    sample = km["scaler"].transform(rows.sample(4000, random_state=42))
    silhouette = float(silhouette_score(sample, km["kmeans"].predict(sample)))
    labels = [km["names"][int(c)] for c in km["kmeans"].predict(km["scaler"].transform(rows))]
    by_hour = pd.crosstab(df.loc[rows.index, "slot"].dt.hour, pd.Series(labels, index=rows.index))

    print("4/4 Saving results ...")
    METRICS_PATH.write_text(json.dumps({
        "forecast_mae": {m: {str(h): float(v) for h, v in r.items()} for m, r in results.items()},
        "feature_importance": importance,
        "kmeans_elbow_inertia": {str(k): v for k, v in elbow.items()},
        "kmeans_silhouette_k4": silhouette,
        "levels_by_hour": {str(h): row.to_dict() for h, row in by_hour.iterrows()},
    }, indent=2))

    print("\nTest-set MAE by model (occupancy rate, lower is better):")
    print(pd.DataFrame(results).rename_axis("slots_ahead").round(4).to_string())
    print(f"\nK-Means silhouette (k=4): {silhouette:.3f}")
    print("Top XGBoost features:", ", ".join(sorted(importance, key=importance.get, reverse=True)[:5]))
    print("\nDone. Next: python simulation.py  and  streamlit run dashboard/Home.py")


if __name__ == "__main__":
    main()
