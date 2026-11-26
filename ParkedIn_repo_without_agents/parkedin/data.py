"""Data preparation: turns the raw CSV into one clean row per lot per 30-minute slot,
adds the features the forecaster needs, and splits the days into train / validation / test."""
import numpy as np
import pandas as pd

from common import DATA_RAW, FEATURES_PATH, load_config

# Inputs for the forecast models (all known at the moment we forecast)
FEATURES = ["occ_rate", "occ_lag_1", "occ_lag_2", "occ_lag_3", "occ_roll_3", "occ_diff",
            "occ_same_slot_yesterday", "hour", "dow", "traffic", "QueueLength",
            "IsSpecialDay", "Capacity"]


def build_features(save=True):
    if not DATA_RAW.exists():
        raise SystemExit("Dataset not found. Put dataset.csv in data/raw/ (see README).")
    df = pd.read_csv(DATA_RAW)

    # 1. Round every reading to the nearest 30-minute slot, keep the latest per lot and slot
    df["time"] = pd.to_datetime(df["LastUpdatedDate"] + " " + df["LastUpdatedTime"],
                                format="%d-%m-%Y %H:%M:%S")
    df["slot"] = df["time"].dt.round("30min")
    df = df.sort_values("time").drop_duplicates(["SystemCodeNumber", "slot"], keep="last")

    # 2. Same slots for every lot (skip rare stray slots where most lots have no reading)
    lots = sorted(df["SystemCodeNumber"].unique())
    lots_per_slot = df.groupby("slot")["SystemCodeNumber"].nunique()
    slots = sorted(lots_per_slot[lots_per_slot >= 0.8 * len(lots)].index)
    grid = pd.MultiIndex.from_product([lots, slots], names=["SystemCodeNumber", "slot"])
    df = df.set_index(["SystemCodeNumber", "slot"]).reindex(grid).reset_index()

    # 3. Fill the few missing readings
    df["traffic"] = df["TrafficConditionNearby"].map({"low": 0, "average": 1, "high": 2})
    by_lot = df.groupby("SystemCodeNumber")
    for col in ["Capacity", "Latitude", "Longitude"]:
        df[col] = by_lot[col].transform("first")
    for col in ["Occupancy", "QueueLength", "traffic"]:
        df[col] = by_lot[col].transform(lambda s: s.interpolate(limit_direction="both"))
    df["QueueLength"] = df["QueueLength"].round()
    df["traffic"] = df["traffic"].round()
    df["date"] = df["slot"].dt.normalize()
    df["IsSpecialDay"] = df.groupby("date")["IsSpecialDay"].transform("max").fillna(0).astype(int)

    # 4. One lot has wrong coordinates (20, 78): place it next to the others on the map
    bad = df["Latitude"] < 25
    df.loc[bad, "Latitude"] = df.loc[~bad, "Latitude"].median() + 0.004
    df.loc[bad, "Longitude"] = df.loc[~bad, "Longitude"].median() - 0.004

    # 5. Target and features (lags only look at the past)
    df = df.sort_values(["SystemCodeNumber", "slot"]).reset_index(drop=True)
    df["occ_rate"] = (df["Occupancy"] / df["Capacity"]).clip(0, 1)
    df["hour"] = df["slot"].dt.hour + df["slot"].dt.minute / 60
    df["dow"] = df["slot"].dt.dayofweek
    occ = df.groupby("SystemCodeNumber")["occ_rate"]
    for k in (1, 2, 3):
        df[f"occ_lag_{k}"] = occ.shift(k)
    df["occ_roll_3"] = occ.transform(lambda s: s.shift(1).rolling(3).mean())
    df["occ_diff"] = occ.diff()
    df["occ_same_slot_yesterday"] = df.groupby(
        ["SystemCodeNumber", df["slot"].dt.time])["occ_rate"].shift(1)

    # 6. Split by whole days in time order (never randomly)
    cfg = load_config()
    days = sorted(df["date"].unique())
    train_end = days[int(len(days) * cfg["train_fraction"]) - 1]
    val_end = days[int(len(days) * (cfg["train_fraction"] + cfg["val_fraction"])) - 1]
    df["split"] = np.select([df["date"] <= train_end, df["date"] <= val_end],
                            ["train", "val"], "test")

    if save:
        df.to_pickle(FEATURES_PATH)
    return df


def load_features():
    """The cleaned feature table saved by build_features() (run train.py first)."""
    return pd.read_pickle(FEATURES_PATH)


def test_slots(df):
    """The test days' 30-minute slots, in time order (what the simulation replays)."""
    return sorted(df.loc[df["split"] == "test", "slot"].unique())


def slot_rows(df, slot):
    """All lots' rows for one slot."""
    return df[(df["split"] == "test") & (df["slot"] == slot)].copy()


def rows_up_to(df, slot):
    """Everything seen up to and including a slot (used when retraining)."""
    return df[df["slot"] <= slot]


def lots_table(df):
    """One row per lot: capacity and map position."""
    return (df.groupby("SystemCodeNumber")[["Capacity", "Latitude", "Longitude"]]
            .first().reset_index())
