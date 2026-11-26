"""SQLite database shared by the simulation (which writes it) and the dashboard (which reads it)."""
import sqlite3

import pandas as pd

from common import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS lots (lot_id TEXT PRIMARY KEY, capacity INTEGER,
    latitude REAL, longitude REAL, min_price REAL, max_price REAL);
CREATE TABLE IF NOT EXISTS readings (lot_id TEXT, ts TEXT, occupancy REAL, occ_rate REAL,
    queue_length REAL, traffic REAL, is_special_day INTEGER);
CREATE TABLE IF NOT EXISTS forecasts (lot_id TEXT, made_at TEXT, slot_ahead INTEGER, predicted REAL);
CREATE TABLE IF NOT EXISTS prices (lot_id TEXT, ts TEXT, price REAL, level TEXT,
    expected_occ REAL, reason TEXT);
CREATE TABLE IF NOT EXISTS metrics (ts TEXT, lot_id TEXT, name TEXT, value REAL);
CREATE TABLE IF NOT EXISTS alerts (ts TEXT, message TEXT);
CREATE TABLE IF NOT EXISTS control (key TEXT PRIMARY KEY, value TEXT);
"""


def connect():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")      # dashboard can read while the simulation writes
    return con


def run(sql, params=()):
    with connect() as con:
        con.execute(sql, params)


def read(sql, params=()):
    with connect() as con:
        return pd.read_sql(sql, con, params=params)


def init(lots_df, cfg):
    """Create the tables, the control settings and the lots (with default price limits)."""
    with connect() as con:
        con.executescript(SCHEMA)
        for key, value in [("running", 0), ("replay_index", 0), ("tick_seconds", cfg["tick_seconds"]),
                           ("over_weight", cfg["over_weight"]), ("under_weight", cfg["under_weight"])]:
            con.execute("INSERT OR IGNORE INTO control VALUES (?, ?)", (key, str(value)))
        con.executemany("INSERT OR IGNORE INTO lots VALUES (?, ?, ?, ?, ?, ?)", [
            (r.SystemCodeNumber, int(r.Capacity), r.Latitude, r.Longitude,
             cfg["min_price"], cfg["max_price"]) for r in lots_df.itertuples()])


# ---------- control: how the dashboard's buttons talk to the simulation ----------

def get_control(key, default=None):
    df = read("SELECT value FROM control WHERE key = ?", (key,))
    return df["value"].iloc[0] if len(df) else default


def set_control(key, value):
    run("INSERT OR REPLACE INTO control VALUES (?, ?)", (key, str(value)))


def reset_simulation():
    """Delete everything the replay wrote and go back to the first slot."""
    with connect() as con:
        for table in ["readings", "forecasts", "prices", "metrics", "alerts"]:
            con.execute(f"DELETE FROM {table}")
    set_control("replay_index", 0)
    set_control("running", 0)
    set_control("reset_requested", 1)


# ---------- writing (simulation) ----------

def save_tick(ts, batch, forecasts, decisions, metrics, alerts):
    ts = str(ts)
    with connect() as con:
        con.executemany("INSERT INTO readings VALUES (?, ?, ?, ?, ?, ?, ?)", [
            (r.SystemCodeNumber, ts, r.Occupancy, r.occ_rate, r.QueueLength, r.traffic,
             int(r.IsSpecialDay)) for r in batch.itertuples()])
        con.executemany("INSERT INTO forecasts VALUES (?, ?, ?, ?)", [
            (lot, ts, h + 1, v) for lot, values in forecasts.items() for h, v in enumerate(values)])
        con.executemany("INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?)", [
            (lot, ts, d["price"], d["level"], d["expected_occ"], d["reason"])
            for lot, d in decisions.items()])
        con.executemany("INSERT INTO metrics VALUES (?, ?, ?, ?)",
                        [(ts, lot, name, value) for lot, name, value in metrics])
        con.executemany("INSERT INTO alerts VALUES (?, ?)", [(ts, a) for a in alerts])


def add_alert(ts, message):
    run("INSERT INTO alerts VALUES (?, ?)", (str(ts), message))


def get_limits():
    df = read("SELECT lot_id, min_price, max_price FROM lots")
    return {r.lot_id: (r.min_price, r.max_price) for r in df.itertuples()}


def set_limits(lot_id, min_price, max_price):
    run("UPDATE lots SET min_price = ?, max_price = ? WHERE lot_id = ?",
        (float(min_price), float(max_price), lot_id))


# ---------- reading (dashboard) ----------

def latest_status():
    """One row per lot for the latest slot."""
    return read("""
        SELECT l.*, p.ts, p.price, p.level, p.expected_occ, p.reason,
               r.occ_rate, r.queue_length, r.is_special_day
        FROM prices p
        JOIN lots l ON l.lot_id = p.lot_id
        JOIN readings r ON r.lot_id = p.lot_id AND r.ts = p.ts
        WHERE p.ts = (SELECT MAX(ts) FROM prices)
        ORDER BY l.lot_id""")


def history(lot_id):
    return read("""
        SELECT r.ts, r.occ_rate, r.queue_length, p.price, p.level, p.expected_occ
        FROM readings r JOIN prices p ON p.lot_id = r.lot_id AND p.ts = r.ts
        WHERE r.lot_id = ? ORDER BY r.ts""", (lot_id,))


def latest_forecast(lot_id):
    return read("""SELECT slot_ahead, predicted FROM forecasts
                   WHERE lot_id = ? AND made_at = (SELECT MAX(made_at) FROM forecasts)
                   ORDER BY slot_ahead""", (lot_id,))


def lots():
    return read("SELECT * FROM lots ORDER BY lot_id")


def metric_series(name, lot_id="ALL"):
    return read("SELECT ts, value FROM metrics WHERE name = ? AND lot_id = ? ORDER BY ts",
                (name, lot_id))


def metric_totals():
    return read("""SELECT name, SUM(value) AS total, AVG(value) AS mean, COUNT(*) AS n
                   FROM metrics WHERE lot_id = 'ALL' GROUP BY name""")


def recent_alerts(n=20):
    return read("SELECT ts, message FROM alerts ORDER BY rowid DESC LIMIT ?", (n,))
