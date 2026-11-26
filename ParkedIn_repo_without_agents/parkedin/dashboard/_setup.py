"""Shared by the dashboard pages: lets them import db.py from the project folder.

On Streamlit Cloud there is no second terminal for the simulation, so the dashboard
starts everything itself in the background: it downloads the dataset, trains the models
(first launch only, about 2-3 minutes), then runs the simulation loop.
On a laptop this only happens if you open the dashboard without starting
python simulation.py first.
"""
import os
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import db  # noqa: E402
from common import (DATA_RAW, DB_PATH, FEATURES_PATH, FORECAST_MODEL_PATH,  # noqa: E402,F401
                    MODELS_DIR, PATTERN_MODEL_PATH, load_config)

COLOURS = {"low": "#2e7d32", "medium": "#9e9d24", "high": "#ef6c00", "peak": "#c62828"}
DATA_URL = ("https://raw.githubusercontent.com/sh-kartik18/"
            "Dynamic-Pricing-for-Urban-Parking-Lots/main/dataset.csv")
ON_CLOUD = str(ROOT).startswith("/mount/src") or os.environ.get("PARKEDIN_AUTOSTART") == "1"


@st.cache_resource          # runs once per server, shared by every visitor and page
def start_backend():
    state = {"status": "Starting", "error": None}

    def work():
        try:
            for folder in (DATA_RAW.parent, FEATURES_PATH.parent, MODELS_DIR):
                folder.mkdir(parents=True, exist_ok=True)
            if not DATA_RAW.exists():
                state["status"] = "Downloading the dataset"
                urllib.request.urlretrieve(DATA_URL, DATA_RAW)
            if not (FORECAST_MODEL_PATH.exists() and PATTERN_MODEL_PATH.exists()):
                state["status"] = "Training the models (first launch only, about 2-3 minutes)"
                import train
                train.main()
            state["status"] = "Starting the simulation"
            import simulation
            sim = simulation.Simulation()
            state["status"] = "ready"
            if not (MODELS_DIR / "pricing_results.csv").exists():   # table for the Admin page
                import evaluate_pricing
                threading.Thread(target=evaluate_pricing.main, daemon=True).start()
            sim.run_forever()
        except Exception as exc:                   # noqa: BLE001
            traceback.print_exc()
            state["status"], state["error"] = "error", repr(exc)

    threading.Thread(target=work, daemon=True).start()
    return state


def check_ready(st):
    """Wait for the background setup (on the cloud), or start it on a laptop if the simulation is not running."""
    if ON_CLOUD or not DB_PATH.exists():
        state = start_backend()
        if state["status"] == "error":
            st.error(f"Setup failed: {state['error']}. Open 'Manage app' to see the logs.")
            st.stop()
        if state["status"] != "ready":
            st.info(f"⏳ {state['status']} … This page refreshes by itself.")
            time.sleep(3)
            st.rerun()


def refresh_seconds():
    return max(1.0, float(db.get_control("tick_seconds", 5)))
