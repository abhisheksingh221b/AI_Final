"""Grounded Gemini features for parking operations.

All data selection and calculations happen locally. Gemini receives compact JSON
context and returns concise JSON; it never writes to the database or executes SQL.
"""
from __future__ import annotations

import json
import os
from datetime import timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from openai import OpenAI

import db
from common import load_config

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
SYSTEM_RULES = (
    "You are ParkedIn's business operations assistant. Use only supplied data. "
    "Do not invent measurements, causes, or guarantees. State uncertainty when useful. "
    "Keep the response concise, specific, and business-friendly. Return only JSON matching the schema."
)

SCHEMAS = {
    "recommendation": {"type": "OBJECT", "properties": {
        "recommendation": {"type": "STRING"}, "alternatives": {"type": "ARRAY", "items": {"type": "STRING"}},
        "caveat": {"type": "STRING"}}, "required": ["recommendation", "alternatives", "caveat"]},
    "explanation": {"type": "OBJECT", "properties": {
        "summary": {"type": "STRING"}, "drivers": {"type": "ARRAY", "items": {"type": "STRING"}},
        "caveat": {"type": "STRING"}}, "required": ["summary", "drivers", "caveat"]},
    "report": {"type": "OBJECT", "properties": {
        "headline": {"type": "STRING"}, "summary": {"type": "STRING"},
        "highlights": {"type": "ARRAY", "items": {"type": "STRING"}},
        "actions": {"type": "ARRAY", "items": {"type": "STRING"}}},
        "required": ["headline", "summary", "highlights", "actions"]},
    "chat": {"type": "OBJECT", "properties": {
        "answer": {"type": "STRING"}, "supporting_facts": {"type": "ARRAY", "items": {"type": "STRING"}},
        "limitation": {"type": "STRING"}}, "required": ["answer", "supporting_facts", "limitation"]},
    "insights": {"type": "OBJECT", "properties": {
        "insights": {"type": "ARRAY", "items": {"type": "STRING"}},
        "anomalies": {"type": "ARRAY", "items": {"type": "STRING"}},
        "follow_up": {"type": "STRING"}}, "required": ["insights", "anomalies", "follow_up"]},
    "strategy": {"type": "OBJECT", "properties": {
        "recommendations": {"type": "ARRAY", "items": {"type": "STRING"}},
        "expected_tradeoffs": {"type": "ARRAY", "items": {"type": "STRING"}},
        "next_measurement": {"type": "STRING"}},
        "required": ["recommendations", "expected_tradeoffs", "next_measurement"]},
}


@lru_cache(maxsize=4)
def _gemini_client_for_key(key):
    """Keep a live SDK client across Streamlit reruns instead of recreating it per call."""
    return genai.Client(api_key=key)


@lru_cache(maxsize=4)
def _openai_client_for_key(key):
    return OpenAI(api_key=key)


def _gemini_client(api_key=None):
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    return _gemini_client_for_key(key)


def _openai_client(api_key=None):
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    return _openai_client_for_key(key)


def _strict_openai_schema(schema):
    """OpenAI strict JSON schema requires all properties and closed objects."""
    result = dict(schema)
    kind = result.get("type")
    if kind == "OBJECT":
        result["type"] = "object"
        result["additionalProperties"] = False
        result["required"] = list(result.get("properties", {}))
        result["properties"] = {key: _strict_openai_schema(value)
                                for key, value in result.get("properties", {}).items()}
    elif kind == "ARRAY":
        result["type"] = "array"
        result["items"] = _strict_openai_schema(result["items"])
    elif kind == "STRING":
        result["type"] = "string"
    return result


def _generate_with_provider(provider, kind, prompt, gemini_key, openai_key):
    if provider == "gemini":
        response = _gemini_client(gemini_key).models.generate_content(
            model=MODEL, contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json",
                                               response_schema=SCHEMAS[kind], temperature=0.2),
        )
        response_text = response.text
    else:
        response = _openai_client(openai_key).responses.create(
            model=OPENAI_MODEL, input=prompt,
            text={"format": {"type": "json_schema", "name": kind,
                             "schema": _strict_openai_schema(SCHEMAS[kind]), "strict": True}},
        )
        response_text = response.output_text
    if not response_text:
        raise RuntimeError("provider returned an empty response")
    result = json.loads(response_text)
    if not isinstance(result, dict) or any(field not in result for field in SCHEMAS[kind]["required"]):
        raise RuntimeError("provider response did not match the expected schema")
    return result


def _generate(kind, instruction, context, api_key=None, openai_api_key=None):
    prompt = (f"{SYSTEM_RULES}\nTask: {instruction}\n"
              f"Trusted parking data (JSON):\n{json.dumps(context, default=str, ensure_ascii=False)}")
    gemini_key = api_key or os.getenv("GEMINI_API_KEY")
    openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    primary = os.getenv("AI_PRIMARY_PROVIDER", "gemini").strip().lower()
    if primary not in ("gemini", "openai"):
        primary = "gemini"
    keys = {"gemini": gemini_key, "openai": openai_key}
    providers = [provider for provider in (primary, "openai" if primary == "gemini" else "gemini")
                 if keys[provider]]
    if not providers:
        raise RuntimeError("Configure GEMINI_API_KEY, OPENAI_API_KEY, or both to enable AI features.")
    errors = []
    for provider in providers:
        try:
            return _generate_with_provider(provider, kind, prompt, gemini_key, openai_key)
        except Exception as exc:  # API, network, and invalid structured output trigger failover.
            errors.append(f"{provider}: {type(exc).__name__}: {str(exc)[:240]}")
    raise RuntimeError("AI providers failed. " + " | ".join(errors))


def _latest():
    return db.latest_status()


def _history(days=7):
    rows = db.read("""SELECT r.lot_id, r.ts, r.occ_rate, r.queue_length,
                            p.price, p.level
                     FROM readings r JOIN prices p ON p.lot_id=r.lot_id AND p.ts=r.ts
                     ORDER BY r.ts""")
    if rows.empty:
        return rows
    rows["ts"] = pd.to_datetime(rows["ts"])
    cutoff = rows["ts"].max() - timedelta(days=days)
    return rows[rows["ts"] >= cutoff].copy()


def _metrics_frame(frame):
    if frame.empty:
        return {"message": "No completed simulation readings are available yet."}
    return {
        "period_start": str(frame.ts.min()), "period_end": str(frame.ts.max()),
        "lot_slots": int(len(frame)), "average_occupancy_pct": round(float(frame.occ_rate.mean() * 100), 1),
        "average_price_rs_per_hour": round(float(frame.price.mean()), 1),
        "lot_slots_95pct_or_more_full": int((frame.occ_rate >= .95).sum()),
        "occupancy_by_lot_pct": (frame.groupby("lot_id").occ_rate.mean().mul(100).round(1).to_dict()),
        "price_by_lot_rs": frame.groupby("lot_id").price.mean().round(1).to_dict(),
    }


def recommend_parking(latitude, longitude, api_key=None, openai_api_key=None):
    lots = _latest()
    if lots.empty:
        raise RuntimeError(
            "No live parking data is available yet. Start the simulation first.")
    lat, lon = float(latitude), float(longitude)
    # Haversine distance from user-entered coordinates to lot coordinates.
    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2, lon2 = np.radians(lots.latitude.to_numpy()), np.radians(
        lots.longitude.to_numpy())
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1) * \
        np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    lots["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    # Rank locally: shorter distance, more expected space, and lower price are preferred.
    lots["rank_score"] = (lots.distance_km / max(float(lots.distance_km.max()), .1)
                          + lots.expected_occ
                          + lots.price / max(float(lots.price.max()), 1))
    shortlist = lots.sort_values("rank_score").head(5)
    candidates = [{"lot": r.lot_id, "distance_km": round(float(r.distance_km), 2),
                   "current_occupancy_pct": round(float(r.occ_rate*100), 1),
                   "expected_occupancy_at_suggested_price_pct": round(float(r.expected_occ*100), 1),
                   "price_rs_per_hour": round(float(r.price), 1), "demand_level": r.level}
                  for r in shortlist.itertuples()]
    return _generate("recommendation", "Recommend the best option for a driver. Explain the tradeoff and list up to two alternatives.",
                     {"user_location": {"latitude": lat, "longitude": lon}, "ranked_candidates": candidates},
                     api_key, openai_api_key)


def explain_price(lot_id, api_key=None, openai_api_key=None):
    rows = _latest()
    match = rows[rows.lot_id == lot_id]
    if match.empty:
        raise RuntimeError("No current price is available for that lot.")
    r = match.iloc[0]
    context = {"lot": lot_id, "suggested_price_rs_per_hour": float(r.price),
               "current_occupancy_pct": round(float(r.occ_rate*100), 1),
               "expected_occupancy_at_suggested_price_pct": round(float(r.expected_occ*100), 1),
               "demand_level": r.level, "stored_pricing_reason": r.reason,
               "price_limits_rs": [float(r.min_price), float(r.max_price)],
               "target_occupancy_pct": round(load_config()["target_occupancy"]*100, 1)}
    return _generate("explanation", "Explain the selected price to a parking operator. Separate the rule and occupancy signals from the pricing objective; mention the elasticity estimate is assumed.", context, api_key, openai_api_key)


def weekly_report(api_key=None, openai_api_key=None):
    frame = _history(7)
    context = _metrics_frame(frame)
    if not frame.empty:
        start = str(frame.ts.min())
        end = str(frame.ts.max())
        metrics = db.read("""SELECT name, SUM(value) AS total, AVG(value) AS mean
                              FROM metrics WHERE lot_id='ALL' AND ts >= ? AND ts <= ?
                              GROUP BY name""", (start, end))
        context["fixed_price_comparison_for_period"] = (
            metrics.set_index("name").round(3).to_dict("index") if not metrics.empty else {})
    else:
        context["fixed_price_comparison_for_period"] = {}
    return _generate("report", "Write an executive weekly business report. Keep it factual; identify operational implications and practical next actions.", context, api_key, openai_api_key)


def chat_about_data(question, api_key=None, openai_api_key=None):
    frame = _history(30)
    context = _metrics_frame(frame)
    if not frame.empty:
        context["daily_summary"] = frame.assign(day=frame.ts.dt.date.astype(str)).groupby("day").agg(
            average_occupancy=("occ_rate", "mean"), average_price=("price", "mean"),
            lot_slots=("lot_id", "size")).round(3).tail(30).to_dict("index")
        context["available_lots"] = sorted(frame.lot_id.unique().tolist())
    return _generate("chat", "Answer the user's question using only the supplied data. If it cannot be answered, say what is missing. Never make up SQL results.",
                     {"question": question[:1000], "data": context}, api_key, openai_api_key)


def discover_insights(api_key=None, openai_api_key=None):
    frame = _history(30)
    if frame.empty:
        raise RuntimeError("No historical simulation data is available yet.")
    frame["hour"] = frame.ts.dt.hour
    hourly = frame.groupby("hour").agg(occupancy=("occ_rate", "mean"), price=("price", "mean"),
                                       records=("lot_id", "size")).round(3)
    lot_day = frame.assign(day=frame.ts.dt.date.astype(str)).groupby(["lot_id", "day"]).agg(
        occupancy=("occ_rate", "mean"), price=("price", "mean")).reset_index()
    # Flag unusually high/low daily occupancy via IQR; deterministic statistics, not model claims.
    q1, q3 = lot_day.occupancy.quantile([.25, .75])
    spread = q3-q1
    outliers = lot_day[(lot_day.occupancy < q1-1.5*spread)
                       | (lot_day.occupancy > q3+1.5*spread)]
    context = {"period": [str(frame.ts.min()), str(frame.ts.max())],
               "overall": _metrics_frame(frame), "hourly_patterns": hourly.to_dict("index"),
               "daily_occupancy_iqr_outliers": outliers.round(3).to_dict("records")[:30],
               "method_note": "Outliers are lot-days outside 1.5 IQR fences for daily mean occupancy."}
    return _generate("insights", "Identify the strongest supported trends and explain listed anomalies without claiming causation.", context, api_key, openai_api_key)


def advise_revenue_strategy(api_key=None, openai_api_key=None):
    frame = _history(30)
    if frame.empty:
        raise RuntimeError("No historical simulation data is available yet.")
    by_lot = frame.groupby("lot_id").agg(avg_occupancy=("occ_rate", "mean"), avg_price=("price", "mean"),
                                         high_occupancy_slots=(
                                             "occ_rate", lambda s: int((s >= .9).sum())),
                                         observations=("occ_rate", "size")).round(3)
    by_hour = frame.assign(hour=frame.ts.dt.hour).groupby("hour").agg(
        avg_occupancy=("occ_rate", "mean"), avg_price=("price", "mean")).round(3)
    context = {"period": [str(frame.ts.min()), str(frame.ts.max())], "by_lot": by_lot.to_dict("index"),
               "by_hour": by_hour.to_dict("index"), "pricing_assumption":
               "Revenue simulations use a configured price elasticity, not measured customer response."}
    return _generate("strategy", "Suggest up to three cautious, testable revenue actions based on the observed patterns. Include tradeoffs and one metric to track for each decision cycle.", context, api_key, openai_api_key)
