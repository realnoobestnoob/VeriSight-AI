"""Aggregated dashboards. Usage numbers are app activity, NOT real-world scam/threat rates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def operational_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total": 0}
    total = len(df)
    ok = df[df.status == "succeeded"]
    lat = ok["inference_latency_ms"].dropna()
    return {
        "total": total,
        "failure_rate": float((df.status != "succeeded").mean()),
        "rejected_input_rate": float((df.status == "rejected_input").mean()),
        "latency_p50_ms": float(lat.quantile(0.5)) if len(lat) else None,
        "latency_p95_ms": float(lat.quantile(0.95)) if len(lat) else None,
        "llm_used_rate": float(df.llm_used.astype(bool).mean()),
    }


def scans_per_day(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["day", "scans"])
    d = pd.to_datetime(df["created_at"], utc=True).dt.date
    return d.value_counts().sort_index().rename_axis("day").reset_index(name="scans")


def count_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df.empty or column not in df:
        return pd.DataFrame(columns=[column, "count"])
    s = df[column].fillna("none").value_counts()
    return s.rename_axis(column).reset_index(name="count")


def verified_model_performance(artifacts_dir: str = "model_artifacts") -> dict | None:
    """Metrics from the labelled evaluation of the active release (not live traffic)."""
    pointer = Path(artifacts_dir) / "current.json"
    if not pointer.is_file():
        return None
    rid = json.loads(pointer.read_text(encoding="utf-8"))["release_id"]
    meta = Path(artifacts_dir) / "releases" / rid / "metadata.json"
    if not meta.is_file():
        return None
    data = json.loads(meta.read_text(encoding="utf-8"))
    return {
        "model_release_id": rid,
        "dataset": data.get("dataset", {}),
        "metrics": data.get("metrics_summary", {}),
    }
