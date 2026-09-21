# -*- coding: utf-8 -*-
"""
04 Recursive stress-test rollout.

The rollout starts from the December 2019 cohort state and follows the realized
January–December 2020 unemployment path. The December 2019 cohort composition
and current-UPB weights remain fixed across all horizons.

Inputs
------
processed/03_learner/model_selected.joblib
processed/03_learner/feature_spec.json
processed/02_cohort/cohort_month_panel.parquet
macro/UNRATE.csv

Outputs
-------
processed/04_rollout/mu_hat.csv
processed/04_rollout/rollout_by_cohort.parquet
processed/04_rollout/summary.json

Run
---
python 04_rollout.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


BASE = Path(os.environ.get("FREDDIE_BASE", Path(__file__).resolve().parent)).expanduser()
LEARNER_DIR = BASE / "processed" / "03_learner"
OUT_DIR = BASE / "processed" / "04_rollout"
UNRATE_CSV = BASE / "macro" / "UNRATE.csv"
PANEL_PATH = BASE / "processed" / "02_cohort" / "cohort_month_panel.parquet"

STATE_MONTH = "2019-12"
HORIZON = 12
MIN_ACTIVE = 100
EXPECTED_STARTING_COHORTS = 62


def load_inputs():
    with open(LEARNER_DIR / "feature_spec.json", encoding="utf-8") as file:
        feature_spec = json.load(file)
    model = joblib.load(LEARNER_DIR / "model_selected.joblib")

    if not PANEL_PATH.exists():
        sys.exit("Cohort panel not found. Run 02_build_cohort_panel.py first.")
    panel = pd.read_parquet(PANEL_PATH)
    if "month" in panel.columns:
        panel["month"] = pd.to_datetime(panel["month"])
    else:
        panel["month"] = pd.to_datetime(
            panel["monthly_reporting_period"], format="%Y%m"
        )
    panel["month"] = panel["month"].dt.to_period("M").dt.to_timestamp()

    unemployment = pd.read_csv(UNRATE_CSV)
    date_column = (
        "observation_date" if "observation_date" in unemployment.columns else "DATE"
    )
    unemployment["month"] = (
        pd.to_datetime(unemployment[date_column]).dt.to_period("M").dt.to_timestamp()
    )
    unemployment_by_month = unemployment.set_index("month")["UNRATE"].astype(float)
    return feature_spec, model, panel, unemployment_by_month


def feature_row(
    previous_rate,
    next_unemployment,
    vintage,
    next_month,
    fico_bucket,
    features,
):
    age = max(0, (next_month.year - vintage) * 12 + next_month.month - 1)
    values = {
        "y_lag": previous_rate,
        "unrate_next": next_unemployment,
        "vintage_age": age,
    }
    for feature in features:
        if feature.startswith("fico_"):
            values[feature] = 1.0 if feature == f"fico_{fico_bucket}" else 0.0
    return [values[feature] for feature in features]


def recursive_rollout(model, states, unemployment_path, features):
    records = []
    current_rate = states["initial_rate"].to_numpy().copy()

    for horizon, (month, unemployment) in enumerate(unemployment_path, start=1):
        design = np.array(
            [
                feature_row(
                    current_rate[row],
                    unemployment,
                    states["vintage"].iat[row],
                    month,
                    states["fico_bucket"].iat[row],
                    features,
                )
                for row in range(len(states))
            ]
        )
        current_rate = np.clip(model.predict(design), 0.0, 1.0)
        for row in range(len(states)):
            records.append(
                {
                    "vintage": states["vintage"].iat[row],
                    "fico_bucket": states["fico_bucket"].iat[row],
                    "h": horizon,
                    "month": month,
                    "predicted_rate": current_rate[row],
                    "weight_2019m12": states["weight_2019m12"].iat[row],
                }
            )
    return pd.DataFrame(records)


def aggregate_forecast(cohort_forecasts: pd.DataFrame) -> pd.Series:
    return (
        cohort_forecasts.groupby("h")
        .apply(
            lambda group: np.average(
                group["predicted_rate"], weights=group["weight_2019m12"]
            ),
            include_groups=False,
        )
        .rename("mu_recursive")
    )


def main() -> None:
    log("04 Recursive rollout | December 2019 state | realized 2020 UNRATE")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_spec, model, panel, unemployment = load_inputs()
    state_month = pd.Timestamp(STATE_MONTH)
    state = panel[
        (panel["month"] == state_month) & (panel["n_active"] >= MIN_ACTIVE)
    ].copy()
    if len(state) != EXPECTED_STARTING_COHORTS:
        raise ValueError(
            f"Expected {EXPECTED_STARTING_COHORTS} December 2019 cohorts; "
            f"found {len(state)}."
        )

    states = pd.DataFrame(
        {
            "vintage": state["vintage"].to_numpy(),
            "fico_bucket": state["fico_bucket"].to_numpy(),
            "initial_rate": state["default_rate"].to_numpy(),
            "weight_2019m12": state["sum_upb"].to_numpy(),
        }
    ).reset_index(drop=True)

    unemployment_path = [
        (
            state_month + pd.DateOffset(months=step),
            float(unemployment.get(state_month + pd.DateOffset(months=step), np.nan)),
        )
        for step in range(1, HORIZON + 1)
    ]
    if any(np.isnan(value) for _, value in unemployment_path):
        sys.exit("The realized 2020 UNRATE path is incomplete.")

    cohort_forecasts = recursive_rollout(
        model, states, unemployment_path, feature_spec["features"]
    )
    cohort_forecasts.to_parquet(
        OUT_DIR / "rollout_by_cohort.parquet", index=False
    )
    portfolio_forecast = aggregate_forecast(cohort_forecasts)

    result = portfolio_forecast.to_frame()
    result.index.name = "h"
    result["month"] = [month for month, _ in unemployment_path]
    result["unrate"] = [value for _, value in unemployment_path]
    result.to_csv(OUT_DIR / "mu_hat.csv")

    summary = {
        "state_month": STATE_MONTH,
        "horizon": HORIZON,
        "n_cohorts": int(len(states)),
        "weight_definition": "December 2019 cohort current UPB",
        "unrate_path": [value for _, value in unemployment_path],
        "mu_recursive": {
            str(h): float(value) for h, value in portfolio_forecast.items()
        },
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    log(f"Outputs written to {OUT_DIR.relative_to(BASE)}")


if __name__ == "__main__":
    main()
