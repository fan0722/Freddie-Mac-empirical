# -*- coding: utf-8 -*-
"""
06 Calibration of the confounding envelope.

This script estimates the treatment-side NFCI calibration and two outcome-side
calibrations reported in the paper:

1. portfolio AR(1) residuals regressed on lagged standardized NFCI;
2. portfolio prediction residuals regressed on lagged standardized NFCI.

It then evaluates the closed-form confounding envelope for both outcome-side
specifications along the realized 2020 unemployment path.

Inputs
------
macro/NFCI.csv
macro/UNRATE.csv
processed/02_cohort/cohort_month_panel.parquet
processed/03_learner/beta_ols.json
processed/03_learner/feature_spec.json
processed/03_learner/model_selected.joblib
processed/03_learner/train_sample.parquet

Outputs
-------
processed/06_calibration/calibration_params.json
processed/06_calibration/ch.csv
processed/06_calibration/summary.json

Run
---
python 06_calibrate_ch.py
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
OUT_DIR = BASE / "processed" / "06_calibration"
LEARNER_DIR = BASE / "processed" / "03_learner"
NFCI_CSV = BASE / "macro" / "NFCI.csv"
UNRATE_CSV = BASE / "macro" / "UNRATE.csv"
PANEL_PATH = BASE / "processed" / "02_cohort" / "cohort_month_panel.parquet"

CALIBRATION_START = "1990-01"
CALIBRATION_END = "2019-12"
OUTCOME_START = "1999-01"
STATE_MONTH = "2019-12"
HORIZON = 12
MIN_ACTIVE = 100


def read_series(path: Path, value_column: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    date_column = "observation_date" if "observation_date" in data.columns else "DATE"
    data["date"] = pd.to_datetime(data[date_column])
    return data[["date", value_column]].dropna()


def fit_ar1(values: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Fit x[t+1] = intercept + persistence*x[t] + error."""
    outcome = values[1:]
    lag = values[:-1]
    design = np.column_stack([np.ones_like(lag), lag])
    coefficients, *_ = np.linalg.lstsq(design, outcome, rcond=None)
    residuals = outcome - design @ coefficients
    return (
        float(coefficients[0]),
        float(coefficients[1]),
        float(residuals.std(ddof=2)),
        residuals,
    )


def regression_stats(outcome: np.ndarray, regressor: np.ndarray) -> dict:
    """Return coefficient and HC1 inference for a regression with an intercept."""
    import statsmodels.api as sm

    model = sm.OLS(outcome, sm.add_constant(regressor)).fit(cov_type="HC1")
    return {
        "est": float(model.params[1]),
        "se": float(model.bse[1]),
        "t": float(model.tvalues[1]),
        "n": int(model.nobs),
    }


def load_panel() -> pd.DataFrame:
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
    return panel


def main() -> None:
    log("06 Calibration | two outcome-side gamma_Y specifications")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Monthly standardized NFCI and its AR(1) dynamics.
    nfci_weekly = read_series(NFCI_CSV, "NFCI")
    nfci_monthly = (
        nfci_weekly.set_index("date")["NFCI"].resample("ME").last().dropna()
    )
    nfci_monthly.index = nfci_monthly.index.to_period("M").to_timestamp()
    nfci_monthly = nfci_monthly.loc[CALIBRATION_START:CALIBRATION_END]
    nfci_standardized = (
        nfci_monthly - nfci_monthly.mean()
    ) / nfci_monthly.std(ddof=1)
    _, phi_u, sigma_nu, _ = fit_ar1(nfci_standardized.to_numpy())

    # UNRATE dynamics and treatment-side calibration gamma_A.
    unemployment = read_series(UNRATE_CSV, "UNRATE")
    unemployment["month"] = unemployment["date"].dt.to_period("M").dt.to_timestamp()
    unemployment_monthly = (
        unemployment.set_index("month")["UNRATE"]
        .astype(float)
        .loc[CALIBRATION_START:CALIBRATION_END]
    )
    intercept_a, phi_a, sigma_eta, residual_a = fit_ar1(
        unemployment_monthly.to_numpy()
    )
    residual_a = pd.Series(residual_a, index=unemployment_monthly.index[1:])
    lagged_nfci_a = nfci_standardized.reindex(
        residual_a.index - pd.DateOffset(months=1)
    )
    valid_a = lagged_nfci_a.notna().to_numpy()
    gamma_a = regression_stats(
        residual_a.to_numpy()[valid_a], lagged_nfci_a.to_numpy()[valid_a]
    )

    panel = load_panel()
    analysis_panel = panel[
        (panel["n_active"] >= MIN_ACTIVE)
        & (panel["month"] <= pd.Timestamp(CALIBRATION_END))
    ].copy()

    # Specification 1: portfolio AR(1) residuals.
    portfolio_rate = (
        analysis_panel.groupby("month")
        .apply(
            lambda group: np.average(
                group["default_rate"], weights=group["sum_upb"]
            ),
            include_groups=False,
        )
        .loc[OUTCOME_START:CALIBRATION_END]
    )
    _, portfolio_persistence, _, residual_y_ar1 = fit_ar1(
        portfolio_rate.to_numpy()
    )
    residual_y_ar1 = pd.Series(
        residual_y_ar1, index=portfolio_rate.index[1:]
    )
    lagged_nfci_y1 = nfci_standardized.reindex(
        residual_y_ar1.index - pd.DateOffset(months=1)
    )
    valid_y1 = lagged_nfci_y1.notna().to_numpy()
    gamma_y_ar1 = regression_stats(
        residual_y_ar1.to_numpy()[valid_y1],
        lagged_nfci_y1.to_numpy()[valid_y1],
    )
    gamma_y_ar1["spec"] = "portfolio AR(1) residuals on lagged standardized NFCI"

    # Specification 2: portfolio prediction residuals.
    with open(LEARNER_DIR / "feature_spec.json", encoding="utf-8") as file:
        feature_spec = json.load(file)
    selected_model = joblib.load(LEARNER_DIR / "model_selected.joblib")
    training_sample = pd.read_parquet(LEARNER_DIR / "train_sample.parquet")
    training_sample["month"] = pd.to_datetime(training_sample["month"])
    predictions = np.clip(
        selected_model.predict(training_sample[feature_spec["features"]].to_numpy()),
        0,
        1,
    )
    training_sample["prediction_residual"] = (
        training_sample["y_next"].to_numpy() - predictions
    )
    residual_y_prediction = training_sample.groupby("month").apply(
        lambda group: np.average(
            group["prediction_residual"], weights=group["sum_upb"]
        ),
        include_groups=False,
    )
    lagged_nfci_y2 = nfci_standardized.reindex(
        residual_y_prediction.index - pd.DateOffset(months=1)
    )
    valid_y2 = lagged_nfci_y2.notna().to_numpy()
    gamma_y_prediction = regression_stats(
        residual_y_prediction.to_numpy()[valid_y2],
        lagged_nfci_y2.to_numpy()[valid_y2],
    )
    gamma_y_prediction["spec"] = (
        "portfolio prediction residuals on lagged standardized NFCI"
    )

    with open(LEARNER_DIR / "beta_ols.json", encoding="utf-8") as file:
        beta = float(json.load(file)["beta_lag_y"])

    # Closed-form confounding envelopes along the realized 2020 path.
    state_month = pd.Timestamp(STATE_MONTH)
    unemployment_full = unemployment.set_index("month")["UNRATE"].astype(float)
    stress_path = [
        float(unemployment_full.get(state_month + pd.DateOffset(months=step)))
        for step in range(HORIZON + 1)
    ]
    if any(pd.isna(value) for value in stress_path):
        sys.exit("The realized 2020 UNRATE path is incomplete.")

    macro_residuals = np.array(
        [
            stress_path[step]
            - (intercept_a + phi_a * stress_path[step - 1])
            for step in range(1, HORIZON + 1)
        ]
    )
    latent_variance = sigma_nu**2 / (1 - phi_u**2)
    indices = np.arange(HORIZON)
    covariance = latent_variance * phi_u ** np.abs(
        indices[:, None] - indices[None, :]
    )

    def envelope(gamma_y: float) -> tuple[dict[int, float], np.ndarray]:
        values = {}
        for horizon in range(1, HORIZON + 1):
            sigma = covariance[:horizon, :horizon]
            conditional_u = gamma_a["est"] * sigma @ np.linalg.solve(
                gamma_a["est"] ** 2 * sigma
                + sigma_eta**2 * np.eye(horizon),
                macro_residuals[:horizon],
            )
            weights = beta ** (horizon - 1 - np.arange(horizon))
            values[horizon] = abs(gamma_y) * abs(float(weights @ conditional_u))

        conditional_u_full = gamma_a["est"] * covariance @ np.linalg.solve(
            gamma_a["est"] ** 2 * covariance
            + sigma_eta**2 * np.eye(HORIZON),
            macro_residuals,
        )
        return values, conditional_u_full

    ch_ar1, conditional_u = envelope(gamma_y_ar1["est"])
    ch_prediction, _ = envelope(gamma_y_prediction["est"])

    for horizon in range(1, HORIZON + 1):
        log(
            f"h={horizon:>2}: "
            f"c_h AR(1)={ch_ar1[horizon]:.6f}; "
            f"c_h prediction={ch_prediction[horizon]:.6f}"
        )

    ch_table = pd.DataFrame(
        {
            "h": range(1, HORIZON + 1),
            "macro_residual": macro_residuals,
            "conditional_u_h12": conditional_u,
            "ch_ar1_residual": [ch_ar1[h] for h in range(1, HORIZON + 1)],
            "ch_prediction_residual": [
                ch_prediction[h] for h in range(1, HORIZON + 1)
            ],
        }
    )
    ch_table.to_csv(OUT_DIR / "ch.csv", index=False)

    parameters = {
        "nfci": {
            "phi_U": phi_u,
            "sigma_nu": sigma_nu,
            "n": int(len(nfci_standardized)),
        },
        "unrate": {
            "const": intercept_a,
            "phi_A": phi_a,
            "sigma_eta": sigma_eta,
            "window": f"{CALIBRATION_START}..{CALIBRATION_END}",
        },
        "gamma_A": gamma_a,
        "gamma_Y_ar1_residual": gamma_y_ar1,
        "gamma_Y_prediction_residual": gamma_y_prediction,
        "beta": beta,
        "portfolio_persistence": portfolio_persistence,
        "stress_path": stress_path[1:],
        "state_month": STATE_MONTH,
    }
    with open(OUT_DIR / "calibration_params.json", "w", encoding="utf-8") as file:
        json.dump(parameters, file, indent=2)

    summary = {
        "ch_ar1_residual": {str(h): ch_ar1[h] for h in ch_ar1},
        "ch_prediction_residual": {
            str(h): ch_prediction[h] for h in ch_prediction
        },
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    log(f"Outputs written to {OUT_DIR.relative_to(BASE)}")


if __name__ == "__main__":
    main()
