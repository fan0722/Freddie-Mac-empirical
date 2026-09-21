# -*- coding: utf-8 -*-
"""
07 Final tables and figures for the Freddie Mac application.

The reported realized outcome is the borrower-assistance-adjusted 90+DPD rate.
The portfolio contains the 62 cohorts active in December 2019, and their
December 2019 current-UPB weights remain fixed at every 2020 horizon.

The final figure presents the two outcome-side gamma_Y specifications produced
by 06_calibrate_ch.py. When complete pipeline outputs are unavailable, this
script uses the aggregate files distributed in replication_inputs/.

Outputs
-------
processed/07_results/fig_freddie_intervals.png
processed/07_results/fig_freddie_intervals.pdf
processed/07_results/table_coverage.csv
processed/07_results/calibration_summary.txt
processed/07_results/diagnostics.csv
processed/07_results/summary.json

Run
---
python 07_results.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


BASE = Path(os.environ.get("FREDDIE_BASE", Path(__file__).resolve().parent)).expanduser()
PROCESSED = BASE / "processed"
OUT_DIR = PROCESSED / "07_results"
REPLICATION_INPUTS = Path(__file__).resolve().parent / "replication_inputs"
PANEL_PATH = PROCESSED / "02_cohort" / "cohort_month_panel.parquet"

STATE_MONTH = "2019-12"
HORIZON = 12
MIN_ACTIVE = 100
EXPECTED_STARTING_COHORTS = 62
ABSTENTION_THRESHOLD = 2.0


def input_path(processed_relative_path: str, filename: str) -> Path:
    """Prefer a newly generated pipeline output, then use the distributed aggregate."""
    candidate = PROCESSED / processed_relative_path
    if candidate.exists():
        return candidate
    fallback = REPLICATION_INPUTS / filename
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing required input: {candidate} or {fallback}")


def load_inputs():
    mu = pd.read_csv(
        input_path("04_rollout/mu_hat.csv", "mu_hat.csv"), index_col="h"
    )
    intervals = pd.read_csv(
        input_path("05_conformal/intervals.csv", "intervals.csv"),
        index_col="h",
    )
    ch = pd.read_csv(
        input_path("06_calibration/ch.csv", "ch.csv")
    ).set_index("h")
    with open(
        input_path("06_calibration/calibration_params.json", "calibration_params.json"),
        encoding="utf-8",
    ) as file:
        parameters = json.load(file)
    scores = pd.read_csv(
        input_path("05_conformal/scores.csv", "scores.csv")
    )
    return mu, intervals, ch, parameters, scores


def adjusted_realized_series() -> pd.Series:
    """Construct the 2020 adjusted rate using fixed December 2019 cohorts and weights."""
    if not PANEL_PATH.exists():
        fallback = REPLICATION_INPUTS / "realized_excl_fb_2020.csv"
        if not fallback.exists():
            sys.exit(
                "Missing cohort panel and replication_inputs/realized_excl_fb_2020.csv"
            )
        data = pd.read_csv(fallback)
        if not data["n_starting_cohorts"].eq(EXPECTED_STARTING_COHORTS).all():
            raise ValueError("The distributed realized series does not use 62 cohorts.")
        return data.set_index("h")["realized_excl_fb"]

    panel = pd.read_parquet(PANEL_PATH)
    if "month" in panel.columns:
        panel["month"] = pd.to_datetime(panel["month"])
    else:
        panel["month"] = pd.to_datetime(
            panel["monthly_reporting_period"], format="%Y%m"
        )
    panel["month"] = panel["month"].dt.to_period("M").dt.to_timestamp()

    state_month = pd.Timestamp(STATE_MONTH)
    starting = panel[
        (panel["month"] == state_month) & (panel["n_active"] >= MIN_ACTIVE)
    ][["vintage", "fico_bucket", "sum_upb"]].copy()
    if len(starting) != EXPECTED_STARTING_COHORTS:
        raise ValueError(
            f"Expected {EXPECTED_STARTING_COHORTS} December 2019 cohorts; "
            f"found {len(starting)}."
        )

    realized = {}
    for horizon in range(1, HORIZON + 1):
        month = state_month + pd.DateOffset(months=horizon)
        current = panel[panel["month"] == month].merge(
            starting,
            on=["vintage", "fico_bucket"],
            how="inner",
            suffixes=("", "_2019m12"),
        )
        valid = (
            current["default_rate_excl_fb"].notna()
            & current["sum_upb_2019m12"].notna()
            & (current["sum_upb_2019m12"] > 0)
        )
        current = current.loc[valid]
        if current.empty:
            raise ValueError(f"No valid observations for {month:%Y-%m}.")
        realized[horizon] = float(
            np.average(
                current["default_rate_excl_fb"],
                weights=current["sum_upb_2019m12"],
            )
        )
    return pd.Series(realized, name="realized_excl_fb")


def coverage_row(method: str, halfwidth: pd.Series, forecast: pd.Series, realized: pd.Series):
    covered = {
        horizon: abs(forecast[horizon] - realized[horizon]) <= halfwidth[horizon]
        for horizon in range(1, HORIZON + 1)
        if not (pd.isna(realized[horizon]) or pd.isna(halfwidth[horizon]))
    }
    horizons = sorted(covered)
    short = [h for h in horizons if h <= 3]
    long = [h for h in horizons if h >= 4]
    return {
        "method": method,
        "coverage_h1_12": f"{sum(covered[h] for h in horizons)}/{len(horizons)}",
        "coverage_h1_3": f"{sum(covered[h] for h in short)}/{len(short)}",
        "coverage_h4_12": f"{sum(covered[h] for h in long)}/{len(long)}",
        "mean_halfwidth": float(np.mean([halfwidth[h] for h in horizons])),
        "covered_h1_12": "".join("1" if covered[h] else "0" for h in horizons),
    }


def main() -> None:
    log("07 Final results | borrower-assistance-adjusted outcome")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mu, intervals, ch, parameters, scores = load_inputs()
    forecast = mu["mu_recursive"]
    realized = adjusted_realized_series()
    q_weighted = intervals["q_w"]
    ch_ar1 = ch["ch_ar1_residual"]
    ch_prediction = ch["ch_prediction_residual"]

    halfwidths = {
        "Linear ST": intervals["linst_hw"],
        "EnbPI": intervals["enbpi_hw"],
        "Ours (cal)": q_weighted,
        "Ours (full, AR(1) residual)": q_weighted + ch_ar1,
        "Ours (full, prediction residual)": q_weighted + ch_prediction,
    }
    coverage = pd.DataFrame(
        [
            coverage_row(method, width, forecast, realized)
            for method, width in halfwidths.items()
        ]
    )
    coverage.to_csv(OUT_DIR / "table_coverage.csv", index=False)

    for row in coverage.itertuples(index=False):
        log(
            f"{row.method:<36} | h1-12 {row.coverage_h1_12:>5} | "
            f"h1-3 {row.coverage_h1_3} | h4-12 {row.coverage_h4_12:>3} | "
            f"mean halfwidth {row.mean_halfwidth:.5f}"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = np.arange(1, HORIZON + 1)
    forecast_values = forecast.reindex(horizons).to_numpy()
    realized_values = realized.reindex(horizons).to_numpy()
    q_values = q_weighted.reindex(horizons).to_numpy()
    abstention = intervals.index[intervals["B_eff"] <= ABSTENTION_THRESHOLD]
    abstention_start = None if len(abstention) == 0 else int(abstention.min())

    def draw_panel(axis, confounding_width, title):
        outer = q_values + confounding_width.reindex(horizons).to_numpy()
        axis.fill_between(
            horizons,
            forecast_values - outer,
            forecast_values + outer,
            color="#c6dbef",
            alpha=0.9,
            label=r"Ours (full): $\pm(q^w_h+c_h)$",
        )
        axis.fill_between(
            horizons,
            forecast_values - q_values,
            forecast_values + q_values,
            color="#6baed6",
            alpha=0.9,
            label=r"Ours (cal): $\pm q^w_h$",
        )
        axis.plot(
            horizons,
            forecast_values,
            "-o",
            color="#08519c",
            linewidth=1.8,
            markersize=4,
            label=r"$\hat\mu_h$",
        )
        axis.plot(
            horizons,
            realized_values,
            "--^",
            color="#ff7f0e",
            linewidth=1.5,
            markersize=4,
            label="Realized (excl. forbearance)",
        )
        if abstention_start is not None:
            axis.axvspan(
                abstention_start - 0.5,
                HORIZON + 0.5,
                color="grey",
                alpha=0.12,
            )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Stress horizon $h$ (months, 2020)")
        axis.set_xticks(horizons)
        axis.grid(alpha=0.25)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    draw_panel(
        axes[0],
        ch_ar1,
        r"Spec 1 (main): $\hat\gamma_Y$ from AR(1) residuals",
    )
    draw_panel(
        axes[1],
        ch_prediction,
        r"Spec 2: $\hat\gamma_Y$ from prediction residuals",
    )
    axes[0].set_ylabel("Portfolio 90+DPD rate")
    axes[0].legend(fontsize=7.5, loc="upper left", framealpha=0.92)
    figure.tight_layout()
    figure.savefig(OUT_DIR / "fig_freddie_intervals.png", dpi=300)
    figure.savefig(OUT_DIR / "fig_freddie_intervals.pdf")
    plt.close(figure)

    gamma_a = parameters["gamma_A"]
    gamma_y_ar1 = parameters["gamma_Y_ar1_residual"]
    gamma_y_prediction = parameters["gamma_Y_prediction_residual"]
    calibration_text = (
        f"Calibration window: {parameters['unrate']['window']}\n"
        f"NFCI AR(1): phi_U={parameters['nfci']['phi_U']:.6f}, "
        f"sigma_nu={parameters['nfci']['sigma_nu']:.6f}\n"
        f"UNRATE AR(1): phi_A={parameters['unrate']['phi_A']:.6f}, "
        f"sigma_eta={parameters['unrate']['sigma_eta']:.6f}\n"
        f"gamma_A={gamma_a['est']:.8f}, HC1 SE={gamma_a['se']:.8f}, "
        f"t={gamma_a['t']:.4f}, n={gamma_a['n']}\n"
        f"gamma_Y, AR(1)-residual specification={gamma_y_ar1['est']:.10f}, "
        f"HC1 SE={gamma_y_ar1['se']:.10f}, t={gamma_y_ar1['t']:.4f}, "
        f"n={gamma_y_ar1['n']}\n"
        f"gamma_Y, prediction-residual specification="
        f"{gamma_y_prediction['est']:.10f}, "
        f"HC1 SE={gamma_y_prediction['se']:.10f}, "
        f"t={gamma_y_prediction['t']:.4f}, n={gamma_y_prediction['n']}\n"
        f"Lagged-outcome coefficient beta={parameters['beta']:.8f}\n"
    )
    (OUT_DIR / "calibration_summary.txt").write_text(
        calibration_text, encoding="utf-8"
    )

    placebo = (
        scores.groupby("h")["S"]
        .agg(["mean", "median", "max"])
        .rename(columns=lambda column: f"placebo_S_{column}")
    )
    diagnostics = intervals[
        ["B", "R_weight", "B_eff", "W_max_share"]
    ].join(placebo)
    diagnostics["abstain"] = diagnostics["B_eff"] <= ABSTENTION_THRESHOLD
    diagnostics.to_csv(OUT_DIR / "diagnostics.csv")

    summary = {
        "realized_outcome": "90+DPD excluding borrower-assistance and disaster-coded loan-months from the numerator",
        "starting_cohorts": EXPECTED_STARTING_COHORTS,
        "weight_month": STATE_MONTH,
        "coverage_table": coverage.to_dict(orient="records"),
        "abstention_from_h": (
            int(diagnostics.index[diagnostics["abstain"]].min())
            if diagnostics["abstain"].any()
            else None
        ),
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    log(f"Outputs written to {OUT_DIR.relative_to(BASE)}")


if __name__ == "__main__":
    main()
