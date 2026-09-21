# -*- coding: utf-8 -*-
"""
03 Select and fit the base prediction model.

The temporal design uses 1999–2014 for training, 2015–2019 for conformal
calibration, and 2020 for stress evaluation. Candidate learners are compared
with rolling validation folds and active-loan weights. The selected model,
feature specification, supervised sample, OLS benchmark, and unemployment
AR(1) parameters are saved for the subsequent stages.

Outputs
-------
processed/03_learner/cv_results.csv
processed/03_learner/selected_model.json
processed/03_learner/model_selected.joblib
processed/03_learner/model_ols.joblib
processed/03_learner/beta_ols.json
processed/03_learner/unrate_ar1.json
processed/03_learner/feature_spec.json
processed/03_learner/train_sample.parquet

Run
---
python 03_learner.py
"""

import os
from pathlib import Path
from datetime import datetime
import json
import sys
import numpy as np
import pandas as pd
import joblib


def log(msg: str):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# ============================================================
# 配置区
# ============================================================
BASE = Path(os.environ.get("FREDDIE_BASE", Path(__file__).resolve().parent)).expanduser()
OUT_DIR = BASE / "processed" / "03_learner"
UNRATE_CSV = BASE / "macro" / "UNRATE.csv"

PANEL_PATH = BASE / "processed" / "02_cohort" / "cohort_month_panel.parquet"

TRAIN_START, TRAIN_END = "1999-01", "2014-12"   # 训练段 (CV与最终拟合都限于此)
FEATURE_END = "2019-12"                          # 监督样本构造到此 (05校准复用)
CV_VAL_YEARS = list(range(2006, 2015))           # 滚动CV: 验证年2006..2014, 训练=1999..验证年前一年
MIN_ACTIVE = 100                                 # 组×月活跃贷款数下限 (剔除小样本噪声组)
SEED = 42


# ============================================================
# 特征工程: 监督样本 (y_next ~ y_lag + cohort特征 + unrate_next)
# ============================================================
def build_supervised_sample() -> tuple[pd.DataFrame, list]:
    if not PANEL_PATH.exists():
        sys.exit(f"Missing output from step 02: {PANEL_PATH}")
    log(f"Reading panel: {PANEL_PATH.relative_to(BASE)}")
    df = pd.read_parquet(PANEL_PATH)

    un = pd.read_csv(UNRATE_CSV)
    date_col = "observation_date" if "observation_date" in un.columns else "DATE"
    un["month"] = pd.to_datetime(un[date_col])
    un_map = un.set_index("month")["UNRATE"]

    df["month"] = pd.to_datetime(df["month"]) if "month" in df.columns else \
        pd.to_datetime(df["monthly_reporting_period"], format="%Y%m")
    df = df[df["n_active"] >= MIN_ACTIVE].copy()
    df = df.sort_values(["vintage", "fico_bucket", "month"])

    g = df.groupby(["vintage", "fico_bucket"], observed=True)
    df["y_lag"] = g["default_rate"].shift(1)
    df["month_lag"] = g["month"].shift(1)
    # 保证滞后恰好是上一个月 (组内断月的行剔除)
    ok = (df["month"] - df["month_lag"]).dt.days.between(28, 31)
    df = df[ok & df["y_lag"].notna()].copy()

    df["unrate_next"] = df["month"].map(un_map)          # 结果月t+1的宏观 = A_{t+1}
    df = df[df["unrate_next"].notna()].copy()
    df["vintage_age"] = ((df["month"].dt.year - df["vintage"]) * 12
                         + df["month"].dt.month - 1).clip(lower=0)

    fico_d = pd.get_dummies(df["fico_bucket"], prefix="fico", dtype=float)
    df = pd.concat([df, fico_d], axis=1)

    features = ["y_lag", "unrate_next", "vintage_age"] + sorted(fico_d.columns)
    df = df.rename(columns={"default_rate": "y_next"})
    keep = ["vintage", "fico_bucket", "month", "y_next", "n_active", "sum_upb"] + features
    df = df[keep][df["month"] <= pd.Timestamp(FEATURE_END) + pd.offsets.MonthEnd(0)]
    log(f"监督样本: {len(df):,} 行  特征: {features}")
    return df.reset_index(drop=True), features


# ============================================================
# 五学习器
# ============================================================
def make_models():
    from sklearn.linear_model import LinearRegression, LassoCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from lightgbm import LGBMRegressor
    return {
        "OLS": LinearRegression(),
        "LASSO": Pipeline([("sc", StandardScaler()),
                           ("m", LassoCV(cv=5, random_state=SEED, max_iter=50000))]),
        "LightGBM": LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                  num_leaves=31, random_state=SEED, verbose=-1),
        "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                              random_state=SEED, n_jobs=-1),
        "MLP": Pipeline([("sc", StandardScaler()),
                         ("m", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=3000,
                                            early_stopping=True, random_state=SEED))]),
    }


def fit_one(model, X, y, w):
    """Fit with sample weights when supported by the estimator."""
    try:
        if hasattr(model, "steps"):                      # Pipeline
            last = model.steps[-1][0]
            model.fit(X, y, **{f"{last}__sample_weight": w})
        else:
            model.fit(X, y, sample_weight=w)
    except (TypeError, ValueError):
        model.fit(X, y)
    return model


def wrmse(y, yhat, w):
    return float(np.sqrt(np.average((y - yhat) ** 2, weights=w)))


# ============================================================
# Rolling cross-validation within the training period
# ============================================================
def cross_validate_models(df: pd.DataFrame, features: list) -> pd.DataFrame:
    rows = []
    for val_year in CV_VAL_YEARS:
        tr = df[(df["month"] >= TRAIN_START)
                & (df["month"].dt.year <= val_year - 1)]
        va = df[df["month"].dt.year == val_year]
        if len(tr) < 200 or len(va) < 50:
            log(f"Skipping validation year {val_year}: insufficient observations (train={len(tr)}, validation={len(va)})")
            continue
        Xtr, ytr, wtr = tr[features].values, tr["y_next"].values, tr["n_active"].values
        Xva, yva, wva = va[features].values, va["y_next"].values, va["n_active"].values
        for name, model in make_models().items():
            m = fit_one(model, Xtr, ytr, wtr)
            score = wrmse(yva, m.predict(Xva), wva)
            rows.append({"val_year": val_year, "model": name,
                         "wrmse": score, "n_train": len(tr), "n_val": len(va)})
        log(f"[validation {val_year}] " + "  ".join(
            f"{r['model']}={r['wrmse']:.5f}" for r in rows if r["val_year"] == val_year))
    return pd.DataFrame(rows)


# ============================================================
# UNRATE AR(1) (训练段, 05密度权重/06标定共用)
# ============================================================
def fit_unrate_ar1() -> dict:
    un = pd.read_csv(UNRATE_CSV)
    date_col = "observation_date" if "observation_date" in un.columns else "DATE"
    un["month"] = pd.to_datetime(un[date_col])
    un = un[(un["month"] >= TRAIN_START) &
            (un["month"] <= pd.Timestamp(TRAIN_END) + pd.offsets.MonthEnd(0))]
    a = un.sort_values("month")["UNRATE"].astype(float).values
    y, x = a[1:], a[:-1]
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    params = {"const": float(coef[0]), "phi": float(coef[1]),
              "sigma_eta": float(resid.std(ddof=2)),
              "window": f"{TRAIN_START}..{TRAIN_END}", "n_obs": int(len(y))}
    log(f"UNRATE AR(1): const={params['const']:.4f}  phi={params['phi']:.4f}  "
        f"sigma_eta={params['sigma_eta']:.4f}  (n={params['n_obs']})")
    return params


# ============================================================
# 主流程
# ============================================================
def main():
    np.random.seed(SEED)
    log("=" * 64)
    log("03 Learner | rolling model selection and final fit")
    log(f"train {TRAIN_START}..{TRAIN_END} | CV验证年 {CV_VAL_YEARS[0]}..{CV_VAL_YEARS[-1]}"
        f" | MIN_ACTIVE={MIN_ACTIVE}")
    log("=" * 64)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, features = build_supervised_sample()
    df.to_parquet(OUT_DIR / "train_sample.parquet", index=False)
    json.dump({"features": features, "min_active": MIN_ACTIVE,
               "train": [TRAIN_START, TRAIN_END], "feature_end": FEATURE_END},
              open(OUT_DIR / "feature_spec.json", "w"), indent=2)

    # ---- Rolling model comparison ----
    cv = cross_validate_models(df, features)
    cv.to_csv(OUT_DIR / "cv_results.csv", index=False)
    mean_scores = cv.groupby("model")["wrmse"].mean().sort_values()
    winner = mean_scores.index[0]
    log("Mean validation wRMSE: " + "  ".join(f"{k}={v:.5f}" for k, v in mean_scores.items()))
    log(f"Selected model: {winner}")
    json.dump({"selected": winner,
               "mean_wrmse": {k: float(v) for k, v in mean_scores.items()},
               "criterion": "one-step weighted RMSE, rolling-origin CV within training window"},
              open(OUT_DIR / "selected_model.json", "w"), indent=2)

    # ---- Refit the selected model and OLS benchmark on the full training period ----
    tr = df[(df["month"] >= TRAIN_START)
            & (df["month"] <= pd.Timestamp(TRAIN_END) + pd.offsets.MonthEnd(0))]
    Xtr, ytr, wtr = tr[features].values, tr["y_next"].values, tr["n_active"].values

    models = make_models()
    m_sel = fit_one(models[winner], Xtr, ytr, wtr)
    joblib.dump(m_sel, OUT_DIR / "model_selected.joblib")

    m_ols = fit_one(models["OLS"], Xtr, ytr, wtr) if winner != "OLS" else m_sel
    joblib.dump(m_ols, OUT_DIR / "model_ols.joblib")

    beta = float(m_ols.coef_[features.index("y_lag")])
    b2 = float(m_ols.coef_[features.index("unrate_next")])
    json.dump({"beta_lag_y": beta, "coef_unrate": b2,
               "note": "OLS on full training window, n_active-weighted; "
                       "beta_lag_y feeds Corollary 1 in 06"},
              open(OUT_DIR / "beta_ols.json", "w"), indent=2)
    log(f"OLS coefficients: beta_lag_y={beta:.4f}; coef_unrate={b2:.5f}")

    # ---- UNRATE AR(1) ----
    json.dump(fit_unrate_ar1(), open(OUT_DIR / "unrate_ar1.json", "w"), indent=2)

    log("=" * 64)
    log(f"完成. 产出目录: {OUT_DIR.relative_to(BASE)}")


if __name__ == "__main__":
    main()
