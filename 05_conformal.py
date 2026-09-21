# -*- coding: utf-8 -*-
"""
05 Weighted conformal calibration and benchmark intervals.

The calibration period is January 2015 through December 2019. The reported
implementation uses a one-month rolling-origin gap and alpha=0.10. It produces
the weighted conformal interval, EnbPI and Linear ST benchmarks, and the
extrapolation diagnostics used in the final results.

Outputs
-------
processed/05_conformal/intervals.csv
processed/05_conformal/scores.csv
processed/05_conformal/summary.json

Run
---
python 05_conformal.py
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
L03 = BASE / "processed" / "03_learner"
OUT_DIR = BASE / "processed" / "05_conformal"
UNRATE_CSV = BASE / "macro" / "UNRATE.csv"
PANEL_PATH = BASE / "processed" / "02_cohort" / "cohort_month_panel.parquet"

CAL_START, CAL_END = "2015-01", "2019-12"
STRESS_STATE_MONTH = "2019-12"       # 压力路径的状态起点 (a^S_0 = 该月UNRATE)
H = 12
ALPHA = 0.10
ROLLING_GAP = 1
MIN_ACTIVE = 100


# ============================================================
# 载入
# ============================================================
def load_all():
    spec = json.load(open(L03 / "feature_spec.json"))
    model = joblib.load(L03 / "model_selected.joblib")
    ar1 = json.load(open(L03 / "unrate_ar1.json"))
    beta = json.load(open(L03 / "beta_ols.json"))["beta_lag_y"]
    if not PANEL_PATH.exists():
        sys.exit("找不到cohort面板")
    panel = pd.read_parquet(PANEL_PATH)
    panel["month"] = (pd.to_datetime(panel["month"]) if "month" in panel.columns
                      else pd.to_datetime(panel["monthly_reporting_period"], format="%Y%m"))
    panel["month"] = panel["month"].dt.to_period("M").dt.to_timestamp()
    panel = panel[panel["n_active"] >= MIN_ACTIVE]
    un = pd.read_csv(UNRATE_CSV)
    dc = "observation_date" if "observation_date" in un.columns else "DATE"
    un["month"] = pd.to_datetime(un[dc]).dt.to_period("M").dt.to_timestamp()
    un_map = un.set_index("month")["UNRATE"].astype(float)
    return spec, model, ar1, beta, panel, un_map


def feature_row(y_prev, unrate_next, vintage, month_next, fico_bucket, features):
    age = max(0, (month_next.year - vintage) * 12 + month_next.month - 1)
    vals = {"y_lag": y_prev, "unrate_next": unrate_next, "vintage_age": age}
    for f in features:
        if f.startswith("fico_"):
            vals[f] = 1.0 if f == f"fico_{fico_bucket}" else 0.0
    return [vals[f] for f in features]


def rollout_portfolio(model, states, path, features):
    """返回长度=len(path)的组合(UPB加权)预测序列."""
    y = states["y0"].values.copy()
    out = []
    for mth, ur in path:
        X = np.array([feature_row(y[i], ur, states["vintage"].iat[i], mth,
                                  states["fico_bucket"].iat[i], features)
                      for i in range(len(states))])
        y = np.clip(model.predict(X), 0.0, 1.0)
        out.append(float(np.average(y, weights=states["weight"].values)))
    return out


def gauss_logpdf(x, mean, sigma):
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((x - mean) / sigma) ** 2


# ============================================================
# 核心: 一个g值下的完整校准
# ============================================================
def calibrate(g, spec, model, ar1, panel, un_map, stress_path, a_s0):
    features = spec["features"]
    c, phi, sig = ar1["const"], ar1["phi"], ar1["sigma_eta"]
    cal_months = pd.date_range(CAL_START, CAL_END, freq="MS")[::g]

    recs = []
    for t_b in cal_months:
        st = panel[panel["month"] == t_b]
        if len(st) < 5:
            continue
        states = pd.DataFrame({"vintage": st["vintage"].values,
                               "fico_bucket": st["fico_bucket"].values,
                               "y0": st["default_rate"].values,
                               "weight": st["sum_upb"].values})
        # 起点b的可用horizon: 不越过校准期末
        h_max = min(H, (pd.Timestamp(CAL_END).year - t_b.year) * 12
                    + pd.Timestamp(CAL_END).month - t_b.month)
        if h_max < 1:
            continue
        path_b = [(t_b + pd.DateOffset(months=k),
                   float(un_map.get(t_b + pd.DateOffset(months=k), np.nan)))
                  for k in range(1, h_max + 1)]
        if any(np.isnan(v) for _, v in path_b):
            continue
        preds = rollout_portfolio(model, states, path_b, features)

        # 逐h: 误差S_b 与 log权重 (逐期累积)
        J_prev = float(un_map.get(t_b))          # 起点b的宏观状态 a_{b,0}
        aS_prev = a_s0                            # 压力路径状态 a^S_0
        log_w = 0.0
        for h in range(1, h_max + 1):
            mth = t_b + pd.DateOffset(months=h)
            r = panel[panel["month"] == mth].merge(
                st[["vintage", "fico_bucket", "sum_upb"]],
                on=["vintage", "fico_bucket"], suffixes=("", "_w"))
            if len(r) == 0:
                break
            realized = float(np.average(r["default_rate"], weights=r["sum_upb_w"]))
            S = abs(preds[h - 1] - realized)
            a_bj = path_b[h - 1][1]
            aS_j = stress_path[h - 1][1]
            # w_b 的第j项: p(a^S_j | J_{b,j-1}) / p(a_{b,j} | J_{b,j-1})
            log_w += gauss_logpdf(aS_j, c + phi * J_prev, sig) \
                   - gauss_logpdf(a_bj, c + phi * J_prev, sig)
            recs.append({"origin": t_b.strftime("%Y-%m"), "h": h,
                         "S": S, "log_w": log_w})
            J_prev = a_bj
            aS_prev = aS_j
    scores = pd.DataFrame(recs)

    # ---- 逐h: 加权分位数 + 诊断 + 基线 ----
    rows = []
    for h in range(1, H + 1):
        d = scores[scores["h"] == h].copy()
        if len(d) < 3:
            rows.append({"h": h, "B": len(d)})
            continue
        # log-sum-exp稳定化: 权重平移不改变加权分位与诊断比值
        lw = d["log_w"].values - d["log_w"].max()
        w = np.exp(lw)
        W_max, W_sum = w.max(), w.sum()
        order = np.argsort(d["S"].values)
        S_sorted, w_sorted = d["S"].values[order], w[order]
        cum = np.cumsum(w_sorted)
        thresh = (1 - ALPHA) * (W_sum + W_max) - W_max
        idx = np.searchsorted(cum, thresh, side="left")
        q_w = float(S_sorted[min(idx, len(S_sorted) - 1)]) if thresh > 0 else float(S_sorted[0])
        # 若阈值超过全部权重和(极端退化), 取最大误差 (保守)
        if thresh > cum[-1]:
            q_w = float(S_sorted[-1])
        R_weight = float(W_max / (W_sum + W_max))
        B_eff = float(W_sum ** 2 / (w ** 2).sum())
        enbpi = float(np.quantile(d["S"].values, 1 - ALPHA))
        rows.append({"h": h, "B": int(len(d)), "q_w": q_w, "enbpi_hw": enbpi,
                     "R_weight": R_weight, "B_eff": B_eff,
                     "W_max_share": float(W_max / W_sum)})
    return pd.DataFrame(rows).set_index("h"), scores


# ============================================================
# Linear ST 基线: 组合一步残差std按β复合
# ============================================================
def linear_st_halfwidths(model, beta, panel, un_map, spec):
    """训练段内逐月组合层面的一步预测残差 -> std -> β复合."""
    features = spec["features"]
    tr = panel[(panel["month"] >= "1999-02") & (panel["month"] <= "2014-12")]
    resid = []
    for mth, d in tr.groupby("month"):
        prev = panel[panel["month"] == mth - pd.DateOffset(months=1)]
        m = d.merge(prev[["vintage", "fico_bucket", "default_rate", "sum_upb"]],
                    on=["vintage", "fico_bucket"], suffixes=("", "_prev"))
        if len(m) < 5 or pd.isna(un_map.get(mth)):
            continue
        X = np.array([feature_row(m["default_rate_prev"].iat[i], float(un_map[mth]),
                                  m["vintage"].iat[i], mth, m["fico_bucket"].iat[i],
                                  features) for i in range(len(m))])
        pred = float(np.average(np.clip(model.predict(X), 0, 1),
                                weights=m["sum_upb_prev"].values))
        real = float(np.average(m["default_rate"], weights=m["sum_upb_prev"].values))
        resid.append(pred - real)
    sigma_p = float(np.std(resid, ddof=1))
    z = 1.6449
    hw = {h: z * sigma_p * float(np.sqrt(sum(beta ** (2 * j) for j in range(h))))
          for h in range(1, H + 1)}
    return hw, sigma_p


# ============================================================
# 主流程
# ============================================================
def main():
    log("=" * 64)
    log(
        f"05 Conformal | calibration {CAL_START}..{CAL_END} | "
        f"alpha={ALPHA} | gap={ROLLING_GAP}"
    )
    log("=" * 64)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec, model, ar1, beta, panel, un_map = load_all()

    t0 = pd.Timestamp(STRESS_STATE_MONTH)
    a_s0 = float(un_map.get(t0))
    stress_path = [(t0 + pd.DateOffset(months=k),
                    float(un_map.get(t0 + pd.DateOffset(months=k), np.nan)))
                   for k in range(1, H + 1)]
    if any(np.isnan(v) for _, v in stress_path):
        sys.exit("压力路径UNRATE缺失")
    log(f"压力路径 a^S (a^S_0={a_s0:.1f}): " +
        " ".join(f"{v:.1f}" for _, v in stress_path))

    linst_hw, sigma_p = linear_st_halfwidths(model, beta, panel, un_map, spec)
    log(f"Linear ST: 组合一步残差std σ_p={sigma_p:.6f}, β={beta:.4f}")

    tab, scores = calibrate(
        ROLLING_GAP, spec, model, ar1, panel, un_map, stress_path, a_s0
    )
    tab["linst_hw"] = pd.Series(linst_hw)
    tab.to_csv(OUT_DIR / "intervals.csv")
    scores.to_csv(OUT_DIR / "scores.csv", index=False)
    log(" h |  B  |   q_w    | enbpi_hw | linst_hw | R_weight | B_eff | Wmax share")
    for h in range(1, H + 1):
        if h not in tab.index or pd.isna(tab.loc[h].get("q_w", np.nan)):
            log(f"{h:>2} | insufficient data")
            continue
        r = tab.loc[h]
        log(
            f"{h:>2} | {int(r['B']):>3} | {r['q_w']:.5f}  | "
            f"{r['enbpi_hw']:.5f}  | {r['linst_hw']:.5f}  | "
            f"{r['R_weight']:.3f} | {r['B_eff']:>5.1f} | "
            f"{r['W_max_share']:.2f}"
        )
    summary = {
        "intervals": {
            str(h): {
                key: (None if pd.isna(value) else float(value))
                for key, value in tab.loc[h].items()
            }
            for h in tab.index
        },
        "rolling_gap": ROLLING_GAP,
        "sigma_p": sigma_p,
        "alpha": ALPHA,
    }
    json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)
    log("=" * 64)
    log(f"完成. 产出目录: {OUT_DIR.relative_to(BASE)}")


if __name__ == "__main__":
    main()
