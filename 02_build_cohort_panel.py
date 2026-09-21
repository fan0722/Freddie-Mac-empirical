# -*- coding: utf-8 -*-
"""
02 Build the vintage-by-FICO cohort-month panel.

The script constructs the active-loan denominator, recorded 90+DPD outcome,
borrower-assistance-adjusted 90+DPD outcome, fixed cohort identifiers, and
cohort-level balance totals used by the remaining pipeline.

Inputs
------
processed/01_loan_month/loan_month_YYYY.parquet

Outputs
-------
processed/02_cohort/cohort_month_panel.parquet
processed/02_cohort/overall_month_series.csv
processed/02_cohort/cohort_build_summary.txt

Run
---
python 02_build_cohort_panel.py
"""

import os
from pathlib import Path
from datetime import datetime
import sys
import numpy as np
import pandas as pd


def log(msg: str):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# ============================================================
# Configuration
# ============================================================
BASE = Path(os.environ.get("FREDDIE_BASE", Path(__file__).resolve().parent)).expanduser()
LOAN_MONTH_DIR = BASE / "processed" / "01_loan_month"
OUT_DIR = BASE / "processed" / "02_cohort"

YEARS = list(range(1999, 2020))

# ---- 口径1: 月度违约指标 (default indicator) ----
DPD_THRESHOLD = 3        # dlq_status数值 >= 3 即 90+ DPD; 改成 2 则为 60+ DPD
INCLUDE_RA = True        # True: dlq_status == "RA" (REO Acquisition, 已收房) 也计为违约
DROP_UNKNOWN_DLQ = True  # True: dlq_status 缺失/无法解析的行, 从分子分母同时剔除

# ---- 口径2: 活跃贷款 (active loans) ----
# 定义: 当月有表现记录 且 尚未终止 (zero_balance_code 为空).
# 贷款终止当月带有ZB code (01=提前还清, 02=第三方出售, 03=短售, 09=REO处置, 96=回购...),
# EXCLUDE_ZB_MONTH=True 表示终止当月即不再计入活跃 (标准做法)
EXCLUDE_ZB_MONTH = True

# ---- 口径3: 分组方式 (cohort definition) ----
# 可选维度: "vintage", "fico_bucket", "ltv_bucket", "dti_bucket", "property_state", "loan_purpose"
COHORT_DIMS = ["vintage", "fico_bucket"]

# 分档边界 (右闭). Freddie缺失编码: credit_score=9999, ltv/dti=999 -> 自动归入 "NA" 档
FICO_BINS = [300, 680, 740, 850]
FICO_LABELS = ["FICO<=680", "FICO681-740", "FICO>740"]
LTV_BINS = [0, 80, 105]
LTV_LABELS = ["LTV<=80", "LTV>80"]
DTI_BINS = [0, 36, 65]
DTI_LABELS = ["DTI<=36", "DTI>36"]

# ---- 口径4: (可选) 宏观变量合并 ----
# UNRATE CSV is read from the macro directory under FREDDIE_BASE.
# FRED导出格式: 两列, observation_date(或DATE), UNRATE. 文件不存在则自动跳过合并.
UNRATE_CSV = BASE / "macro" / "UNRATE.csv"

# 信用事件类的终止代码 (统计退出流量用; 01=提前还清单独计)
ZB_CREDIT_CODES = {"02", "03", "09"}
ZB_PREPAY_CODES = {"01"}


# ============================================================
# 口径实现
# ============================================================
def make_default_flag(df: pd.DataFrame) -> pd.DataFrame:
    """构造违约旗标. dlq_num: 数值化的逾期档位; is_ra; default_flag; dlq_known."""
    s = df["dlq_status"].astype("string").str.strip()
    df["is_ra"] = (s == "RA")
    df["dlq_num"] = pd.to_numeric(s.where(~df["is_ra"]), errors="coerce")
    df["dlq_known"] = df["is_ra"] | df["dlq_num"].notna()
    df["default_flag"] = (df["dlq_num"] >= DPD_THRESHOLD)
    if INCLUDE_RA:
        df["default_flag"] = df["default_flag"] | df["is_ra"]
    return df


def make_fb_flag(df: pd.DataFrame) -> pd.DataFrame:
    """宽限/灾害纾困识别 (字段2014-01起有值; 缺列时全False保持兼容).
       in_fb = 处于任一纾困计划(F/R/T) 或 灾害致逾期Y."""
    bac = (df["borrower_assistance_code"].astype("string").str.strip()
           if "borrower_assistance_code" in df.columns else pd.Series("", index=df.index, dtype="string"))
    dis = (df["dlq_due_to_disaster"].astype("string").str.strip()
           if "dlq_due_to_disaster" in df.columns else pd.Series("", index=df.index, dtype="string"))
    df["in_fb"] = bac.isin(["F", "R", "T"]).fillna(False) | (dis == "Y").fillna(False)
    return df


def make_active_flag(df: pd.DataFrame) -> pd.DataFrame:
    """活跃 = 有记录 且 (可选)未带终止代码 且 (可选)逾期状态可知."""
    zb = df["zero_balance_code"].astype("string").str.strip()
    zb_isnull = zb.isna() | (zb == "")
    df["active"] = zb_isnull if EXCLUDE_ZB_MONTH else True
    if DROP_UNKNOWN_DLQ:
        df["active"] = df["active"] & df["dlq_known"]
    # 终止事件类型 (仅在带ZB code的行上非空)
    df["zb_credit"] = zb.isin(ZB_CREDIT_CODES)
    df["zb_prepay"] = zb.isin(ZB_PREPAY_CODES)
    return df


def bucketize(df: pd.DataFrame) -> pd.DataFrame:
    """静态特征分档. 缺失编码(9999/999)先转NaN, 分档后缺失归 'NA'."""
    def cut(col, bins, labels, na_hi):
        x = df[col].where(df[col] < na_hi)          # 9999/999 缺失编码 -> NaN
        b = pd.cut(x, bins=bins, labels=labels, include_lowest=True)
        return b.cat.add_categories("NA").fillna("NA").astype(str)

    if "fico_bucket" in COHORT_DIMS:
        df["fico_bucket"] = cut("credit_score", FICO_BINS, FICO_LABELS, na_hi=851)
    if "ltv_bucket" in COHORT_DIMS:
        df["ltv_bucket"] = cut("orig_ltv", LTV_BINS, LTV_LABELS, na_hi=106)
    if "dti_bucket" in COHORT_DIMS:
        df["dti_bucket"] = cut("orig_dti", DTI_BINS, DTI_LABELS, na_hi=66)
    return df


def flag_new_default(df: pd.DataFrame) -> pd.DataFrame:
    """流量口径: 每笔贷款首次进入违约状态的那个月 new_default=1 (仅标一次)."""
    df = df.sort_values(["loan_sequence_number", "monthly_reporting_period"])
    first_def = (df[df["default_flag"]]
                 .groupby("loan_sequence_number")["monthly_reporting_period"]
                 .min().rename("first_def_period"))
    df = df.merge(first_def, on="loan_sequence_number", how="left")
    df["new_default"] = (df["monthly_reporting_period"] == df["first_def_period"])
    return df.drop(columns=["first_def_period"])


def aggregate_counts(df: pd.DataFrame) -> pd.DataFrame:
    """按 cohort × month 聚合成计数 (先计数后合并, 保证跨vintage文件可再聚合)."""
    keys = COHORT_DIMS + ["monthly_reporting_period"]
    act = df[df["active"]].copy()
    act = act.assign(default_excl_fb=act["default_flag"] & ~act["in_fb"],
                     def_in_fb=act["default_flag"] & act["in_fb"])
    g = act.groupby(keys, observed=True)
    out = g.agg(
        n_active=("loan_sequence_number", "size"),
        n_default=("default_flag", "sum"),
        n_default_excl_fb=("default_excl_fb", "sum"),   # 剔除宽限口径分子
        n_def_in_fb=("def_in_fb", "sum"),               # 违约且在宽限中的数量(描述用)
        n_new_default=("new_default", "sum"),
        sum_upb=("current_upb", "sum"),
    ).reset_index()
    # 终止流量在非活跃行上, 单独聚合后并入
    term = (df.groupby(keys, observed=True)
              .agg(n_exit_credit=("zb_credit", "sum"),
                   n_exit_prepay=("zb_prepay", "sum"),
                   sum_actual_loss=("actual_loss", "sum"))
              .reset_index())
    return out.merge(term, on=keys, how="outer")


# ============================================================
# 主流程: 逐vintage文件处理 -> 计数合并 -> 比率 -> UNRATE -> 落盘
# ============================================================
def main():
    log("=" * 64)
    log("Steps 5-7 | 违约指标 + 活跃定义 + cohort-month 聚合")
    log(f"口径: {DPD_THRESHOLD*30}+DPD  RA计入={INCLUDE_RA}  剔除未知dlq={DROP_UNKNOWN_DLQ}")
    log(f"分组: {COHORT_DIMS}")
    log("=" * 64)

    pieces = []
    if not LOAN_MONTH_DIR.exists():
        sys.exit(f"Missing output from step 01: {LOAN_MONTH_DIR}")
    lm_dir = LOAN_MONTH_DIR
    log(f"输入目录: {lm_dir.relative_to(BASE)}")
    for y in YEARS:
        f = lm_dir / f"loan_month_{y}.parquet"
        if not f.exists():
            log(f"[跳过] 缺 {f.name}")
            continue
        df = pd.read_parquet(f)
        df = make_default_flag(df)
        df = make_fb_flag(df)
        df = make_active_flag(df)
        df = bucketize(df)
        df = flag_new_default(df)
        c = aggregate_counts(df)
        pieces.append(c)
        n_act = int(df["active"].sum())
        n_def = int((df["active"] & df["default_flag"]).sum())
        log(f"[{y}] 行={len(df):>10,}  活跃行={n_act:>10,}  "
            f"违约行={n_def:>8,}  ({n_def/max(n_act,1):.3%})")
        del df

    if not pieces:
        sys.exit("没有任何输入文件 — 先跑 01 脚本")

    # ---- 跨vintage合并计数, 再按最终分组维度求和 (支持 COHORT_DIMS 不含 vintage 的情形) ----
    keys = COHORT_DIMS + ["monthly_reporting_period"]
    counts = (pd.concat(pieces, ignore_index=True)
                .groupby(keys, observed=True).sum(min_count=1).reset_index())

    # ---- 比率 ----
    counts["default_rate"] = counts["n_default"] / counts["n_active"]
    counts["default_rate_excl_fb"] = counts["n_default_excl_fb"] / counts["n_active"]
    counts["new_default_rate"] = counts["n_new_default"] / counts["n_active"]
    counts["month"] = pd.to_datetime(counts["monthly_reporting_period"], format="%Y%m")

    # ---- (可选) 合并 UNRATE ----
    if UNRATE_CSV.exists():
        un = pd.read_csv(UNRATE_CSV)
        date_col = "observation_date" if "observation_date" in un.columns else "DATE"
        un["month"] = pd.to_datetime(un[date_col])
        counts = counts.merge(un[["month", "UNRATE"]], on="month", how="left")
        log(f"已合并 UNRATE ({UNRATE_CSV.name}), 覆盖 {counts['UNRATE'].notna().mean():.1%} 的行")
    else:
        log(f"[提示] 未找到 {UNRATE_CSV} — 跳过宏观合并. "
            f"从FRED下载UNRATE.csv放到 macro/ 后重跑即可")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts.to_parquet(OUT_DIR / "cohort_month_panel.parquet", index=False)
    log(f"-> processed/02_cohort/cohort_month_panel.parquet  "
        f"({len(counts):,} 行 = {counts.groupby(COHORT_DIMS, observed=True).ngroups} 组 × 月)")

    # ---- Aggregate monthly series ----
    overall = (counts.groupby("month")
               .agg(n_active=("n_active", "sum"), n_default=("n_default", "sum"),
                    n_default_excl_fb=("n_default_excl_fb", "sum"),
                    n_new_default=("n_new_default", "sum"))
               .reset_index())
    overall["default_rate"] = overall["n_default"] / overall["n_active"]
    overall["default_rate_excl_fb"] = overall["n_default_excl_fb"] / overall["n_active"]
    overall["new_default_rate"] = overall["n_new_default"] / overall["n_active"]
    overall.to_csv(OUT_DIR / "overall_month_series.csv", index=False)

    # ---- Construction summary ----
    with open(OUT_DIR / "cohort_build_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(f"运行时间: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        fh.write(f"DPD_THRESHOLD={DPD_THRESHOLD} ({DPD_THRESHOLD*30}+DPD)\n")
        fh.write(f"INCLUDE_RA={INCLUDE_RA}\nDROP_UNKNOWN_DLQ={DROP_UNKNOWN_DLQ}\n")
        fh.write(f"EXCLUDE_ZB_MONTH={EXCLUDE_ZB_MONTH}\nCOHORT_DIMS={COHORT_DIMS}\n")
        fh.write(f"FICO_BINS={FICO_BINS}\nLTV_BINS={LTV_BINS}\nDTI_BINS={DTI_BINS}\n")
        fh.write(f"组数={counts.groupby(COHORT_DIMS, observed=True).ngroups}\n")
        fh.write(f"月份范围={overall['month'].min():%Y-%m} ~ {overall['month'].max():%Y-%m}\n")
        m2020 = overall[overall['month'].dt.year == 2020]
        if len(m2020):
            fh.write(
                "2020 borrower-assistance-adjusted peak="
                f"{m2020['default_rate_excl_fb'].max():.4%}\n"
            )
        fh.write(
            "Adjusted outcome full-sample mean="
            f"{overall['default_rate_excl_fb'].mean():.4%}\n"
        )
    log("-> processed/02_cohort/cohort_build_summary.txt")
    log("=" * 64)
    log("完成.")


if __name__ == "__main__":
    main()
