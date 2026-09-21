# -*- coding: utf-8 -*-
"""
01 Build the Freddie Mac loan-month panel.

The script reads the 1999–2019 SFLLD sample origination and monthly-performance
files, assigns the documented positional field names, joins records by loan
sequence number, and writes one loan-month Parquet file per origination vintage.
Files are processed one vintage at a time to limit memory use.

Outputs
-------
processed/01_loan_month/orig_combined.parquet
processed/01_loan_month/loan_month_YYYY.parquet
processed/01_loan_month/build_summary.csv

Run
---
python 01_build_loan_month_panel.py
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
import zipfile
import sys
import pandas as pd


def log(msg: str):
    """带时间戳的规整输出 (批处理日志友好)."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)

# ============================================================
# Configuration
# ============================================================
BASE = Path(os.environ.get("FREDDIE_BASE", Path(__file__).resolve().parent)).expanduser()
RAW_ZIP = BASE / "raw_zip"
RAW_TXT = BASE / "raw_txt"
PROCESSED = BASE / "processed"
LOAN_MONTH_DIR = PROCESSED / "01_loan_month"

YEARS = list(range(1999, 2020))

# False keeps the analysis fields required by the downstream pipeline.
# True retains every field in the documented input layout.
KEEP_ALL_COLUMNS = False

SEP = "|"

# ============================================================
# 文件布局 (Pre-July 2026, 32 + 32 字段) —— 顺序不能动
# ============================================================
ORIG_COLS = [
    "credit_score",              # 1  Credit Score
    "first_payment_date",        # 2  First Payment Date (YYYYMM)
    "first_time_homebuyer_flag", # 3
    "maturity_date",             # 4  (YYYYMM)
    "msa",                       # 5
    "mi_pct",                    # 6
    "num_units",                 # 7
    "occupancy_status",          # 8
    "orig_cltv",                 # 9
    "orig_dti",                  # 10
    "orig_upb",                  # 11
    "orig_ltv",                  # 12
    "orig_interest_rate",        # 13
    "channel",                   # 14
    "ppm_flag",                  # 15
    "amortization_type",         # 16
    "property_state",            # 17
    "property_type",             # 18
    "postal_code",               # 19
    "loan_sequence_number",      # 20  主键
    "loan_purpose",              # 21
    "orig_loan_term",            # 22
    "num_borrowers",             # 23
    "seller_name",               # 24
    "servicer_name",             # 25
    "super_conforming_flag",     # 26
    "pre_harp_loan_seq",         # 27
    "program_indicator",         # 28
    "harp_indicator",            # 29
    "property_valuation_method", # 30
    "io_indicator",              # 31
    "mi_cancellation_indicator", # 32
]

SVCG_COLS = [
    "loan_sequence_number",        # 1  主键
    "monthly_reporting_period",    # 2  (YYYYMM)
    "current_upb",                 # 3
    "dlq_status",                  # 4  Current Loan Delinquency Status (字符串! 含 RA)
    "loan_age",                    # 5
    "remaining_months",            # 6
    "defect_settlement_date",      # 7
    "modification_flag",           # 8
    "zero_balance_code",           # 9  (字符串, 如 "01","03","09")
    "zero_balance_effective_date", # 10 (YYYYMM)
    "current_interest_rate",       # 11
    "current_deferred_upb",        # 12
    "ddlpi",                       # 13
    "mi_recoveries",               # 14
    "net_sales_proceeds",          # 15 (Alpha-Numeric, 可能含 "C"/"U", 保持字符串)
    "non_mi_recoveries",           # 16
    "expenses",                    # 17
    "legal_costs",                 # 18
    "maintenance_costs",           # 19
    "taxes_insurance",             # 20
    "misc_expenses",               # 21
    "actual_loss",                 # 22
    "modification_cost",           # 23
    "step_mod_flag",               # 24
    "deferred_payment_plan",       # 25
    "eltv",                        # 26
    "zero_balance_removal_upb",    # 27
    "delinquent_accrued_interest", # 28
    "dlq_due_to_disaster",         # 29
    "borrower_assistance_code",    # 30
    "current_month_mod_cost",      # 31
    "interest_bearing_upb",        # 32
]

# ---- 需要转成数值的列 ----
ORIG_NUMERIC = [
    "credit_score", "msa", "mi_pct", "num_units", "orig_cltv", "orig_dti",
    "orig_upb", "orig_ltv", "orig_interest_rate", "orig_loan_term",
    "num_borrowers", "property_valuation_method",
]
SVCG_NUMERIC = [
    "current_upb", "loan_age", "remaining_months", "current_interest_rate",
    "current_deferred_upb", "mi_recoveries", "non_mi_recoveries", "expenses",
    "legal_costs", "maintenance_costs", "taxes_insurance", "misc_expenses",
    "actual_loss", "modification_cost", "eltv", "zero_balance_removal_upb",
    "delinquent_accrued_interest", "current_month_mod_cost",
    "interest_bearing_upb",
]

# ---- Analysis fields retained when KEEP_ALL_COLUMNS is False ----
ORIG_KEEP = [
    "loan_sequence_number", "credit_score", "orig_ltv", "orig_cltv",
    "orig_dti", "orig_upb", "orig_interest_rate", "orig_loan_term",
    "loan_purpose", "property_state", "occupancy_status",
    "first_payment_date",
]
SVCG_KEEP = [
    "loan_sequence_number", "monthly_reporting_period", "current_upb",
    "dlq_status", "loan_age", "modification_flag",
    "zero_balance_code", "zero_balance_effective_date",
    "actual_loss", "zero_balance_removal_upb",
    "dlq_due_to_disaster",        # 灾害致逾期 Y/空 (2014-01起有值)
    "borrower_assistance_code",   # 纾困状态 F=宽限/R/T/空 (2014-01起有值)
]


# ============================================================
# 工具函数
# ============================================================
def check_field_count(txt_path: Path, expected: int) -> int:
    """读第一行, 数 '|' 分隔的字段数, 校验布局版本 (防 July-2026 新版式混入).

    注意区分两种以 '|' 结尾的情况:
      - 恰好 expected 个字段且最后一个字段为空  -> 正常, 不警告
      - expected+1 个token且最后为空            -> 悬挂分隔符, 加一个哑列吸收
      - 其他任何数量                            -> 布局可能不对, 强警告
    """
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline().rstrip("\n").rstrip("\r")
    n = first.count(SEP) + 1          # split后的token数
    if n == expected:
        return n
    if n == expected + 1 and first.endswith(SEP):
        return n                      # 悬挂分隔符, read_pipe_file 会加哑列
    log(f"  [!!警告!!] {txt_path.name}: 检测到 {n} 个字段, 预期 {expected}."
          f" 该文件可能是新版(July 2026)布局, 必须人工核对 File Layout, 否则列名会错位!")
    return n


def read_pipe_file(txt_path: Path, colnames: list, numeric_cols: list,
                   keep: list | None) -> pd.DataFrame:
    """读无表头 | 分隔文件 -> 赋列名 -> 数值转换 -> (可选)裁剪列."""
    n_fields = check_field_count(txt_path, len(colnames))
    names = list(colnames)
    extra = None
    if n_fields > len(colnames):          # 容忍悬挂分隔符产生的空尾列
        extra = [f"_extra{i}" for i in range(n_fields - len(colnames))]
        names = names + extra

    usecols = None
    if keep is not None:
        usecols = [names.index(c) for c in keep]

    df = pd.read_csv(
        txt_path, sep=SEP, header=None, names=names, usecols=usecols,
        dtype=str, engine="c", na_values=["", " "], keep_default_na=False,
        encoding="utf-8", encoding_errors="replace",
    )
    if extra and keep is None:
        df = df.drop(columns=extra)

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 去掉关键字符串列的首尾空格
    for c in ("loan_sequence_number", "dlq_status", "zero_balance_code",
              "dlq_due_to_disaster", "borrower_assistance_code"):
        if c in df.columns:
            df[c] = df[c].str.strip()
    return df


# ============================================================
# Step 1: 解压
# ============================================================
def unzip_all():
    RAW_TXT.mkdir(parents=True, exist_ok=True)
    for y in YEARS:
        zpath = RAW_ZIP / f"sample_{y}.zip"
        if not zpath.exists():
            log(f"[跳过]     找不到 {zpath.name}")
            continue
        orig_txt = RAW_TXT / f"sample_orig_{y}.txt"
        svcg_txt = RAW_TXT / f"sample_svcg_{y}.txt"
        if orig_txt.exists() and svcg_txt.exists():
            log(f"[跳过解压] {y}: txt已存在")
            continue
        log(f"[解压]     {zpath.name}")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(RAW_TXT)
    log("Step 1 (解压) 完成.")


# ============================================================
# Steps 2-4: 逐年 读取 -> 赋名 -> 合并 -> 落盘
# ============================================================
def build_year(y: int, summary: list) -> pd.DataFrame | None:
    orig_txt = RAW_TXT / f"sample_orig_{y}.txt"
    svcg_txt = RAW_TXT / f"sample_svcg_{y}.txt"
    if not orig_txt.exists() or not svcg_txt.exists():
        print(f"[跳过] {y}: txt 文件缺失")
        return None

    keep_o = None if KEEP_ALL_COLUMNS else ORIG_KEEP
    keep_s = None if KEEP_ALL_COLUMNS else SVCG_KEEP

    orig = read_pipe_file(orig_txt, ORIG_COLS, ORIG_NUMERIC, keep_o)
    svcg = read_pipe_file(svcg_txt, SVCG_COLS, SVCG_NUMERIC, keep_s)

    orig["vintage"] = y

    # ---- Step 4: 一对多合并 (svcg 为主表, 左连 orig 的静态特征) ----
    panel = svcg.merge(orig, on="loan_sequence_number",
                       how="left", validate="many_to_one")

    # ---- 质量检查 ----
    n_orig_loans = orig["loan_sequence_number"].nunique()
    n_svcg_loans = svcg["loan_sequence_number"].nunique()
    n_rows = len(panel)
    unmatched = panel["vintage"].isna().sum()   # 左连后 vintage 缺失 = svcg里有但orig里没有
    dlq_counts = (panel["dlq_status"].value_counts(dropna=False)
                  .head(8).to_dict())

    log(f"[{y}] orig={n_orig_loans:>8,}  svcg={n_svcg_loans:>8,}  "
        f"loan-month={n_rows:>12,}  未匹配={unmatched:,}")
    if unmatched > 0:
        log(f"  [!!警告!!] {y}: {unmatched} 行月度记录在发放文件中找不到对应贷款")

    summary.append({
        "vintage": y,
        "orig_loans": n_orig_loans,
        "svcg_loans": n_svcg_loans,
        "loan_month_rows": n_rows,
        "unmatched_rows": int(unmatched),
        "min_period": panel["monthly_reporting_period"].min(),
        "max_period": panel["monthly_reporting_period"].max(),
        "dlq_top_values": str(dlq_counts),
    })

    # ---- 落盘 ----
    LOAN_MONTH_DIR.mkdir(parents=True, exist_ok=True)
    out = LOAN_MONTH_DIR / f"loan_month_{y}.parquet"
    panel.to_parquet(out, index=False)
    log(f"      -> {out.relative_to(BASE)}")
    return orig


def main():
    log("=" * 64)
    log("Freddie Mac SFLLD  |  Steps 1-4  |  构建 loan-month 面板")
    log(f"模式: {'全部列' if KEEP_ALL_COLUMNS else '精简列'}   年份: {YEARS[0]}-{YEARS[-1]}   路径: {BASE}")
    log("=" * 64)

    unzip_all()

    summary, orig_frames = [], []
    for y in YEARS:
        o = build_year(y, summary)
        if o is not None:
            orig_frames.append(o)

    if not orig_frames:
        sys.exit("没有处理任何年份 — 检查 raw_zip 下的文件名是否为 sample_YYYY.zip")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    orig_all = pd.concat(orig_frames, ignore_index=True)
    orig_all.to_parquet(LOAN_MONTH_DIR / "orig_combined.parquet", index=False)
    pd.DataFrame(summary).to_csv(LOAN_MONTH_DIR / "build_summary.csv", index=False)

    log("=" * 64)
    log(f"全部完成 | 发放文件合计 {len(orig_all):,} 笔贷款 | {len(summary)} 个vintage")
    log(f"摘要表:   processed/01_loan_month/build_summary.csv  (先看这个核对!)")
    log(f"面板读法: pd.read_parquet('{LOAN_MONTH_DIR}')")
    log("=" * 64)


if __name__ == "__main__":
    main()
