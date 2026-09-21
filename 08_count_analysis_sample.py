#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08 Audit the final Freddie Mac analysis-sample counts.

The reported sample contains origination vintages 1999–2019 and monthly
performance records through December 2020. The script reads only the loan
identifier and monthly reporting period from the loan-month Parquet files.

Outputs
-------
processed/08_sample_counts/counts_by_vintage.csv
processed/08_sample_counts/sample_counts.json

Run
---
python 08_count_analysis_sample.py
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


BASE = Path(os.environ.get("FREDDIE_BASE", Path(__file__).resolve().parent)).expanduser()
PROCESSED = BASE / "processed"
OUT_DIR = PROCESSED / "08_sample_counts"

MIN_VINTAGE = 1999
MAX_VINTAGE = 2019
MIN_PERIOD = 199901
MAX_PERIOD = 202012
BATCH_SIZE = 500_000
ID_COLUMN = "loan_sequence_number"
PERIOD_COLUMN = "monthly_reporting_period"


def discover_parquet_files() -> list[Path]:
    files = sorted(
        (PROCESSED / "01_loan_month").glob("loan_month_*.parquet")
    )
    if not files:
        raise FileNotFoundError(
            "No loan-month Parquet files found under "
            f"{PROCESSED / '01_loan_month'}."
        )
    return files


def extract_vintage(path: Path) -> int:
    flat_match = re.search(r"loan_month_(\d{4})\.parquet$", path.name)
    if flat_match:
        return int(flat_match.group(1))

    raise ValueError(f"Cannot infer vintage year from path: {path}")


def scan_files(files: list[Path]) -> pd.DataFrame:
    rows_by_vintage: dict[int, int] = defaultdict(int)
    loan_ids_by_vintage: dict[int, set[str]] = defaultdict(set)
    min_period_by_vintage: dict[int, int] = {}
    max_period_by_vintage: dict[int, int] = {}

    eligible_files = [
        path
        for path in files
        if MIN_VINTAGE <= extract_vintage(path) <= MAX_VINTAGE
    ]
    print(f"Scanning {len(eligible_files):,} Parquet files...")

    for file_number, path in enumerate(eligible_files, start=1):
        vintage = extract_vintage(path)
        parquet_file = pq.ParquetFile(path)
        available_columns = set(parquet_file.schema_arrow.names)
        missing = {ID_COLUMN, PERIOD_COLUMN} - available_columns
        if missing:
            raise KeyError(f"{path} is missing required columns: {sorted(missing)}")

        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=[ID_COLUMN, PERIOD_COLUMN],
        ):
            data = batch.to_pandas()
            period = pd.to_numeric(data[PERIOD_COLUMN], errors="coerce")
            valid = period.notna() & period.between(MIN_PERIOD, MAX_PERIOD)
            if not valid.any():
                continue

            valid_period = period.loc[valid].astype("int64")
            valid_ids = data.loc[valid, ID_COLUMN].astype("string").str.strip()
            valid_ids = valid_ids[valid_ids.notna() & (valid_ids != "")]

            rows_by_vintage[vintage] += int(valid.sum())
            loan_ids_by_vintage[vintage].update(valid_ids.tolist())
            batch_min = int(valid_period.min())
            batch_max = int(valid_period.max())
            min_period_by_vintage[vintage] = min(
                min_period_by_vintage.get(vintage, batch_min), batch_min
            )
            max_period_by_vintage[vintage] = max(
                max_period_by_vintage.get(vintage, batch_max), batch_max
            )

        if file_number == 1 or file_number % 20 == 0:
            print(f"[{file_number:>4}/{len(eligible_files)}] finished {path.name}")

    records = []
    for vintage in sorted(rows_by_vintage):
        records.append(
            {
                "vintage": vintage,
                "n_mortgages": len(loan_ids_by_vintage[vintage]),
                "n_loan_months": rows_by_vintage[vintage],
                "min_reporting_period": min_period_by_vintage[vintage],
                "max_reporting_period": max_period_by_vintage[vintage],
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = discover_parquet_files()
    by_vintage = scan_files(files)
    if by_vintage.empty:
        raise ValueError("No observations found in the final analysis window.")

    sample_counts = {
        "minimum_vintage": MIN_VINTAGE,
        "maximum_vintage": MAX_VINTAGE,
        "maximum_reporting_period": MAX_PERIOD,
        "n_vintages": int(by_vintage["vintage"].nunique()),
        "n_mortgages": int(by_vintage["n_mortgages"].sum()),
        "n_loan_month_observations": int(by_vintage["n_loan_months"].sum()),
    }

    by_vintage.to_csv(OUT_DIR / "counts_by_vintage.csv", index=False)
    with open(OUT_DIR / "sample_counts.json", "w", encoding="utf-8") as file:
        json.dump(sample_counts, file, indent=2)

    print(f"Mortgages: {sample_counts['n_mortgages']:,}")
    print(
        "Loan-month observations: "
        f"{sample_counts['n_loan_month_observations']:,}"
    )
    print(f"Vintage range: {MIN_VINTAGE}-{MAX_VINTAGE}")
    print("Monthly records end: 2020-12")
    print(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
