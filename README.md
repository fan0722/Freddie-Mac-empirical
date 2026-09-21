# Freddie Mac stress-test replication package

This repository reproduces the Freddie Mac empirical application. It contains the final analysis pipeline, public macroeconomic inputs, aggregate final-stage inputs, and reference outputs. Intermediate specifications and unused figures are not included.

## Quick reproduction of the reported results

The aggregate files in `replication_inputs/` allow the final table and figure to be regenerated without redistributing Freddie Mac loan-level data.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python 07_results.py
```

The outputs are written to `processed/07_results/`:

```text
fig_freddie_intervals.png
fig_freddie_intervals.pdf
table_coverage.csv
calibration_summary.txt
diagnostics.csv
summary.json
```

The reported realized outcome is the borrower-assistance-adjusted 90+DPD rate. The final figure compares the two outcome-side `gamma_Y` calibrations described in the paper: portfolio AR(1) residuals and portfolio prediction residuals. Reference outputs are provided in `expected_outputs/`.

## Analysis sample and design

| Item | Final specification |
| --- | --- |
| Origination vintages | 1999–2019 |
| Last performance month | 2020-12 |
| Distinct mortgages | 1,049,994 |
| Loan-month records | 55,718,573 |
| Starting portfolio | 62 vintage × FICO-bucket cohorts in December 2019 |
| Minimum starting cohort size | 100 active loans |
| Training period | 1999–2014 |
| Conformal calibration period | 2015–2019 |
| Stress evaluation | January–December 2020 under the realized UNRATE path |
| Portfolio weights | December 2019 cohort current-UPB weights, fixed across 2020 |
| Reported realized outcome | 90+DPD excluding borrower-assistance and disaster-coded loan-months from the numerator |

See [DATA_README.md](DATA_README.md) for the field definitions and construction details.

## Complete pipeline from source data

Obtain the annual 1999–2019 SFLLD **sample** archives directly from Freddie Mac under the applicable access terms. No Freddie Mac loan-level record is redistributed here.

Set a private working directory and copy the included macro snapshots into it:

```bash
export FREDDIE_BASE="$HOME/freddiemac_sflld"
mkdir -p "$FREDDIE_BASE/macro"
cp macro/UNRATE.csv macro/NFCI.csv "$FREDDIE_BASE/macro/"
```

Place the Freddie Mac sample archives in `$FREDDIE_BASE/raw_zip/`, then run:

```bash
python 01_build_loan_month_panel.py
python 02_build_cohort_panel.py
python 03_learner.py
python 04_rollout.py
python 05_conformal.py
python 06_calibrate_ch.py
python 07_results.py
python 08_count_analysis_sample.py
```

| Step | Script | Output used in the final analysis |
| --- | --- | --- |
| 01 | `01_build_loan_month_panel.py` | Loan-month Parquet files for 1999–2019 vintages |
| 02 | `02_build_cohort_panel.py` | Vintage × FICO cohort-month panel and adjusted outcome |
| 03 | `03_learner.py` | Selected base learner, feature specification, and OLS persistence estimate |
| 04 | `04_rollout.py` | Recursive 2020 portfolio forecast with fixed December 2019 weights |
| 05 | `05_conformal.py` | Weighted conformal and benchmark interval widths |
| 06 | `06_calibrate_ch.py` | Both reported `gamma_Y` calibrations and their confounding envelopes |
| 07 | `07_results.py` | Final adjusted-outcome table, two-panel figure, and diagnostics |
| 08 | `08_count_analysis_sample.py` | Final sample-count audit |

The early stages require substantial memory and compute time but do not depend on a particular cluster scheduler.

## Included and excluded data

Included:

- `macro/UNRATE.csv`, monthly through 2020-12;
- `macro/NFCI.csv`, weekly through 2020-12;
- aggregate, non-loan-level files in `replication_inputs/`;
- final reference outputs in `expected_outputs/`.

Excluded:

- Freddie Mac origination and monthly-performance records;
- `raw_zip/` and `raw_txt/`;
- loan-level and cohort-level Parquet files;
- generated `processed/` files and logs.

## Repository layout

```text
freddie_replication/
├── 01_build_loan_month_panel.py
├── 02_build_cohort_panel.py
├── 03_learner.py
├── 04_rollout.py
├── 05_conformal.py
├── 06_calibrate_ch.py
├── 07_results.py
├── 08_count_analysis_sample.py
├── macro/
├── replication_inputs/
├── expected_outputs/
├── README.md
├── DATA_README.md
├── PATHS.md
├── variable_dictionary.csv
├── requirements.txt
└── .gitignore
```

## Software

Use Python 3.10 or newer. Install the nonstandard dependencies from `requirements.txt`.

## Source documentation

- Freddie Mac SFLLD: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
- Freddie Mac SFLLD data guide: https://www.freddiemac.com/research/pdf/user_guide.pdf
- FRED UNRATE: https://fred.stlouisfed.org/series/UNRATE
- Chicago Fed NFCI: https://www.chicagofed.org/research/data/nfci/current-data
