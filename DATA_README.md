# Data and variable construction

## Sources and access

1. **Freddie Mac SFLLD sample files.** Download the annual sample archives for origination vintages 1999–2019 from the [Freddie Mac SFLLD page](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset) using the access method and terms in force at download time. The sample guide describes paired `sample_orig_YYYY.txt` and `sample_svcg_YYYY.txt` pipe-delimited files inside `sample_YYYY.zip` archives. Use the **sample**, not the full standard dataset. No Freddie Mac record is included in this package. Freddie Mac states that the dataset can be revised; record the release or download date for reproducibility.
2. **UNRATE.** The package includes the monthly, seasonally adjusted U.S. civilian unemployment rate from [FRED series UNRATE](https://fred.stlouisfed.org/series/UNRATE) in `macro/UNRATE.csv`, through 2020-12. The 2020 realized series is the stress path; pre-2020 observations are used for modeling and calibration.
3. **NFCI.** The package includes the Chicago Fed's weekly [National Financial Conditions Index](https://www.chicagofed.org/research/data/nfci/current-data) in `macro/NFCI.csv`, through 2020-12. The calibration uses the last weekly observation in each month and standardizes the resulting monthly series over the relevant pre-2020 window.

The included macro snapshots are sufficient for the documented analysis window. See [PATHS.md](PATHS.md) for both the quick final-stage reproduction and the complete private-data layout.

## Aggregate replication inputs

`replication_inputs/` contains only the aggregate model and portfolio outputs required by `07_results.py`. It contains no loan sequence numbers, borrower identifiers, origination records, or monthly loan-performance records. `realized_excl_fb_2020.csv` has one row per 2020 stress horizon. It was constructed from the 62 cohorts selected in December 2019 and uses the same December 2019 current-UPB weights at every horizon.

`07_results.py` uses newly generated files under `processed/` when they exist. Otherwise, it uses the distributed aggregate inputs. `expected_outputs/` contains the final reference table, figure, calibration summary, and diagnostics.

## Raw-to-analysis field map

The table gives the raw-to-code mapping implemented by `01_build_loan_month_panel.py`. Freddie Mac's raw files are positional, so confirm that the downloaded files use the pre-July 2026 layout and filenames before running.

| Freddie Mac raw field | Analysis field or use | Construction |
| --- | --- | --- |
| Loan Sequence Number (origination and performance) | `loan_sequence_number` | Unique loan key used to join the two files and count distinct mortgages. |
| Credit Score (origination) | `credit_score`, `fico_bucket` | Origination FICO; `9999` is missing. Bucket boundaries must match the final `02_build_cohort_panel.py`. |
| Origination vintage / source sample year | `vintage` | Keep 1999–2019 vintages. Do not add 2020 originations to the December 2019 starting portfolio. |
| Monthly Reporting Period (performance) | `monthly_reporting_period`, `month` | `YYYYMM`; retain records through `202012`. |
| Current Loan Delinquency Status (performance) | `dlq_status`, `default_flag` | Numeric status `>=3` means 90+ days past due; `RA` is also included. |
| Zero Balance Code (performance) | `zero_balance_code`, `active` | A populated termination code means the loan is excluded from that month's active denominator in the documented construction. |
| Current Actual UPB (performance) | `current_upb`, `sum_upb`, fixed weight | Sum active-loan balances by cohort in December 2019; freeze these cohort weights for 2020 portfolio aggregation. |
| Borrower Assistance Status Code (performance) | `borrower_assistance_code`, `in_fb` | `F`, `R`, or `T` marks the corresponding assisted loan-month. |
| Delinquency Due to Disaster (performance) | `dlq_due_to_disaster`, `in_fb` | `Y` marks the corresponding disaster-related loan-month. |

Freddie Mac defines delinquency status `3` as 90–119 days late and `RA` as REO acquisition. `02_build_cohort_panel.py` includes `RA` as a delinquency event and omits unknown delinquency status from the active denominator. Borrower-assistance and disaster fields are populated only for January 2014 onward in Freddie Mac's data guide.

## Cohort-month outcome

For each month, start with loans having a performance record and meeting the active-loan rule. Group by origination vintage and FICO bucket. Let `n_active` be the number of active loans and `n_default` the number with a 90+DPD status. The recorded `default_rate` is `n_default / n_active`. Training and calibration retain cohort-months with at least 100 active loans. The stress portfolio is selected using the same threshold in December 2019, leaving 62 cohorts; those cohorts are then followed through the 12 stress horizons without reselecting the portfolio each month.

The reported **borrower-assistance-adjusted** outcome changes only the numerator: an active loan-month with assistance code `F`, `R`, or `T`, or disaster-delinquency flag `Y`, is not counted in `n_default_excl_fb`. The loan-month remains in `n_active`, and `default_rate_excl_fb = n_default_excl_fb / n_active`. The model is trained using the recorded cohort rate `default_rate`; the realized 2020 comparison in the final table and figure uses `default_rate_excl_fb`. This adjustment is a measurement choice and does not assert that assisted borrowers had no economic distress.

The 2020 portfolio rate is a weighted average of cohort rates with weights proportional to each cohort's **December 2019 current UPB**. The weights remain fixed for all 12 2020 months. This prevents changing cohort balances from mechanically changing the portfolio mix in the plotted series. The learner uses unemployment, lagged cohort delinquency, cohort age, and FICO indicators; the exact final feature construction is controlled by `03_learner.py`.

## Audit totals and limits

`08_count_analysis_sample.py` reported 1,049,994 mortgages and 55,718,573 loan-month observations for 1999–2019 origination vintages with reporting months through 2020-12. These are **raw analysis-window counts** and should not be confused with the number of loans active in December 2019 or with the cohort-month modeling rows. Re-downloads of a revised SFLLD release can change counts. Keep a private record of archive filenames, checksums, and download dates, but do not place licensed raw data in this submission package.

## Documentation references

- Freddie Mac SFLLD access and sample description: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
- Freddie Mac raw field definitions and codes: https://www.freddiemac.com/research/pdf/user_guide.pdf
- FRED UNRATE: https://fred.stlouisfed.org/series/UNRATE
- Chicago Fed NFCI: https://www.chicagofed.org/research/data/nfci/current-data
