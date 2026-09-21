# Paths and local setup

## Quick reproduction

When `FREDDIE_BASE` is not set, `07_results.py` uses the aggregate files distributed in `replication_inputs/` and writes the regenerated final outputs to `processed/07_results/`:

```bash
python 07_results.py
```

No Freddie Mac loan-level file is required for this command.

## Complete source-data reproduction

All scripts read their private data root from the `FREDDIE_BASE` environment variable. Keep the licensed source data and large generated files outside the shared repository.

```text
${FREDDIE_BASE}/
├── raw_zip/             # Freddie Mac sample archives
├── raw_txt/             # extracted origination and performance files
├── macro/
│   ├── UNRATE.csv
│   └── NFCI.csv
├── processed/           # generated analysis files
└── logs/                # optional local logs

/path/to/freddie_replication/    # shared code and aggregate inputs
```

Set the private root and install the included macro snapshots:

```bash
export FREDDIE_BASE="$HOME/freddiemac_sflld"
mkdir -p "$FREDDIE_BASE/macro"
cp /path/to/freddie_replication/macro/UNRATE.csv "$FREDDIE_BASE/macro/"
cp /path/to/freddie_replication/macro/NFCI.csv "$FREDDIE_BASE/macro/"
```

Place `sample_YYYY.zip`, or the corresponding extracted `sample_orig_YYYY.txt` and `sample_svcg_YYYY.txt` files, under the private root for each vintage from 1999 through 2019.

Do not place `raw_zip/`, `raw_txt/`, Parquet files, logs, or the generated `processed/` directory in the shared repository.
