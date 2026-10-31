# Claims ledger

Every number published in the README, report or release notes has a row here. If a
number is not in this table, it is not published.

| Claim | Value | Hardware | Run ID | Command | Commit |
|---|---|---|---|---|---|
| Ratings in the dataset | 100,480,507 | Kaggle CPU notebook | 2026-10-31 environment check | `scripts/kaggle_env_check.py` | `f2663a0` |
| Films in the dataset | 17,770 | Kaggle CPU notebook | 2026-10-31 environment check | `scripts/kaggle_env_check.py` | `f2663a0` |
| Rating date range | 1999-11-11 to 2005-12-31 | Kaggle CPU notebook | 2026-10-31 environment check | `scripts/kaggle_env_check.py` | `f2663a0` |
| Users in the dataset | 480,189 | Kaggle CPU notebook | 2026-10-31 ingest | `trs ingest --check-netflix` | `45f72d1` |
| Duplicate ratings | 0 | Kaggle CPU notebook | 2026-10-31 ingest | `trs ingest --check-netflix` | `45f72d1` |
| Share of ratings from 2005 | 53.0% | Kaggle CPU notebook | 2026-10-31 ingest | `trs ingest --check-netflix` | `45f72d1` |
| Read time saved by monthly partitions, 3-month window, against flat Parquet | 75.6% to 76.7% | Kaggle CPU notebook, 4 cores, 31.3 GB | read-benchmark runs 1 and 2 | `trs benchmark` | `45f72d1` |
| Read time saved by monthly partitions, 12-month window, against flat Parquet | 38.3% to 41.6% | Kaggle CPU notebook, 4 cores, 31.3 GB | read-benchmark runs 1 and 2 | `trs benchmark` | `45f72d1` |
| Read time saved by date sorting, 3-month window, against flat Parquet | 44.5% to 50.4% | Kaggle CPU notebook, 4 cores, 31.3 GB | read-benchmark runs 1 and 2 | `trs benchmark` | `45f72d1` |
| Read time saved by date sorting, 12-month window, against flat Parquet | 19.9% to 25.0% | Kaggle CPU notebook, 4 cores, 31.3 GB | read-benchmark runs 1 and 2 | `trs benchmark` | `45f72d1` |
