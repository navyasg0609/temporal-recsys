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
| Qualifying entries found | 2,817,131 | Kaggle CPU notebook | 2026-10-31 explore | `scripts/kaggle_explore.py` | see LAB_NOTEBOOK |
| Share of the October to December 2005 gap closed by the qualifying set | 17.6% | Kaggle CPU notebook | 2026-10-31 explore | `scripts/kaggle_explore.py` | see LAB_NOTEBOOK |
| New users, October to December 2005 | 88.6% fewer | Kaggle CPU notebook | 2026-10-31 explore | `scripts/kaggle_explore.py` | see LAB_NOTEBOOK |
| Temporal NDCG@10, preference ALS, 12-month window | 0.0478 (0.0458 to 0.0508) | Kaggle CPU notebook, 4 cores, 31.3 GB | baseline-table | `trs baseline` | `b4f46e9` |
| Temporal NDCG@10, popularity, 12-month window | 0.0306 (0.0292 to 0.0318) | Kaggle CPU notebook, 4 cores, 31.3 GB | baseline-table | `trs baseline` | `b4f46e9` |
| Temporal NDCG@10, rating ALS, 12-month window | 0.0072 (0.0057 to 0.0081) | Kaggle CPU notebook, 4 cores, 31.3 GB | baseline-table | `trs baseline` | `b4f46e9` |
| Temporal Recall@10, preference ALS, 12-month window | 0.0561 (0.0539 to 0.0599) | Kaggle CPU notebook, 4 cores, 31.3 GB | baseline-table | `trs baseline` | `b4f46e9` |
| Temporal Recall@10, popularity, 12-month window | 0.0350 (0.0335 to 0.0360) | Kaggle CPU notebook, 4 cores, 31.3 GB | baseline-table | `trs baseline` | `b4f46e9` |
| Temporal Recall@10, rating ALS, 12-month window | 0.0096 (0.0075 to 0.0111) | Kaggle CPU notebook, 4 cores, 31.3 GB | baseline-table | `trs baseline` | `b4f46e9` |
| Mean fit time, preference ALS, NumPy and SciPy | 474.0 s | Kaggle CPU notebook, 4 cores, 31.3 GB | baseline-table | `trs baseline` | `b4f46e9` |
| Mean fit time, rating ALS, NumPy and SciPy | 345.4 s | Kaggle CPU notebook, 4 cores, 31.3 GB | baseline-table | `trs baseline` | `b4f46e9` |
| NDCG@10 ratio, random over temporal, preference ALS, one seed and cutoff | 3.45 | Kaggle CPU notebook, 4 cores, 31.3 GB | baseline-table | `trs baseline` | `b4f46e9` |
| NDCG@10 ratio, random over temporal, popularity, one seed and cutoff | 1.96 | Kaggle CPU notebook, 4 cores, 31.3 GB | baseline-table | `trs baseline` | `b4f46e9` |
| NDCG@10 ratio, random over temporal, rating ALS, one seed and cutoff | 1.13 | Kaggle CPU notebook, 4 cores, 31.3 GB | baseline-table | `trs baseline` | `b4f46e9` |
| NDCG@10, popularity for new users | 0.1133 (0.1115 to 0.1149) | Kaggle CPU notebook, 4 cores, 31.3 GB | baseline-table | `trs baseline` | `b4f46e9` |
| Whole-job training time saved by monthly partitions, 12-month window | No clear saving: ranges overlap, median 3.5% | Kaggle CPU notebook, 4 cores, 31.3 GB, Spark local mode | training-latency | `trs training-latency` | `47d06a7` |
| Whole-job training time saved by monthly partitions, 3-month window | No clear saving: ranges overlap, median 1.2% | Kaggle CPU notebook, 4 cores, 31.3 GB, Spark local mode | training-latency | `trs training-latency` | `47d06a7` |
| Read time saved in training jobs by monthly partitions, 3-month and 12-month windows | 27.0% and 12.8% | Kaggle CPU notebook, 4 cores, 31.3 GB, Spark local mode | training-latency | `trs training-latency` | `47d06a7` |
| Whole-job time on 3 months against 12 months, monthly layout, from fewer rows | 65.1% less | Kaggle CPU notebook, 4 cores, 31.3 GB, Spark local mode | training-latency | `trs training-latency` | `47d06a7` |
| Temporal NDCG@10, Spark preference ALS, 12-month window | 0.0491 (0.0477 to 0.0520) | Kaggle CPU notebook, 4 cores, 31.3 GB, Spark local mode | spark-als | `trs spark-als` | `47d06a7` |
| Temporal Recall@10, Spark preference ALS, 12-month window | 0.0573 (0.0551 to 0.0608) | Kaggle CPU notebook, 4 cores, 31.3 GB, Spark local mode | spark-als | `trs spark-als` | `47d06a7` |
| Temporal NDCG@10, NumPy preference ALS, widened grid | 0.0488 (0.0471 to 0.0517) | Kaggle CPU notebook, 4 cores, 31.3 GB | baseline-table-v2 | `trs baseline` | `47d06a7` |
| Temporal Recall@10, NumPy preference ALS, widened grid | 0.0568 (0.0542 to 0.0602) | Kaggle CPU notebook, 4 cores, 31.3 GB | baseline-table-v2 | `trs baseline` | `47d06a7` |
| Mean fit time, Spark preference ALS, 12-month window | 370.1 s | Kaggle CPU notebook, 4 cores, 31.3 GB, Spark local mode | spark-als | `trs spark-als` | `47d06a7` |
| Mean fit time, NumPy preference ALS, widened grid | 484.8 s | Kaggle CPU notebook, 4 cores, 31.3 GB | baseline-table-v2 | `trs baseline` | `47d06a7` |
| Whole-job time, one Spark fit on 89,313,064 ratings | 676.2 s | Kaggle CPU notebook, 4 cores, 31.3 GB, Spark local mode | spark-full-history | `trs spark-als` | `47d06a7` |
