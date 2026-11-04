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
| Temporal NDCG@10, two-tower model, 12-month window | 0.1023 (0.0909 to 0.1150) | Kaggle GPU notebook, one Tesla T4, 4 cores, 31.3 GB | two-tower | `trs two-tower` | `4c5a1ec` |
| Temporal Recall@10, two-tower model, 12-month window | 0.1197 (0.1082 to 0.1320) | Kaggle GPU notebook, one Tesla T4, 4 cores, 31.3 GB | two-tower | `trs two-tower` | `4c5a1ec` |
| Temporal NDCG@10, popularity over the last month | 0.0806 (0.0680 to 0.1052) | Kaggle GPU notebook, one Tesla T4, 4 cores, 31.3 GB | two-tower | `trs two-tower` | `4c5a1ec` |
| Temporal Recall@10, popularity over the last month | 0.0923 (0.0771 to 0.1148) | Kaggle GPU notebook, one Tesla T4, 4 cores, 31.3 GB | two-tower | `trs two-tower` | `4c5a1ec` |
| Two-tower against 1-month popularity | No clear difference: ranges overlap | Kaggle GPU notebook, one Tesla T4, 4 cores, 31.3 GB | two-tower | `trs two-tower` | `4c5a1ec` |
| Two-tower NDCG@10 over Spark ALS | 2.08 times, no overlap | Kaggle GPU notebook, one Tesla T4, 4 cores, 31.3 GB, ALS on a CPU notebook of the same size | two-tower, spark-als | `trs two-tower` | `4c5a1ec` |
| 1-month popularity NDCG@10 over Spark ALS | 1.64 times, no overlap | Kaggle GPU notebook, one Tesla T4, 4 cores, 31.3 GB, ALS on a CPU notebook of the same size | two-tower, spark-als | `trs two-tower` | `4c5a1ec` |
| Mean training time, two-tower model, 12-month window | 424.2 s | Kaggle GPU notebook, one Tesla T4, 4 cores, 31.3 GB | two-tower | `trs two-tower` | `4c5a1ec` |
| Training time, one two-tower run on 89,313,064 ratings | 783.7 s | Kaggle GPU notebook, one Tesla T4, 4 cores, 31.3 GB | two-tower-full-history | `trs two-tower` | `4c5a1ec` |
| NDCG@10, Recall@10, rows and training seconds for every model and window | `experiments/results/history-*/summary.json`, tables in LAB_NOTEBOOK 2026-11-02 | Popularity and Spark ALS: Kaggle CPU notebook, 4 cores, 31.3 GB, Spark local mode. Two-tower: Kaggle GPU notebook, one Tesla T4, 4 cores, 31.3 GB | history-windows | `trs history` | `2fe0302` |
| History needed, popularity | 1 month | Kaggle CPU notebook, 4 cores, 31.3 GB | history-windows | `trs history --model popularity` | `2fe0302` |
| History needed, Spark ALS | 1 month | Kaggle CPU notebook, 4 cores, 31.3 GB, Spark local mode | history-windows | `trs history --model spark-als` | `2fe0302` |
| History needed, two-tower model | 6 months | Kaggle GPU notebook, one Tesla T4, 4 cores, 31.3 GB | history-windows | `trs history --model two-tower` | `2fe0302` |
| NDCG@10, popularity, 1-month window | 0.0837 (0.0697 to 0.1096) | Kaggle CPU notebook, 4 cores, 31.3 GB | history-windows | `trs history --model popularity` | `2fe0302` |
| NDCG@10, Spark ALS, 1-month window | 0.0793 (0.0690 to 0.0918) | Kaggle CPU notebook, 4 cores, 31.3 GB, Spark local mode | history-windows | `trs history --model spark-als` | `2fe0302` |
| NDCG@10, Spark ALS, all history | 0.0471 (0.0462 to 0.0488) | Kaggle CPU notebook, 4 cores, 31.3 GB, Spark local mode | history-windows | `trs history --model spark-als` | `2fe0302` |
| NDCG@10, two-tower model, 6-month window | 0.1007 (0.0906 to 0.1120) | Kaggle GPU notebook, one Tesla T4, 4 cores, 31.3 GB | history-windows | `trs history --model two-tower` | `2fe0302` |
| NDCG@10, two-tower model, 12-month window | 0.1028 (0.0918 to 0.1149) | Kaggle GPU notebook, one Tesla T4, 4 cores, 31.3 GB | history-windows | `trs history --model two-tower` | `2fe0302` |
| Spark ALS NDCG@10, 1 month over all history | 1.68 times | Kaggle CPU notebook, 4 cores, 31.3 GB, Spark local mode | history-windows | `trs history --model spark-als` | `2fe0302` |
| Spark ALS training time, all history over 1 month | 8.68 times (613.7 s against 70.7 s) | Kaggle CPU notebook, 4 cores, 31.3 GB, Spark local mode | history-windows | `trs history --model spark-als` | `2fe0302` |
| Two-tower NDCG@10, 6 months as a share of 12 months | 98.0% | Kaggle GPU notebook, one Tesla T4, 4 cores, 31.3 GB | history-windows | `trs history --model two-tower` | `2fe0302` |
| Two-tower training time, 12 months over 6 months | 1.83 times (424.4 s against 231.6 s) | Kaggle GPU notebook, one Tesla T4, 4 cores, 31.3 GB | history-windows | `trs history --model two-tower` | `2fe0302` |
| Two-tower against Spark ALS and popularity, same window | Higher at every window and test month (paired rule) | Two-tower on a Kaggle GPU notebook, the others on a CPU notebook of the same size | history-windows | `trs history` | `2fe0302` |
| Smallest two-tower margin, over popularity, 1 month, August 2005 | 0.0038 | Two-tower on a Kaggle GPU notebook, popularity on a CPU notebook of the same size | history-windows | `trs history` | `2fe0302` |
| Spark ALS against popularity, 1-month window | No clear difference: popularity higher in August, ALS higher in October | Kaggle CPU notebook, 4 cores, 31.3 GB, Spark local mode | history-windows | `trs history` | `2fe0302` |
| Spark ALS against popularity, 3 months and longer | Higher at every window and test month (paired rule) | Kaggle CPU notebook, 4 cores, 31.3 GB, Spark local mode | history-windows | `trs history` | `2fe0302` |
| Ratings in the scaling window, 12 months before August 2005 | 46,691,107 | Kaggle CPU notebook, AMD EPYC 7B12, 2 physical cores with 2 threads each, 31.3 GB | scaling | `trs scaling` | `1ee6fdd` |
| Whole-job seconds, NumPy ALS, 1 thread | 351.4 (350.8 to 358.1) | Kaggle CPU notebook, AMD EPYC 7B12, 2 physical cores with 2 threads each, 31.3 GB | scaling | `trs scaling` | `1ee6fdd` |
| Whole-job seconds, Spark ALS, 1 core | 662.0 (646.9 to 716.5) | Kaggle CPU notebook, AMD EPYC 7B12, 2 physical cores with 2 threads each, 31.3 GB, Spark local mode | scaling | `trs scaling` | `1ee6fdd` |
| Whole-job seconds, Spark ALS, 2 cores | 362.5 (359.8 to 365.2) | Kaggle CPU notebook, AMD EPYC 7B12, 2 physical cores with 2 threads each, 31.3 GB, Spark local mode | scaling | `trs scaling` | `1ee6fdd` |
| Whole-job seconds, Spark ALS, 4 threads | 323.6 (302.0 to 327.2) | Kaggle CPU notebook, AMD EPYC 7B12, 2 physical cores with 2 threads each, 31.3 GB, Spark local mode | scaling | `trs scaling` | `1ee6fdd` |
| Spark ALS speedup over 1 core | 1.83 times on 2 cores, 2.05 times on 4 threads | Kaggle CPU notebook, AMD EPYC 7B12, 2 physical cores with 2 threads each, 31.3 GB, Spark local mode | scaling | `trs scaling` | `1ee6fdd` |
| Spark ALS on 1 core against NumPy ALS on 1 thread | 1.88 times as long | Kaggle CPU notebook, AMD EPYC 7B12, 2 physical cores with 2 threads each, 31.3 GB, Spark local mode | scaling | `trs scaling` | `1ee6fdd` |
| COST of Spark ALS against single-threaded NumPy ALS | 4 threads, the whole machine. Median 7.9% lower than NumPy, no overlap | Kaggle CPU notebook, AMD EPYC 7B12, 2 physical cores with 2 threads each, 31.3 GB, Spark local mode | scaling | `trs scaling` | `1ee6fdd` |
| Spark ALS on 2 cores against NumPy ALS | 3.2% slower, no overlap | Kaggle CPU notebook, AMD EPYC 7B12, 2 physical cores with 2 threads each, 31.3 GB, Spark local mode | scaling | `trs scaling` | `1ee6fdd` |
| NDCG@10 of the served two-tower arm, retrained | 0.0993, against 0.0994 in Phase 4 | Kaggle GPU notebook, one Tesla T4, 4 cores, 31.3 GB | serving-export | `trs export-serving` | `aa6d52b` |
| Served lists equal to the evaluation's lists | 5,000 of 5,000 sampled users | Kaggle GPU notebook, one Tesla T4, 4 cores, 31.3 GB | serving-export | `trs export-serving` | `aa6d52b` |
| Serving bundle size | 566,831,262 bytes | Kaggle GPU notebook, one Tesla T4, 4 cores, 31.3 GB | serving-export | `trs export-serving` | `aa6d52b` |
| Serving image size | 299 MB | Laptop, Intel Core i7-1355U, 15.7 GB, Docker Desktop with 12 CPUs and 7.6 GiB, kind cluster of 3 nodes on one machine | phase 6 deployment | `docker build -f deploy/Dockerfile` | `b123065` |
| Memory per service copy with the real bundle | 590 MiB | Laptop, Intel Core i7-1355U, 15.7 GB, Docker Desktop with 12 CPUs and 7.6 GiB, kind cluster of 3 nodes on one machine | phase 6 deployment | `deploy/cluster.sh` | `726604c` |
