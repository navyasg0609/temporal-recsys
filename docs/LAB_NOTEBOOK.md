# Lab notebook

Dated notes on what was expected, what happened and what it changes. Newest last.

## 2026-10-31: Kaggle environment and dataset check

**Expected:** Spark would read the Netflix Prize files on a free Kaggle CPU notebook.
Third-party sources gave 100,480,507 ratings and 17,770 films.

**Ran:** `scripts/kaggle_env_check.py` at commit `f2663a0`.

| Item | Result |
|---|---|
| CPU count | 4 |
| Memory | 31.3 GB |
| Free disk | 19.5 GB |
| Python | 3.13.15 |
| PySpark | 4.0.4 |
| Rating files | 4 files, 1.93 GB |
| Lines | 100,498,277 |
| Ratings | 100,480,507 |
| Film headers | 17,770 |
| Unrecognized lines | 0 |
| Date range | 1999-11-11 to 2005-12-31 |
| Spark start | 9.8 s |
| Full scan | 47.3 s |

**What happened:** the counts matched exactly, and every line was recognized. The
timings come from one run and are a feasibility check, not a benchmark.

**Not measured:** the session time limit and any GPU allowance.

**What it changes:** the dataset is confirmed, and a full read is fast enough for
Phase 1. The project moves to Python 3.13 to match Kaggle (decision 011).
