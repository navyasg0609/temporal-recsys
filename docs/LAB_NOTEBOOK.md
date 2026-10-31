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

## 2026-10-31: Full ingest and read benchmark

**Expected:** see the prediction in `experiments/registered/read-benchmark.yaml`.

**Ran:** `trs ingest` and `trs benchmark` on a Kaggle CPU notebook (4 cores, 31.3 GB) at
commit `45f72d1`, with the benchmark run twice in one session.

**Ingest:** the monthly data matched the source exactly: 100,480,507 ratings, 17,770
films and the expected date range, with no duplicate ratings. Sizes on disk were
533 MB flat, 472 MB monthly and 315 MB date sorted.

**Median read time in seconds, and saving against flat:**

| Window | Run | Flat | Date sorted | Monthly |
|---|---|---|---|---|
| 3 months | 1 | 2.076 | 1.152 (44.5%) | 0.484 (76.7%) |
| 3 months | 2 | 2.328 | 1.154 (50.4%) | 0.568 (75.6%) |
| 12 months | 1 | 3.394 | 2.719 (19.9%) | 2.095 (38.3%) |
| 12 months | 2 | 3.580 | 2.686 (25.0%) | 2.091 (41.6%) |

**Median aggregate time in seconds, and saving against flat:**

| Window | Run | Flat | Date sorted | Monthly |
|---|---|---|---|---|
| 3 months | 1 | 1.524 | 0.918 (39.8%) | 0.543 (64.4%) |
| 3 months | 2 | 1.817 | 0.946 (47.9%) | 0.555 (69.5%) |
| 12 months | 1 | 2.437 | 1.986 (18.5%) | 1.564 (35.8%) |
| 12 months | 2 | 2.699 | 2.157 (20.1%) | 1.704 (36.9%) |

All layouts matched the same rows: 11,167,443 for 3 months and 53,250,073 for 12
months. Monthly and date-sorted ranges did not overlap the flat baseline in any arm.
Monthly and date-sorted ranges did overlap once, for the 12-month read in run 1, when
one monthly pass took 2.786 s.

**Prediction check:**

- Partitioning beats flat Parquet on both windows: confirmed.
- The saving is smaller on the 12-month window: confirmed.
- Date sorting recovers much of the benefit: partly confirmed. It saved about 45 to
  50% against about 76% on 3 months, and 20 to 25% against about 40% on 12 months.

**Surprises:**

- Date sorting also made the files 41% smaller than flat, more than partitioning did.
- 2005 holds 53.0% of all ratings, so 12 months is over half the data. This is why the
  saving falls so sharply as the window grows. A 24-month window would hold 83.1%.
- Monthly ratings fall by more than half from October to December 2005. Check this
  before choosing test periods.
- The first arm of run 2 took 22.9 s, far above the rest. It was in the excluded
  warm-up pass, which is the reason for having one.

**Limits:** absolute times are seconds, on a warm file cache, so the savings are small
in absolute terms. These are read times, not training
times. Model training will decide how much partitioning matters overall, and that is
measured in Phase 3. Both runs shared a session, so this shows repeatability, not
stability across machines. Ingest duration was not recorded.

**What it changes:** results must report rows used alongside window length, and Phase 2
must look at the late-2005 drop before fixing test periods.
