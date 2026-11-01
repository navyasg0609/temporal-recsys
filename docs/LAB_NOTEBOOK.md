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

## 2026-10-31: Why monthly ratings fall at the end of 2005

**Expected:** the fall comes from the ratings organizers held back as the qualifying set.

**Ran:** `scripts/kaggle_explore.py` on a Kaggle CPU notebook. It found the full qualifying
set of 2,817,131 entries and matched all 1,408,395 probe entries to ratings. Figures for
2005 are in `experiments/results/explore/late_2005.json`.

**What happened:** the expectation was mostly wrong. Ratings fall 53.4% from October to
December. Adding the qualifying set back only reduces that to 41.5%, closing 17.6% of the
gap. New users fall 88.6%, active users fall 21.2%, and ratings per active user fall from
23.06 to 13.65.

| Month | Ratings | Held back share | New users |
|---|---|---|---|
| 2005-08 | 5,261,590 | 3.5% | 21,200 |
| 2005-09 | 4,913,537 | 4.0% | 22,856 |
| 2005-10 | 5,156,753 | 5.5% | 24,372 |
| 2005-11 | 3,605,112 | 12.0% | 8,717 |
| 2005-12 | 2,405,578 | 24.6% | 2,777 |

**Hypothesis, not proven:** the organizers kept only users with enough ratings, so people
who joined near the end are missing. The data fits this, but cannot confirm it.

**What it changes:** test periods stop at October 2005 (decision 016).

## 2026-10-31: Baseline table

**Expected:** see the prediction in `experiments/registered/baseline-table.yaml`.

**Ran:** `trs baseline` on a Kaggle CPU notebook (4 cores, 31.3 GB) at commit `b4f46e9`. All
28 arms finished in one session of about 2 hours 28 minutes. Results are in
`experiments/results/baseline-table/`.

**Chosen on the July tuning cutoff:** rating ALS with `reg=0.05`, and preference ALS with
`reg=10` and `confidence=10`.

**Temporal test, 12-month window, known users. Mean and range across cutoffs and seeds:**

| Model | Arms | NDCG@10 | Recall@10 | Mean fit seconds |
|---|---|---|---|---|
| Preference ALS | 6 | 0.0478 (0.0458 to 0.0508) | 0.0561 (0.0539 to 0.0599) | 474.0 |
| Popularity | 3 | 0.0306 (0.0292 to 0.0318) | 0.0350 (0.0335 to 0.0360) | 0.36 |
| Rating ALS | 6 | 0.0072 (0.0057 to 0.0081) | 0.0096 (0.0075 to 0.0111) | 345.4 |

Popularity needs one arm per cutoff, because it has no random start. The ranges do not
overlap, so the plan's rule for "no clear difference" does not apply. Standard errors across
sampled users are about 0.0002 to 0.0005.

**Random split against temporal split, October 2005 cutoff, seed 1, NDCG@10:**

| Model | Temporal | Random | Ratio |
|---|---|---|---|
| Preference ALS | 0.0464 | 0.1602 | 3.45 |
| Popularity | 0.0306 | 0.0599 | 1.96 |
| Rating ALS | 0.0081 | 0.0092 | 1.13 |

The random split also widens preference ALS's lead over popularity, from 1.52 times at this
cutoff to 2.67 times. This is one seed at one cutoff. It also tests a different group of
users: 365,836 are eligible against 174,666 on the temporal split, because the temporal test
month contains only users active in that month. The ratios therefore combine leakage with
that change of group, and should not be read as leakage alone.

**New users, popularity only:** NDCG@10 0.1133 (0.1115 to 0.1149), Recall@10 0.1040 (0.1003
to 0.1065), over 21,124 to 24,474 users. This cannot be compared with the known-user score.
Known users have already rated many popular films, and those are removed from their lists,
while new users have none removed.

**Prediction check:**

- Preference ALS beats rating ALS: confirmed.
- Rating ALS may lose to popularity: confirmed. Popularity scores 4.26 times higher.
- Popularity is hard to beat: partly confirmed. Preference ALS beats it by 1.57 times on
  average, at a cost of about 474 seconds of fitting against 0.36.
- Random splits score higher for the personalized models: confirmed, but very unevenly.

**Surprises:**

- The preference ALS `reg` setting made no real difference. At `confidence=10`, `reg=1` and
  `reg=10` differ by 0.0000007 in NDCG@10, against a standard error of 0.0005. So the choice
  of `reg=10` is noise. The likely reason is that the penalty is tiny next to the other terms
  in the equation, but this was not tested. The grid effectively tested only `confidence`.
- The better `confidence`, 10, is the lowest value in the grid, so a lower value might score
  higher. Tuning cannot be revisited under this plan.
- Rating ALS is sensitive to its setting: NDCG@10 runs from 0.0034 to 0.0098 across the grid.

**Limits:** one hardware setup, one window length and two seeds. Fit times are single-machine
NumPy and SciPy on Kaggle, and say nothing about Spark. They are not the training latency
from Phase 3. Offline scores favor popular films, and an unbiased test of recommendations
would need live users.

**What it changes:** preference ALS is the ALS baseline from here on (decision 019). Any new
model must beat it. Any future grid must reach past its best value in every direction.

## 2026-10-31: Spark pilot

**Expected:** see `experiments/registered/spark-pilot.yaml`. Spark would be slower than the
422 seconds NumPy preference ALS took on this window.

**Ran:** `trs spark-als` on a Kaggle CPU notebook (4 cores, 31.3 GB, Spark local mode, 20 GB
driver) at commit `12212e1`. One arm on the 12-month tuning window, with 44,209,042 rows.

| Part | Seconds |
|---|---|
| Read and cache | 18.2 |
| Fit | 346.9 |
| Copy factors | 5.7 |

NDCG@10 was 0.0522 and Recall@10 was 0.0604, both on the tuning cutoff.

**What happened:** the prediction was wrong. Spark's fit was faster than NumPy's 422 seconds.
A likely reason is that Spark used all 4 cores while the NumPy model solves one row at a time
in Python, but this was not tested. The two runs came from different sessions, so this is one
rough comparison, not a benchmark.

**Limits:** one run, with no spread. The penalty differs from the NumPy model's. The score is
a check that the model learned something, not a result.

**What it changes:** an arm takes minutes, so the later plans run on Kaggle at full size. The
read was about 4.9% of this job, so the saving from partitioning cannot exceed that share on
this window. `training-latency` measures it directly.

## 2026-11-01: Training latency by storage layout

**Expected:** see `experiments/registered/training-latency.yaml`. The saving would be below
10% on 12 months, larger on 3 months, and well below 35% on both.

**Ran:** `trs training-latency` on a Kaggle CPU notebook (4 cores, 31.3 GB, Spark local mode,
20 GB driver) at commit `47d06a7`. One warm-up pass and three measured passes.

**Whole-job seconds, median and range across the measured passes:**

| Window | Rows | Flat | Monthly | Saving |
|---|---|---|---|---|
| 3 months | 15,120,713 | 143.6 (142.2 to 154.4) | 141.9 (139.8 to 145.3) | 1.2% |
| 12 months | 51,585,504 | 420.8 (396.3 to 451.4) | 406.1 (403.7 to 408.2) | 3.5% |

**Median read seconds:**

| Window | Flat | Monthly | Read saving | Seconds saved | Saved as share of job |
|---|---|---|---|---|---|
| 3 months | 5.37 | 3.92 | 27.0% | 1.45 | 1.0% |
| 12 months | 12.07 | 10.53 | 12.8% | 1.54 | 0.4% |

**Result:** the ranges overlap on both windows, so under the plan's rule there is **no clear
saving** in whole-job time from partitioning. Partitioning does read faster, but reading is
only about 3 to 4% of this job.

The 3.5% should not be quoted as a real saving. Most of it is a difference in fit time, and
the layouts give Spark identical cached rows, so that difference is run-to-run noise. One flat
fit took 434 seconds, against 380 and 402 in the other passes. The read itself saved 0.4%.

**Prediction check:**

- Below 10% on 12 months: confirmed.
- Larger on 3 months: wrong as the plan measured it, 1.2% against 3.5%. Comparing only the read
  saving as a share of the job, it holds at 1.0% against 0.4%.
- Well below 35%: confirmed.

**Surprises:** the read saving is far smaller than the 38 to 77% in the read benchmark. That
benchmark read all columns over different windows, while this job reads two columns and caches
them. The two measure different things, and neither replaces the other. The first monthly read
in the warm-up pass took 23.6 seconds, so a cold start may favor layouts differently.

**Limits:** three measured passes on one machine, with a warm file cache and Spark in local mode.

**What it changes:** the 35% training-latency figure is not supported by measurement, and will
not be claimed. Training on 3 months with the monthly layout took 65.1% less time than on 12
months. That comes from using fewer rows, not from partitioning, and it costs an unmeasured
amount of quality. Measuring that trade is the main experiment.

## 2026-11-01: Spark ALS against NumPy ALS

**Expected:** see `spark-als.yaml` and `baseline-table-v2.yaml` in `experiments/registered/`.

**Ran:** `trs spark-als` and `trs baseline` on a Kaggle CPU notebook (4 cores, 31.3 GB) at commit
`47d06a7`. Spark ran in local mode with a 20 GB driver.

**Chosen on the July tuning cutoff:** Spark with `reg=0.01` and `confidence=3`, and NumPy with
`reg=10` and `confidence=3`. Both `reg` values are at the low edge of their grids.

**Temporal test, 12-month window, known users. Mean and range across cutoffs and seeds:**

| Model | Arms | NDCG@10 | Recall@10 | Mean fit seconds |
|---|---|---|---|---|
| Spark preference ALS | 6 | 0.0491 (0.0477 to 0.0520) | 0.0573 (0.0551 to 0.0608) | 370.1 |
| NumPy preference ALS, version 2 | 6 | 0.0488 (0.0471 to 0.0517) | 0.0568 (0.0542 to 0.0602) | 484.8 |
| NumPy preference ALS, version 1 | 6 | 0.0478 (0.0458 to 0.0508) | 0.0561 (0.0539 to 0.0599) | 474.0 |
| Popularity | 3 | 0.0306 (0.0292 to 0.0318) | 0.0350 (0.0335 to 0.0360) | 0.44 |

**Result:** Spark ALS beats popularity by 1.61 times with no overlap in ranges, which meets the
Phase 3 exit criterion. Spark and NumPy overlap, so there is **no clear difference in quality**.
Spark scored higher in 5 of 6 matched arms, by 0.0003 on average, and that is smaller than the
0.0005 standard error of a single arm. Spark's mean fit was 23.7% shorter, on the same hardware,
for the same rank and iterations.

The wider search raised NumPy's mean from 0.0478 to 0.0488, but the ranges overlap, so that too
is no clear difference. Version 2 was widened after version 1's test scores were known.

**Prediction check:**

- Spark beats popularity with no overlap: confirmed.
- Spark is close to NumPy: confirmed.
- A lower confidence scores somewhat higher: weakly confirmed. On July, 0.0522 at 3 against
  0.0518 at 10, which is within one standard error.
- `reg` has little effect at 10 and 100, and begins to matter at 1000: confirmed. At
  `confidence=3` it gives 0.05224, 0.05224 and 0.0459. This also supports the earlier explanation
  that small penalties were too small to matter.

**Surprises:**

- The pilot's prediction was wrong in a lasting way: Spark is faster than the NumPy code on these
  4 cores, not only in one run. The NumPy model solves rows one at a time in Python, which may be
  why, but this was not tested. Faster NumPy code could reverse this result.
- Repeating the pilot's settings gave the same NDCG@10 to every digit, 0.052234756741122795, with
  fit times of 346.9 and 334.1 seconds. So a seed reproduces the score, while time varies.

**Limits:** Spark in local mode on one machine is not distributed. The penalties differ between
the two models, so this compares related models. Offline scores favor popular films.

## 2026-11-01: One Spark fit on all history

**Expected:** see `experiments/registered/spark-full-history.yaml`. The fit would take about twice
the pilot's 347 seconds.

**Ran:** `trs spark-als` on the same Kaggle setup at commit `47d06a7`, on every rating from November
1999 up to the October 2005 cutoff.

| Rows | Read | Fit | Copy factors | Whole job |
|---|---|---|---|---|
| 89,313,064 | 29.5 s | 640.0 s | 6.6 s | 676.2 s |

**Result:** the prediction held. Rows grew 2.02 times over the pilot and the fit grew 1.84 times.
NDCG@10 was 0.0450, which shows the model learned, but it is not a comparison of history lengths.
Those settings differ from the tuned ones, and that comparison belongs to the main experiment.

**Session limit:** no session ended during any run, but the start and end times of this session
were not noted, so Kaggle's limit stays unmeasured. The arms recorded here add up to 4.7 hours of
timed work, and they may not all have shared one session.

**What it changes:** the largest window takes minutes, so the main experiment can run on Kaggle
(decision 022).

## 2026-11-01: Kaggle GPU and title file check

**Expected:** a free GPU would run PyTorch, and the dataset would include film titles.

**Ran:** `scripts/kaggle_gpu_check.py` on a Kaggle GPU notebook at commit `0219cdf`.

| Item | Result |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128 |
| GPU | 2 Tesla T4, 14.6 GB each |
| CPU count | 4 |
| Memory | 31.3 GB |
| Free disk | 19.5 GB |
| 4,096 by 4,096 matrix product | 0.190 s |
| Title lines matched | 17,770 of 17,770 |
| Titles containing a comma | 336 |
| Missing years | 7 |
| Year range | 1896 to 2005 |

**What happened:** both expectations held. The notebook has the same CPU count and memory as the
CPU notebooks used for ALS. One matrix product is a working check, not a benchmark.

**Not measured:** Kaggle's weekly GPU allowance.

**What it changes:** the model uses one T4, titles and release years are available, and PyTorch
stays an optional extra so that Kaggle keeps its own GPU build.

## 2026-11-01: Two-tower pilot

**Expected:** see `experiments/registered/two-tower-pilot.yaml`. The arm would finish within an
hour and score well above zero.

**Ran:** `trs two-tower` on a Kaggle GPU notebook (one Tesla T4, 4 cores, 31.3 GB) at commit
`85444be`. One untuned arm on the 12-month tuning window, with 44,209,042 rows and 3 epochs.

**What happened:** both expectations held. Training took 376.3 seconds. NDCG@10 was 0.1123 and
Recall@10 was 0.1319, on the tuning cutoff.

**Why this is not accepted yet:** the best ALS score on this cutoff was 0.0528. An untuned model
at about twice that is more likely to reveal a flaw than a breakthrough, and the random split in
the baseline table shows how easily this metric inflates. No full plan is registered until the
gain is explained. `experiments/registered/two-tower-diagnostic.yaml` tests the explanations.
