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

## 2026-11-01: Why the two-tower pilot scored so high

**Expected:** see `experiments/registered/two-tower-diagnostic.yaml`. Recency would explain most
of the gain, with no drop from removing unseen films or title and year features.

**Ran:** `trs two-tower-diagnostic` on a Kaggle GPU notebook (one Tesla T4, 4 cores, 31.3 GB) at
commit `f323dfe`, on the July 2005 tuning cutoff only, with one seed. Every arm shares the same
50,000 users and removes the same seen films. Training ended on 2005-06-30, and the test began on
2005-07-01.

| Arm | NDCG@10 | Standard error |
|---|---|---|
| Popularity, 12 months | 0.0337 | 0.0004 |
| Preference ALS, 12 months | 0.0522 | 0.0005 |
| Popularity, 3 months | 0.0711 | 0.0006 |
| Preference ALS, 1 month | 0.0725 | 0.0006 |
| Preference ALS, 3 months | 0.0769 | 0.0006 |
| Popularity, 1 month | 0.0899 | 0.0007 |
| Two-tower, IDs only | 0.1121 | 0.0008 |
| Two-tower, all features | 0.1135 | 0.0008 |
| Two-tower, films absent from training removed | 0.1136 | 0.0008 |

**What happened:** the prediction held, and the gain is not a catalog leak. Removing the 637 films
absent from training changed nothing, and those films hold only 0.03% of relevant test ratings.
Title and year features added little, 0.1135 against 0.1121 on one seed.

Most of the gap is recency. Popularity trained on the last month alone scored 2.66 times its
12-month score. That is higher than preference ALS on any window tested here, and much simpler.
The two-tower model still scored 1.26 times higher than 1-month popularity.

**This corrects an earlier conclusion.** The baseline table compared models with 12-month
popularity only. Preference ALS beat that baseline, but on this cutoff it loses clearly to a
popularity count over the last month. Beating popularity was a weaker test than it looked, so later
results also report recent popularity.

**Also seen:** the same seed gave 0.1123 in the pilot and 0.1135 here, so GPU training is not
exactly repeatable. NumPy ALS repeated its earlier score to every digit.

**Hypothesis, not proven:** July's ratings concentrate on recent releases, which the service also
promotes, so recency may partly reflect what people were shown.

**Limits:** one seed on one cutoff. A short-window model gives zero scores to users with no
ratings in that window, which lowers its score.

**What it changes:** the two-tower plan adds 1-month popularity on the test cutoffs, and uses 3
tuning settings to keep this version basic. The result also points to a strong answer to the main
question, which Phase 4 must test properly.

## 2026-11-02: Two-tower model on the temporal split

**Expected:** see `experiments/registered/two-tower.yaml`. The model would beat Spark ALS and
1-month popularity with no overlap in ranges, and 1-month popularity would beat 12-month ALS.

**Ran:** `trs two-tower` on a Kaggle GPU notebook (one Tesla T4, 4 cores, 31.3 GB) at commit
`4c5a1ec`. ALS results come from a CPU notebook of the same size, so times are not compared.

**Chosen on the July tuning cutoff:** `temperature=0.1`, the top of the grid. It led
`temperature=0.05` by 0.000005, far inside the 0.0008 standard error, so the choice is noise.

**Temporal test, known users. Mean and range across cutoffs and seeds:**

| Model | Arms | NDCG@10 | Recall@10 |
|---|---|---|---|
| Two-tower | 6 | 0.1023 (0.0909 to 0.1150) | 0.1197 (0.1082 to 0.1320) |
| Popularity, last month | 3 | 0.0806 (0.0680 to 0.1052) | 0.0923 (0.0771 to 0.1148) |
| Spark preference ALS | 6 | 0.0491 (0.0477 to 0.0520) | 0.0573 (0.0551 to 0.0608) |
| NumPy preference ALS, version 2 | 6 | 0.0488 (0.0471 to 0.0517) | 0.0568 (0.0542 to 0.0602) |
| Popularity, 12 months | 3 | 0.0306 (0.0292 to 0.0318) | 0.0350 (0.0335 to 0.0360) |

**NDCG@10 by cutoff, with the two seeds shown for the two-tower model:**

| Test month | Two-tower | Popularity, last month |
|---|---|---|
| August 2005 | 0.1150, 0.1129 | 0.1052 |
| September 2005 | 0.0909, 0.0917 | 0.0680 |
| October 2005 | 0.1020, 0.1012 | 0.0686 |

**Prediction check:**

- The two-tower model beats Spark ALS with no overlap: confirmed, by 2.08 times.
- The two-tower model beats 1-month popularity with no overlap: **not confirmed**. The ranges
  overlap, so the plan's rule says no clear difference. The model did score higher in every
  month, but that was not the registered test. The overlap comes from months differing, and only
  three months were tested.
- 1-month popularity beats 12-month ALS: confirmed, by 1.64 times with no overlap.

**What happened:** a popularity count over the last month scored 2.64 times higher than the same
count over 12 months, and beat both tuned 12-month ALS models. The two-tower model beat ALS, which
meets decision 010's requirement, but it is not clearly better than the simple recent baseline.

**Limits:** three test months and two seeds. The grid was smaller than ALS's. The model's input
stops at each user's latest 50 films, whatever the training window. Offline scores favor popular
films, and what the service promoted may explain part of the recency effect, which was not tested.

**What it changes:** the main experiment reports recent popularity next to every model
(decision 029).

## 2026-11-02: One two-tower run on all history

**Expected:** see `experiments/registered/two-tower-full-history.yaml`. Training would take about
twice the pilot's 376 seconds.

**Ran:** `trs two-tower` on the same setup at commit `4c5a1ec`, on 89,313,064 ratings.

**What happened:** the prediction held. Training took 783.7 seconds, 2.08 times the pilot, on 2.02
times the rows. NDCG@10 was 0.1030 with untuned settings. This is a timing check and does not
compare history lengths.

**Not measured:** Kaggle's weekly GPU allowance and session limit. None of these runs reached them.

## 2026-11-02: How much history each model needs

**Expected:** see `experiments/registered/history-windows.yaml`. Popularity would do best on 1
month and Spark ALS on 3 months or less. The two-tower model would need more history than ALS. No
model would gain beyond 12 months, and the two-tower model would score highest at its best window.

**Ran:** `trs history` at commit `2fe0302`, 90 arms: 6 windows × 3 test months, with 2 seeds for
the trained models. Popularity and Spark ALS ran on a Kaggle CPU notebook (4 cores, 31.3 GB, Spark
local mode), and the two-tower model on a Kaggle GPU notebook (one Tesla T4, same CPU and memory).
The ALS session stopped after 35 arms because the notebook sat idle. Rerunning the cell kept those
35 and trained only the last one, so the restart logic worked on a real run.

**NDCG@10, mean and range across test months and seeds.** Rows are the mean training rows across
the three test months:

| Months | Rows | Popularity | Spark ALS | Two-tower |
|---|---|---|---|---|
| 1 | 5.0M | **0.0837** (0.0697 to 0.1096) | **0.0793** (0.0690 to 0.0918) | 0.0913 (0.0774 to 0.1145) |
| 3 | 15.1M | 0.0609 (0.0488 to 0.0722) | 0.0736 (0.0685 to 0.0816) | 0.0968 (0.0861 to 0.1082) |
| 6 | 28.2M | 0.0500 (0.0455 to 0.0537) | 0.0620 (0.0586 to 0.0672) | **0.1007** (0.0906 to 0.1120) |
| 12 | 49.2M | 0.0334 (0.0326 to 0.0340) | 0.0527 (0.0508 to 0.0560) | 0.1028 (0.0918 to 0.1149) |
| 24 | 71.8M | 0.0308 (0.0300 to 0.0320) | 0.0489 (0.0476 to 0.0505) | 0.1026 (0.0914 to 0.1136) |
| all | 84.3M | 0.0305 (0.0292 to 0.0317) | 0.0471 (0.0462 to 0.0488) | 0.1010 (0.0902 to 0.1108) |

Bold marks the history each model needs under the registered rule: the shortest window within 5%
of the model's best window at every test month.

**Training seconds, mean and range.** The hardware differs, so compare times only within a model:

| Months | Spark ALS, CPU | Two-tower, GPU |
|---|---|---|
| 1 | 70.7 (64.8 to 85.5) | 29.7 (28.1 to 32.0) |
| 3 | 144.8 (137.9 to 163.3) | 111.2 (107.4 to 113.9) |
| 6 | 236.6 (225.1 to 242.5) | 231.6 (228.9 to 233.3) |
| 12 | 383.0 (361.6 to 414.4) | 424.4 (406.7 to 440.0) |
| 24 | 529.6 (496.9 to 561.4) | 632.5 (598.8 to 664.4) |
| all | 613.7 (574.0 to 651.6) | 740.7 (698.3 to 781.8) |

Figure: `report/figures/history_windows.png`, drawn by `report/figures/history_windows.py` from the
committed result files.

**Prediction check:** all five predictions held.

- Popularity does best on 1 month: confirmed.
- Spark ALS does best on 3 months or less: confirmed. 1 month is best, and every added window
  scores lower at every test month.
- The two-tower model needs more history than ALS: confirmed, 6 months against 1. One and 3
  months fall below the 95% line in September.
- No model gains beyond 12 months: confirmed. Popularity and ALS keep falling, and the two-tower
  model is flat from 12 to 24 months and slightly lower on all history.
- The two-tower model scores highest at its best window: confirmed by the paired rule. At every
  window it is higher than ALS and popularity at every test month. Its smallest margin is 0.0038,
  over popularity on 1 month in August, more than four times either arm's standard error of 0.0008.

**Paired comparisons:** Spark ALS beats popularity at every window from 3 months up. At 1 month it
does not: popularity is higher in August, ALS in October, and September is nearly a tie.

**What happened:** more history made the two simple models worse. ALS on 1 month scored 1.68 times
its all-history score, and all history took 8.68 times as long to train (613.7 against 70.7 seconds). The
two-tower model works differently: it improved up to 6 months and then stayed flat. Its 6-month
score is 98.0% of its 12-month score, and 12 months took 1.83 times as long to train.

**Against earlier results:** Phase 3b found no clear difference between the two-tower model and
1-month popularity, using the range rule. This plan's paired rule finds the two-tower model higher
at every test month. Both statements hold under their own rules. The plan explains why the rule
changed: the ranges mostly measure how hard each month is. Scores are slightly higher than in
earlier plans (Spark ALS on 12 months: 0.0527 against 0.0491), because every window now removes
all films rated before the cutoff and gives users missing from the window popular films
(decisions 030 and 031).

**Limits:** three test months, all in late 2005, and two seeds. Settings were tuned on 12 months,
which may favor that window, and fixed epochs give short windows fewer training steps. The
two-tower input stops at 50 films per user. The breakdown by user tenure and film popularity was
not run (decision 032). Recency may partly reflect what the service promoted, which this data
cannot separate.

**What it changes:** the answer to the main question depends on the model. A recent window is
enough, and better, for popularity and ALS. The two-tower model needs about 6 months, and history
beyond 12 months buys nothing for any model while training time keeps growing with rows.

## 2026-11-04: How Spark ALS scales with cores, and its COST

**Expected:** see `experiments/registered/scaling.yaml`. Speedup would be below linear: 1.5 to 2.0
times on 2 cores and less than 3.0 times on 4. Spark on 1 core would be slower than single-threaded
NumPy ALS, Spark on 2 cores would not clearly beat it, and COST would be 4.

**Ran:** `trs scaling` at commit `1ee6fdd` on a Kaggle CPU notebook, 31.3 GB, Spark local mode. The
12 months before August 2005, 46,691,107 ratings. Each of the 16 measurements ran in a fresh
process: 1 warm-up pass, then 3 timed passes in shuffled order. Every arm read the same rows, and the
summary rebuilt from `measurements.jsonl` matches the one printed on Kaggle.

**Hardware, from `lscpu`:** AMD EPYC 7B12, 4 CPUs, made of 2 physical cores with 2 threads each.
So the step from 2 to 4 cores adds threads that share a physical core, not new cores.

**Whole-job seconds (read, fit, copy factors), median and range of 3 timed runs:**

| Arm | Seconds | Speedup over Spark on 1 core |
|---|---|---|
| NumPy ALS, 1 thread | 351.4 (350.8 to 358.1) | |
| Spark ALS, 1 core | 662.0 (646.9 to 716.5) | 1.00 |
| Spark ALS, 2 cores | 362.5 (359.8 to 365.2) | 1.83 |
| Spark ALS, 4 cores | 323.6 (302.0 to 327.2) | 2.05 |

Figure: `report/figures/scaling.png`, drawn by `report/figures/scaling.py` from the committed
result files.

**Prediction check:** all five predictions held.

- 2 cores give 1.5 to 2.0 times: confirmed, 1.83 times.
- 4 cores give less than 3.0 times: confirmed, 2.05 times.
- Spark on 1 core is slower than NumPy: confirmed, 1.88 times as long.
- Spark on 2 cores does not clearly beat NumPy: confirmed. It was 3.2% slower, and its fastest run
  (359.8 s) was slower than NumPy's slowest (358.1 s).
- COST is 4: confirmed. On 4 threads Spark's slowest run (327.2 s) beat NumPy's fastest (350.8 s),
  and its median was 7.9% lower.

**What happened:** Spark scaled well across the 2 physical cores, reaching 1.83 of a possible 2.
The 2 extra threads on the same cores added only 1.12 times more. Spark's overhead is large: on 1
core it took 1.88 times as long as a plain single-threaded program doing the same job. It needed
every thread this machine has to win, and then it won by 7.9%.

**An observation, not tested:** held to one thread, NumPy ALS fit in a median of 350.0 s. In
`baseline-table-v2`, the same settings on the same 46,691,107 rows, with no thread limit, fit in
454.9 and 465.3 s on the same kind of notebook. Extra BLAS threads may slow the many tiny matrix
products. If so, the Phase 3 finding that Spark fit 23.7% faster than NumPy (decision 023) partly
reflects that slowdown. The runs came from different sessions and were not interleaved, so this is
not a controlled comparison. Decision 023 stands on its quality result.

**Limits:** one window, one cutoff and one machine. Spark in local mode runs threads in one process,
so this says nothing about a cluster, where network transfer adds cost. NumPy ALS is a plain Python
loop, and an optimized single-threaded program would raise COST. Each engine used its own tuned
settings. Starting a process and a Spark session was not timed.

**What it changes:** on this machine, Spark is not worth its overhead for this job. Its single-core
cost is almost twice a simple program's, and it only pulls ahead when given all 4 threads. The
reason to keep it is the code it shares with a real cluster, not speed on one machine
(decisions 033 and 034).

## 2026-11-04: Serving the scored two-tower model

**Expected:** the export retrains the arm the main experiment scored, the two-tower model on the 6
months before October 2005 with seed 1 (decision 037), and lands within its standard error of the
Phase 4 score. Every served list matches the list the evaluation ranks.

**Ran:** `trs export-serving --config configs/serving.yaml` at commit `aa6d52b` on a Kaggle GPU
notebook (one Tesla T4, 4 cores, 31.3 GB). The manifest is in `experiments/results/serving-export/`.
The bundle itself holds rating data, so it is not committed (decision 038).

**What happened:** the retrained arm scored NDCG@10 0.0993, against 0.0994 in Phase 4, a difference
of 0.00004 with a standard error of 0.0008. It trained on the same 28,970,031 rows in 228.2 seconds,
against 232.6. For 5,000 sampled users, the service's top 10 equalled the evaluation's top 10 every
time. The bundle covers 377,916 users with model vectors and 444,323 with any rating before the
cutoff, and is 566,831,262 bytes.

**Deployment:** on a laptop (Intel Core i7-1355U, 10 cores and 12 threads, 15.7 GB; Docker Desktop
with 12 CPUs and 7.6 GiB), `deploy/cluster.sh` built the 299 MB image, created a three-node kind
cluster, and placed one copy of the service on each worker. A pod inside the cluster got model
recommendations for a user from the window and popular films for an unknown user. Each copy held
590 MiB of memory with the real bundle. CI runs the same deployment with a synthetic bundle and
checks two planted answers on every pull request.

**Dead ends:** the first CI runs failed twice. Scripts committed from Windows lacked the executable
bit. Then the rule spreading copies across nodes also counted the control-plane node, which takes no
pods, so a rolling restart could not place its extra copy. Honouring node taints fixed it.

**Limits:** the kind nodes are containers on one laptop, so the cluster tests scheduling and health
checks, not separate machines. No latency or load numbers are claimed here; Phase 7 measures them.

**What it changes:** the served model is the measured one, so Phase 7 can test a service whose
quality is already known.

## 2026-11-04: Load test pilot, before registering

These are pilot runs of up to 20 users, used to fix the test and set the predictions. None of their
numbers are claims.

**Lockstep users:** with a fixed 1-second pause between requests, every virtual user fired at the
same moment, so requests queued behind each other. The median latency at 10 users was about 30 ms.
A random pause of 0.5 to 1.5 seconds spread them out, and the median fell to about 16 ms. The
registered test uses the random pause.

**Too many BLAS threads:** the service's own timing put the median request at about 17 ms. NumPy
starts one maths thread per CPU it can see, and requests served at the same time fought over the
CPUs. Limiting NumPy to one thread (`OPENBLAS_NUM_THREADS=1`) cut the median to about 3.6 ms and the
p95 to about 6.8 ms. The image now sets this (`deploy/Dockerfile`), and with it the pilot gave a
median of about 4.0 ms and a p95 of about 7.6 ms.

**A mistake on the way:** the first edit to the Dockerfile was made by a script whose check failed
without stopping it, so one image was rebuilt without the fix. Reading the file before rebuilding
caught it.

## 2026-11-04: How many users the service handles on one laptop

**Expected:** see `experiments/registered/load-test.yaml`, version 1. One copy would miss the target
(p95 at most 100 ms, errors at most 1%) between 200 and 600 users, because one request costs about
3 ms of one CPU, so a copy would top out near 300 requests a second. Two copies would reach at least
1.5 times one copy's peak requests a second and miss the target at a higher step.

**Ran:** `trs load-test --plan experiments/registered/load-test.yaml` at commit `5504334`, against
the real bundle, on a laptop (Intel Core i7-1355U, 2 performance and 8 efficiency cores, 12 threads,
15.7 GB; Docker Desktop with 12 CPUs and 7.6 GiB). A three-node kind cluster ran on Docker CPUs 0 to
7, and k6 2.3.0 on CPUs 8 to 11. Each of 9 steps from 50 to 2,000 users ran for 1 and 2 copies, 3
passes each, with the order of copies shuffled every pass: 54 steps of 60 measured seconds, all
saved in `experiments/results/load-test/`.

**Median of 3 passes, range in brackets:**

| Users | 1 copy: p95 ms | 1 copy: requests/s | 2 copies: p95 ms | 2 copies: requests/s |
|---|---|---|---|---|
| 50 | 6.3 (6.2 to 6.6) | 49.8 | 7.0 (6.6 to 7.9) | 50.1 |
| 200 | 35.2 (7.8 to 320.2) | 197.7 | 12.4 (10.1 to 14.4) | 198.1 |
| 400 | 72.4 (71.5 to 451.5) | 391.0 | 62.3 (12.8 to 241.5) | 391.5 |
| 600 | 810.2 (651.4 to 844.7) | 464.8 | 224.0 (36.4 to 536.1) | 564.6 |
| 1,000 | 1,088.5 (1,008.2 to 1,614.4) | 540.0 | 682.9 (591.3 to 1,421.0) | 710.5 |
| 2,000 | 4,105.7 (4,104.5 to 5,000.9) | 374.2 | 5,000.8 (2,999.3 to 5,001.3) | 451.7 |

Every step is in `summary.json`. Figure: `report/figures/load_test.png`, drawn by
`report/figures/load_test.py` from the committed result files.

**Prediction check:**

- One copy misses the target between 200 and 600 users: held, at the edge. It met the target at 400
  users and missed it at 600. The reason given was wrong: one copy reached 540.0 requests a second,
  not about 300.
- Two copies reach 1.5 times the peak: failed. Their peak was 710.5 requests a second, 1.32 times
  one copy's.
- Two copies miss the target at a higher step: failed. Both first missed it at 600 users.

**What happened:** up to 400 users, the service answered almost every user's request each second,
with a p95 of 6 to 72 ms. At 600 users it fell behind, and the p95 rose past 100 ms for both setups.
Between 600 and 1,000 users, two copies answered up to 32% more requests a second and kept latency
lower, but still several times over the target. Beyond 1,000 users both answered fewer requests a
second, not more, and from 1,500 users more than 1% of requests failed, including timeouts at 5 s.
No step below 1,500 users had a median error rate above zero, although one pass with one copy at
1,000 users had 24.9% errors.

**Why one copy beat the prediction, not tested:** the recommendation route is a plain function, so
FastAPI runs it on a pool of threads, and NumPy can let other threads run while it computes. One
Python process may therefore use more than one CPU, which would also explain why a second copy added
less than expected.

**Spread between passes:** the same step can differ a lot between passes. With one copy at 200
users, one pass had a p95 of 320.2 ms while the others had 7.8 and 35.2 ms. The laptop's mix of fast
and slow cores, and its power settings, likely cause this, but it was not tested. The failure point
uses the median of three passes, so one slow pass does not move it.

**Limits:** all three nodes and k6 share one laptop. Pinning sets which of Docker's virtual CPUs each
part uses, but Windows decides which physical core runs them. All traffic enters through one
worker's port. k6 users wait for each answer before asking again, so under overload they slow down
instead of piling up, which hides some of the queueing real traffic would cause. This is not a
production cluster, and the numbers describe this laptop only.

**What it changes:** on this laptop the service meets its target up to 400 simultaneous users and
fails at 600, with one copy or two. By the registered rule, a second copy does not raise capacity
here, which supports decision 039's choice to skip autoscaling on one machine (decision 042).
