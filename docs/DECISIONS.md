# Decisions

Newest last. Each entry records what was chosen and why. To reverse a decision, add
a new entry rather than editing an old one.

## 001: Study training history, not just pipeline throughput

The project asks how much history a recommender needs. This gives the pipeline a
measurable purpose and produces a finding, not only a working system.

## 002: Netflix Prize data

It has about 100M ratings with dates, and its temporal drift is well documented.
MovieLens has no release of comparable size. To be confirmed in Phase 0.

## 003: Evaluate temporally

Train on the past and test on the following period, because random splits let
future information leak into training.

## 004: Partition Parquet by rating month

Training on a recent window then reads only the files it needs. Pruning does little
for full-history training, so its benefit is measured on windowed reads.

## 005: Zero-cost infrastructure

Kaggle CPU notebooks run full-data jobs, and a local kind cluster hosts serving. No
service requiring a payment card is used. As a result, this project does not claim
multi-node Spark, and performance numbers name the setup they came from.

## 006: Serve with NumPy, not an approximate index

The catalog has about 17,770 items, so exact scoring is small enough that an
approximate nearest neighbor index would add complexity without a clear benefit.
Load testing in Phase 7 will confirm this.

## 007: Use synthetic data in CI

The dataset is not redistributed. A generator with planted drift lets CI test the
whole pipeline against a known answer.

## 008: Register experiments before running them

Plans are frozen in `experiments/registered/` so results cannot quietly shape the
baseline or the metrics after the fact.

## 009: Reopen the compute choice

This replaces the compute choice in decision 005. Zero cost remains the goal, but the
platform for large jobs is undecided. The options are compared in
`docs/PROJECT_OVERVIEW.md`, and the choice follows a timed full-data run in Phase 3.

## 010: Add a Two-Tower model, keep ALS

ALS remains the baseline, and a Two-Tower model in PyTorch adds a modern comparison.
This extends the question: does a neural model need more or less history than ALS?

The towers use recent ratings, titles and release years, because ID-only towers are
close to matrix factorization and Rendle et al. (2020) found dot-product models hard
to beat. LightGCN was rejected for now because propagating over about 100M edges is
heavy for limited memory, while Two-Tower trains in small batches. Both models output
embeddings, so the serving design in decision 006 does not change.

## 011: Confirm the dataset and pin Python 3.13 with PySpark 4.0.4

The check in `docs/LAB_NOTEBOOK.md` confirms decision 002: the dated ratings are
available and every line parses. The project moves from Python 3.11 to 3.13 to match
Kaggle, because a different local version could hide failures until a remote run.
PySpark 4.0.4 is pinned when Phase 1 first depends on it.

## 012: Parse raw files in Python, run Spark on Linux only

Each rating gets its film ID from a header line above it, and Spark splits files into
chunks that can separate the two. Reading in order with plain Python avoids that bug
and can be tested without Spark. Spark then handles partitioning. Spark file writes on
Windows need extra native tools, so Spark tests run in CI and on Kaggle, not locally.

## 013: Run Kaggle jobs from an exact commit

Kaggle notebooks clone the repository at a full commit ID and install it from there.
The results record that ID, so every number traces back to the code that produced it.

## 014: Keep month partitions, and report rows alongside window length

Monthly partitions were the fastest layout in the read benchmark on both windows, so
later phases read from them. Ratings are heavily skewed toward 2005, so a window's
length does not describe its size. Results must report the rows used as well.

## 015: Rank against the full catalog, with relevance at 4 or 5 stars

Each user's list ranks all 17,770 films, minus those rated in training. Sampled negatives
can distort results (Krichene and Rendle, 2020). A test rating of 4 or 5 stars counts as
relevant. Both metrics divide by the most hits possible at k, so a perfect list scores 1.0.

## 016: Test on August to October 2005, and tune on July

The check in `docs/LAB_NOTEBOOK.md` shows the last two months are distorted: 12.0% and
24.6% of ratings are held back, and new users fall 88.6% from October to December. Earlier
months hold back at most 5.5%. Settings are chosen on July and never revisited.

## 017: Run baselines with NumPy and SciPy

The popularity baseline and single-machine ALS exist partly to test whether Spark is needed,
so they must not depend on it. Both ALS variants stay in the comparison because rating
prediction and preference weighting can rank films very differently.

## 018: Spark does not run on this Windows machine

This corrects decision 012, which mentioned only writes. Spark also fails to read Parquet
here without native Hadoop tools, so Spark code is tested in CI and on Kaggle.

## 019: Use preference ALS as the ALS baseline

Preference ALS beat rating ALS and popularity on every temporal cutoff, with no overlap in
ranges (`docs/LAB_NOTEBOOK.md`). Comparing later models with rating ALS would set too low a
bar, so Spark ALS uses the preference form and the Two-Tower model must beat this result.
Rating ALS and popularity stay in the reported table.

## 020: Skip MLflow

The roadmap planned MLflow to track runs. `arms.jsonl` and `docs/CLAIMS.md` already tie each number
to its settings and commit, so MLflow would add a dependency and add no evidence.

## 021: Tune Spark ALS on the July cutoff, not on a sample

The roadmap planned to tune on a sample. Tuning on the same cutoff and rows as the NumPy model keeps
the comparison fair, and the pilot showed an arm takes only minutes.

## 022: Keep single-machine runs on Kaggle

This partly settles decision 009. One Spark fit on 89,313,064 ratings took 676 seconds on a free
Kaggle CPU notebook, so ALS needs no paid compute. Kaggle offers one machine, so whether to run on
several machines stays open. The GPU for Phase 3b is confirmed in that phase.

## 023: Use Spark ALS for the main experiment

Spark and NumPy showed no clear difference in quality, and Spark's fit was 23.7% shorter on the same
hardware. Spark therefore runs the main experiment, with `reg=0.01` and `confidence=3`. The later
Two-Tower model must beat the better of the two ALS results, which refines decision 019.

## 024: Do not claim that partitioning shortens training

Monthly partitions read faster, but training showed no clear whole-job saving, because reading is
about 3 to 4% of the job. They stay as the storage layout (decision 014). The time saved by a short
window comes from using fewer rows, and it is reported as that.

## 025: Prepare training examples with NumPy, not Spark

The roadmap planned to use Spark here. The work is sorting that fits in memory, and NumPy keeps it
easy to test. Spark keeps its jobs of building the dataset and training ALS.

## 026: Use only earlier days from inside the training window as model input

Reading history from before the window would defeat the main question, because a 1-month model
could still see years of ratings. Dates have no time of day, so same-day ratings are left out of an
example's input. The model also has no per-user ID and reads only the films a user rated.

## 027: Keep PyTorch as an optional extra, pinned to Kaggle's version

Installing this package on Kaggle must not replace Kaggle's GPU build. The version is pinned so
tests run on the same PyTorch as real training.

## 028: Compare models by quality, and do not rank them by speed across hardware

ALS ran on CPUs and the two-tower model on a GPU, so their seconds measure different machines. Times
are reported with the hardware named, and no model is called faster than one on other hardware.

## 029: Report recent popularity in the main experiment

Popularity over the last month beat both tuned 12-month ALS models, so 12-month popularity alone is
too weak a baseline. This extends decision 023. Every later comparison includes recent popularity,
which takes under a second to fit.

## 030: Score every window on the same users and the same seen films

At each test month, every window is scored on the users who rated anything before the cutoff, and
each user's earlier films are removed from their list however short the window. Otherwise a short
window would face fewer, easier users and could recommend films a user already rated, and scores
across windows would not be comparable.

## 031: Give users missing from a window that window's popular films

A short window leaves out users who rated nothing in it. They get the window's most popular films,
as a real service would give them, instead of a zero score. This keeps a short window from losing
points only because of who was left out. The serving plan in Phase 6 uses the same fallback.

## 032: Cut the user and film breakdown and the second dataset from Phase 4

The main result is clear without them, and the roadmap's cut order lists both before the core
phases. They stay possible future work and are named as limits in the results.

## 033: Measure scaling with threads on one machine

Spark in local mode with 1, 2 and 4 threads shows how Spark uses cores, not how it behaves across a
network. Kaggle offers one machine, and the 4 CPUs are 2 physical cores with 2 threads each. Every
scaling number names this setup, and none is described as distributed.

## 034: Use single-threaded NumPy ALS as the COST baseline

COST asks how many cores Spark needs to beat a single-threaded program doing the same job. The NumPy
ALS already exists, is tested, and matches Spark's quality (decision 023), so it is the baseline. It
is a plain Python loop, and an optimized library would make Spark's COST higher. That is stated as a
limit rather than adding a new dependency.

## 035: Make Spark an optional extra

The serving image needs only NumPy and the web framework, and Spark would add hundreds of megabytes
it never uses. Spark becomes the `spark` extra, as PyTorch became the `neural` extra (decision 027).
Kaggle runs now install the package with `pip install ./repo[spark]`.

## 036: Rank through one function in evaluation and serving

A service that ranks films differently from the evaluation would serve lists nobody measured. Both
call `rank_unseen`, which removes films the user already rated and keeps the top k, and a test
checks that the service returns the same lists as the evaluation for the same model.

## 037: Serve the two-tower model on 6 months before October 2005, seed 1

It is one arm the main experiment scored, so the served model's quality is measured, and the export
retrains it with the same plan and checks the score. Six months is the history this model needed in
Phase 4. `configs/serving.yaml` names the arm, and the export refuses any arm the plan did not score.

## 038: Keep rating data out of the image, the repository and the logs

The bundle holds user IDs and the films each user rated, which is close to the dataset itself, so it
must not be redistributed. It stays in the git-ignored `artifacts/` folder and is mounted read-only
when the service runs. The image holds code only, and leaves out pyarrow and scipy, which the
service never imports. Logs record the route template, never a user ID, and responses give film IDs
without titles.

## 039: Run a fixed two copies, without autoscaling

The roadmap's cut order lists autoscaling first. On one machine every copy of the service shares the
same CPUs, so adding copies under load cannot add capacity, and an autoscaling demo would show the
mechanism without a benefit. The cluster runs two copies, one per worker node, and the kind nodes are
containers on one machine, which every result states.

## 040: Set the latency target before load testing

The service must answer within 100 ms at the 95th percentile, with at most 1% errors, because a
recommendation list should arrive with the page. The failure point is the fewest simultaneous users
at which the median over three passes misses either limit. Both are frozen in
`experiments/registered/load-test.yaml` before any measured run.

## 041: Keep the load generator off the service's CPUs

The service's nodes are pinned to Docker CPUs 0 to 3, the control node to 4 to 7, and k6 to 8 to 11,
so generating load does not take CPU from the service. These are Docker's virtual CPUs: Windows
still schedules them onto physical cores, and this laptop mixes fast and slow cores, so every load
result names that setup.

## 042: Test whether a second copy adds capacity

Decision 039 assumed extra copies cannot add capacity on one machine. Each copy is one Python process
that uses about one CPU, and the service has four, so a second copy may help until the CPUs are
busy. The load test compares one and two copies instead of assuming either answer.
