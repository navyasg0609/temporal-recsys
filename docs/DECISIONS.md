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
