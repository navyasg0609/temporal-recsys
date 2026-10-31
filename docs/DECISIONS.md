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
