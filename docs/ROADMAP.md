# Roadmap

**Status:** Phase 0 not started.

Update this file when a phase finishes. A phase is done only when its exit
criteria have been demonstrated with real output.

## Phase 0: Foundations (2 days)

- Initialize git, `pyproject.toml`, `uv` lockfile, Python 3.11, MIT license.
- Add ruff, mypy, pytest, hypothesis and pre-commit.
- Add GitHub Actions for lint, types and tests on every push.
- Confirm the Netflix Prize data is available on Kaggle with dated ratings.
- Confirm PySpark runs in a Kaggle CPU notebook, and record its real CPU, RAM,
  disk and session limits.
- Time one Spark read of the data on Kaggle.

**Exit:** CI is green on a trivial test, and the Kaggle Spark check is recorded in
`docs/LAB_NOTEBOOK.md`.

## Phase 1: Data pipeline (1 week)

- Parse raw files into a typed schema with validation.
- Write Parquet partitioned by rating month.
- Build a synthetic generator with a planted temporal drift.
- Benchmark reads for a fixed window: unpartitioned Parquet against partitioned.
  Register the benchmark first. Report read time and total job time separately.
- Add a data card describing the dataset, its limits and its privacy history.

**Exit:** row counts reconcile with the source, and the benchmark reproduces.

## Phase 2: Evaluation and baselines (1 week)

- Implement the temporal split, NDCG@10 and Recall@10.
- Test metrics against hand-computed answers.
- Add property tests that no training row is at or after the cutoff.
- Implement the popularity baseline and single-machine ALS.
- Compare random and temporal splits to measure how much leakage inflates scores.

**Exit:** one command produces the baseline table, and the harness recovers the
planted drift in synthetic data.

## Phase 3: Spark ALS (1 week)

- Implement Spark ALS behind the same evaluation interface.
- Tune on the sample, tracking runs in MLflow.
- Time one full-data run, then decide the compute platform and the run budget for
  Phase 4 (see `docs/PROJECT_OVERVIEW.md`).

**Exit:** Spark ALS beats the popularity baseline on the temporal split, or the
reason it does not is understood and documented.

## Phase 3b: Neural model (1 week)

- Implement a Two-Tower model in PyTorch behind the same evaluation interface.
- User tower: encodes the user's recent ratings. Item tower: encodes the film's ID,
  title and release year. ID-only towers would be close to matrix factorization.
- Use Spark to prepare training examples and PyTorch to train.
- Confirm a free GPU is available, and record its real limits.
- Check that it runs on users and films outside the training window.

**Exit:** the model is evaluated with the same split and metrics as ALS on the sample,
and one full-data run is timed.

## Phase 4: Main experiment (1.5 weeks)

- Register the windows (1, 3, 6, 12, 24 months, all history), cutoffs and seeds.
- Make runs resumable if a remote session ends.
- Run the grid for ALS and Two-Tower. Do not retune per window. If the run budget is
  tight, run Two-Tower on fewer windows, and register that choice first.
- Compare how much history each model needs. Compare compute cost only with the
  hardware named, because the models may run on different machines.
- Control for the growth in rating volume over time by also reporting rows read.
- Break results down by user tenure and item popularity.
- Stretch: repeat on a second dated dataset to test whether the finding generalizes.

**Exit:** headline chart of quality against compute cost with uncertainty, and
every number recorded in `docs/CLAIMS.md`.

## Phase 5: Scaling study (3 days)

- Measure speedup at 1, 2 and 4 cores.
- Compare with single-machine ALS, in the spirit of McSherry et al.'s COST metric.

**Exit:** speedup chart, with an honest statement of where Spark helps and where it
does not.

## Phase 6: Serving (1 week)

- Serve precomputed embeddings from the best model through FastAPI, scoring with NumPy.
- Handle unknown users with the popularity fallback.
- Add health checks, structured logs and metrics.
- Package with Docker and deploy to a multi-node kind cluster with autoscaling.

**Exit:** the API returns recommendations from inside the cluster, with
integration tests.

## Phase 7: Load testing (3 days)

- Define the latency target before testing.
- Ramp k6 past 500 virtual users until the service degrades.
- Pin k6 and the service to separate CPU cores.
- Record concurrency, requests per second, p95 latency and error rate.

**Exit:** latency curve, failure point and a stated limitation of the local setup.

## Phase 8: Publication (1 week)

- Write a 6-8 page report: question, related work, method, results, threats to
  validity, ethics note and future work.
- Build the README: the finding first, headline figure, results table, one-command
  reproduction and a short architecture diagram.
- Make the first screen of the README stand alone: the question, the finding and the
  headline figure, readable in under a minute.
- Add a model card, `CITATION.cff`, a tagged release and a Zenodo DOI.
- Set the repository description and topics, and add a CI status badge.
- Record a demo of about one minute and link it from the README.
- Have someone outside the project follow the README from a clean clone.

**Exit:** a stranger reproduces the synthetic run from the README alone.

## If time runs short

Cut in this order: autoscaling, second dataset, user and item breakdown, scaling
study, Two-Tower windows beyond the essential few. Protect Phases 2, 4 and 8.

## Related work to read

- Dacrema et al. (2019), *Are We Really Making Much Progress?*
- Ji et al. (2020), *A Critical Study on Data Leakage in Recommender System Offline
  Evaluation*
- Koren (2009), *Collaborative Filtering with Temporal Dynamics*
- Krichene and Rendle (2020), *On Sampled Metrics for Item Recommendation*
- McSherry et al. (2015), *Scalability! But at what COST?*
- Covington et al. (2016), *Deep Neural Networks for YouTube Recommendations*
- Rendle et al. (2020), *Neural Collaborative Filtering vs. Matrix Factorization
  Revisited*
- He et al. (2020), *LightGCN*
- Narayanan and Shmatikov (2008), *Robust De-anonymization of Large Sparse Datasets*
