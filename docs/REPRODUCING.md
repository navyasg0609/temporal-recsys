# Reproducing the results

## On any laptop, in under a minute: synthetic data

```
uv sync
uv run trs reproduce --sample
```

This runs the main method end to end on generated ratings, with no Spark, no GPU and no
Netflix data. Every synthetic user's tastes change in January 2005, so history from before
then misleads a model. The command trains popularity and single-machine ALS on 1, 3, 6 and
12 months and on all history, scores each on the following month, prints the table, and
exits with an error unless ALS on the 6 months after the change beats ALS on all history at
every test month. Settings are in `configs/sample.yaml`, and results go to `sample-run/`.

Synthetic users pick films at random and only their stars follow their tastes, so the sample
uses ALS that predicts stars. The main experiment uses preference ALS, which would find
nothing in this data. The sample checks the method and the code, not the findings.

## On the full data: Kaggle notebooks

The full runs need the Netflix Prize training files (`combined_data_1.txt` to
`combined_data_4.txt` and `movie_titles.csv`), which this repository does not redistribute.
They ran on free Kaggle notebooks with the Netflix Prize dataset attached, so its files
appear under `/kaggle/input`. Each run starts from the commit its results record
(decision 013):

```
git clone -q https://github.com/navyasg0609/temporal-recsys.git repo
git -C repo checkout -q <commit>
pip install -q "./repo[spark,neural]"
trs ingest --raw-dir /kaggle/input --output-dir /kaggle/working/ratings \
  --driver-memory 20g --check-netflix
```

Spark became an optional extra in Phase 6, so at earlier commits pip warns that `spark` is
unknown and installs Spark anyway. Then one command per experiment. Each saves every finished arm at once, so rerunning the
same command after a session ends keeps the finished arms.

| Result | Command | Commit | Notebook |
|---|---|---|---|
| How much history each model needs | `trs history --model popularity` | `2fe0302` | CPU |
| | `trs history --model spark-als --driver-memory 20g` | `2fe0302` | CPU |
| | `trs history --model two-tower --raw-dir /kaggle/input` | `2fe0302` | GPU |
| Spark ALS against cores, and COST | `trs scaling` | `1ee6fdd` | CPU |
| The served model | `trs export-serving --config configs/serving.yaml` | `aa6d52b` | GPU |

The `trs history` and `trs scaling` commands also take `--data-dir /kaggle/working/ratings`,
`--plan repo/experiments/registered/<plan>.yaml`, `--output-dir` and `--code-commit` with the
full commit ID. Every published number, with its command and commit, is in
[CLAIMS.md](CLAIMS.md), and each run's notes are in [LAB_NOTEBOOK.md](LAB_NOTEBOOK.md).

## The service and the load test: a laptop with Docker

See [deploy/README.md](../deploy/README.md). The load test needs the bundle that
`trs export-serving` writes, which holds rating data and so is not in this repository.
