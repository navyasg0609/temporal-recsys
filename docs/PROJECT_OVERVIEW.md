# Project overview

This document explains the problem, the project and the plan in simple language.
For the detailed checklist of each phase, see [ROADMAP.md](ROADMAP.md). For the
reasons behind technical choices, see [DECISIONS.md](DECISIONS.md).

## The problem

### Background

Streaming services and online shops use **recommender systems** to suggest what you
might like next. A common approach is **collaborative filtering**: if people with
tastes like yours enjoyed a film, you may enjoy it too.

These systems learn from past behaviour, such as the ratings people gave. Companies
collect this data for years, and retraining a model on all of it takes a lot of
computing time and money.

### Why more data is not always better

People's tastes change, and so do the films on offer. A rating from eight years ago
may say little about what someone wants today. Old data could even mislead the model.
So there is a real trade-off:

- **Using more history** gives the model more examples, but costs more to train and
  may include outdated behaviour.
- **Using less history** is cheaper and more current, but the model may miss useful
  patterns.

### The research question

> **How much history does a recommender actually need, and what does each extra
> month of data cost in computing time?**

We do not know the answer yet. The project is designed so that any clear result is
useful, including "more history always helps."

## The project

### What we will do

We will train two types of model, a classic one (ALS) and a neural one (Two-Tower), on
different amounts of recent history: 1, 3, 6, 12 and 24 months, and then all available
data. This also lets us ask whether a neural model needs more or less history than a
classic one. For each run, we will measure two things:

1. **Quality:** how good are its recommendations?
2. **Cost:** how much computing time did it use?

The main result will be one chart showing quality against cost. It will show the
point where adding more data stops being worth it, if such a point exists.

### The data

We use the **Netflix Prize dataset**, which has 100,480,507 ratings of 17,770 films,
dated from November 1999 to December 2005. Each rating has a date, which this question
requires. We confirmed this on Kaggle, and the check is in the
[lab notebook](LAB_NOTEBOOK.md).

The data will not be stored in this repository, because we should not redistribute
it. Automatic tests will use **synthetic data** instead, meaning fake ratings we
generate ourselves. Because we create that data, we know the right answer in advance
and can check that our code finds it.

### The main parts

| Part | What it does in simple words |
|---|---|
| **Parquet files** | Store the ratings in a compact format that is fast to read |
| **Partitioning** | Splits the data into one folder per month, so a job can skip months it does not need. This skipping is called partition pruning. |
| **Apache Spark** | Processes data too large to handle comfortably in ordinary tools |
| **ALS (Alternating Least Squares)** | The recommendation algorithm. It learns a short numeric description of each user and film. |
| **Two-Tower model** | A neural network with one part for users and one for films. It reads a user's recent ratings and a film's title and year, then predicts a match. |
| **PyTorch** | The deep learning library used to build and train the neural model |
| **Evaluation code** | Measures recommendation quality fairly |
| **FastAPI service** | A small web service that returns recommendations on request |
| **Docker** | Packages the service so it runs the same way everywhere |
| **Kubernetes** | Runs several copies of the service and manages them |
| **k6** | A load-testing tool that simulates many users at once |

### How we will keep the results trustworthy

- **Train on the past and test on the future.** A random split would let the model
  learn from ratings made after the period it is tested on. This is called leakage,
  and it makes results look better than they are.
- **Compare with simple methods.** We will compare against recommending the most
  popular films, and against ALS on a single machine. The neural model must also earn
  its place by beating ALS. If a complex system cannot beat a simple one, we need to
  know.
- **Write the plan down first.** Each experiment's baseline, measurements and settings
  are saved before it runs, so results cannot quietly change the rules.
- **Trace every number.** Each published number is linked to the run that produced it
  in [CLAIMS.md](CLAIMS.md).
- **Report failures.** Ideas that did not work are recorded too.
- **Repeat the measurements.** We will use several time periods and random seeds, and
  show how much the results vary.

## The plan, step by step

Timings are estimates, and the full plan is about 8 to 9 weeks of work.

### Phase 0: Set up the foundations (2 days)

We create the project structure, the tools that check code automatically, and the
automatic test run on every code change. We also confirm the dataset is available and
test where large jobs can run.

**Finished when:** the automatic checks pass and those tests are written down.

### Phase 1: Build the data pipeline (1 week)

We convert the raw files into monthly Parquet folders, check the data for errors, and
build the synthetic data generator. We then measure how much time partition pruning
really saves.

**Finished when:** the row counts match the source and the timing test can be repeated.

### Phase 2: Build fair evaluation and baselines (1 week)

We write the code that splits data by time and scores recommendations, and we test the
scoring against answers worked out by hand. We also build the simple methods we will
compare against, and measure how much a random split inflates the scores.

**Finished when:** one command produces the comparison table, and the code finds the
known answer in the synthetic data.

### Phase 3: Train the Spark model (1 week)

We build ALS in Spark, tune its settings on the same month as the NumPy model, and time
one run on almost all the data. That timing tells us how many runs we can afford, and it informs the
compute choice described below.

**Finished when:** the model beats the popularity method, or we understand why not.

### Phase 3b: Train the neural model (1 week)

We build the Two-Tower model in PyTorch and test it with exactly the same split and
scoring as ALS. Spark prepares the training examples, and PyTorch does the training.
The model reads recent ratings and film details rather than only ID numbers, because
an ID-only version would be very close to ALS. We also confirm that a free GPU is
available. A GPU is a chip that greatly speeds up neural network training.

**Finished when:** the model is scored fairly on the sample and one full run is timed.

### Phase 4: Run the main experiment (1.5 weeks)

We write down the experiment plan, then run every combination of model, history
length, time period and seed. If time is tight, the neural model runs on fewer history
lengths, and we write that down first. We also look at whether the answer differs for
new and long-time users, or for popular and niche films.

**Finished when:** the main chart is complete and every number is recorded.

### Phase 5: Measure scaling (3 days)

We test how much faster the job gets when given more computing power, and compare that
with a single-machine method. Spark may not win, and that would be a useful result.

**Finished when:** we have a chart and an honest statement of where Spark helps.

### Phase 6: Serve recommendations (1 week)

We build the web service and package it with Docker. We then run it on Kubernetes. If
a user is unknown, the service falls back to popular films.

**Finished when:** the service returns recommendations and its tests pass.

### Phase 7: Test under load (3 days)

We set a speed target first, then increase simulated users, going past 500, until the
service slows down or fails. We record the number of users, requests per second,
response time and errors.

**Finished when:** we know the service's limit and have described the test setup.

### Phase 8: Publish (1 week)

We write a short research report, a clear README and a demo, and release the code so
others can repeat the work. Someone outside the project then tries to follow the
instructions from scratch.

**Finished when:** a new person can reproduce the synthetic run using only the README.

## Where the heavy computing will run

ALS training stays on free Kaggle notebooks (decision 022). The rest is **still undecided**.
The project aims to stay free, and we are keeping the other options open.

Daily development happens on a laptop using small samples. The options below are for
the large jobs in Phases 3, 3b, 4, 5 and 7. Their limits and terms change often, so check
them again before choosing.

| Option | Strength | Limitation | Multiple real machines? |
|---|---|---|---|
| **Kaggle notebooks** | Free, with no cluster to manage. Reported at about 4 CPU cores and 30 GB of memory. | Sessions end after a time limit. Not suited to hosting a service. | No |
| **Local laptop** | Free and fully under our control | Too small for full-data training | No, but Kubernetes can simulate several machines |
| **Azure for Students** | Reported $100 credit with no payment card | Only for eligible students, and the credit is limited | Yes |
| **Google Cloud free trial** | Reported $300 credit for 90 days | Needs a payment method, for new accounts only | Yes |
| **Oracle Cloud free tier** | A permanent free tier | Needs a card. The allowance was reportedly cut, and it is small for this work. | Limited |
| **Paid short-term rental** | Most flexible | Costs money, and resources must be deleted after use | Yes |

### How we will decide

We will decide after timing one full training run in Phase 3. The choice depends on:

1. **How long one run takes,** and how many runs the experiment needs.
2. **Whether we want to claim distributed computing.** Spark on one machine is not
   distributed, so that claim needs real multi-node runs.
3. **Whether the load test needs its own setup.** On one laptop, the test tool and the
   service compete for the same computer.
4. **The cost and billing risk** of each option. Any option that needs a card should
   have spending limits, and its resources should be deleted after each session.
5. **Whether it offers a free GPU** for training the neural model.

Whatever we choose, each published performance number will state the hardware used.

## What will be delivered

- A public code repository with automatic tests.
- One command that runs the whole pipeline on synthetic data.
- A main chart and a results table.
- A research report of 6 to 8 pages.
- A running service with load-test results.
- A short demo.

## Main risks

| Risk | What we will do |
|---|---|
| The dataset is unavailable or unsuitable | Check first, in Phase 0, before building on it |
| Training is too slow for the full experiment | Time one run early, then reduce seeds or time periods, or review the compute choice |
| Partition pruning saves less time than hoped | Report the measured result, whatever it is |
| The neural model does not beat ALS | Report it. This is a known finding in the field, and explaining why is valuable. |
| No free GPU time is available | Train on a smaller sample and state this clearly |
| The result is not exciting | A clear, honest result still answers the question |
| Time runs out | Cut in this order: autoscaling, second dataset, user and film breakdown, scaling study, extra neural model runs |
