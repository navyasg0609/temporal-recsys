# Model card: the served two-tower recommender

## What it is

A two-tower model that ranks films for a user. The user tower averages embeddings of the films
the user rated most recently, up to 50, and has no per-user ID. The film tower adds a film ID
embedding to embeddings of its release year and title words. Both towers output 64 numbers, and
a film's score is the dot product of the two. The service ranks all 17,770 films with NumPy and
removes films the user already rated.

It is one arm the main experiment scored: trained on the 6 months before 1 October 2005 with seed
1 (decision 037), the history this model needed in Phase 4. `trs export-serving` retrains it from
`configs/serving.yaml` and writes the bundle the service loads, along with a manifest in
`experiments/results/serving-export/`.

## Intended use

A research artifact, for studying how much history a recommender needs and for testing a serving
setup. It is not meant to serve real users: it was trained on data from 2005, has never been
tested online, and its catalog ends in December 2005.

## Training data

Netflix Prize ratings from April to September 2005, with film titles and release years. The data
is not redistributed, and the bundle, which holds user IDs and the films each user rated, stays
out of the repository and the container image (decision 038). See the
[data card](DATA_CARD.md).

## Evaluation

| Measure | Value |
|---|---|
| NDCG@10 on October 2005, full catalog, relevance at 4 or 5 stars | 0.0993, against 0.0994 when first scored |
| Served lists equal to the lists the evaluation scored | 5,000 of 5,000 sampled users |
| Simultaneous users within 100 ms at p95, one laptop | 400; missed at 600 |

The score is for users who rated anything before the cutoff. Users with no ratings in the 6-month
window get that window's most popular films, as does any unknown user. Every measured number here
has a row in [CLAIMS.md](CLAIMS.md), and the main experiment's comparison with popularity and ALS is in the
[report](../report/main.pdf).

## Limits

- Tested on three months in late 2005 only, on one dataset.
- Offline NDCG rewards predicting what users rated, which favors popular films, and ratings reflect
  what Netflix showed its users. No fairness or online measure was taken.
- Training on the GPU was not exactly repeatable with the same seed, so a retrained model can
  differ slightly, as the two scores above do.
- New users get popular films only, and films released after 2005 cannot be recommended.

## Privacy

Ratings and dates in this dataset have been used to re-identify subscribers (Narayanan and
Shmatikov, 2008). The service logs the route, never a user ID, and returns film IDs without
titles.
