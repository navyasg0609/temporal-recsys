# Data card: Netflix Prize ratings

## Summary

| Item | Value |
|---|---|
| Ratings | 100,480,507 |
| Films | 17,770 |
| Users | 480,189 |
| Date range | 1999-11-11 to 2005-12-31 |
| Rating scale | 1 to 5 stars, whole numbers |
| Mean rating | 3.60 |
| Matrix density | 1.18% of possible user and film pairs |
| Duplicate user and film pairs | 0 |

These values were measured on the Kaggle copy of the training files. See
[quality_report.json](../experiments/results/ingest/quality_report.json) and
[CLAIMS.md](CLAIMS.md).

## Source

Netflix released this data for the Netflix Prize competition in 2006. This project
uses the four `combined_data_*.txt` training files. It does not use the competition's
qualifying set, which has no ratings, or its probe list. Test periods are defined by
date instead.

The data is not stored in this repository and must not be redistributed here. Tests
use generated synthetic data.

## Distribution over time

| Year | Ratings | Share |
|---|---|---:|
| 1999 | 2,178 | 0.0% |
| 2000 | 924,443 | 0.9% |
| 2001 | 1,769,031 | 1.8% |
| 2002 | 4,342,871 | 4.3% |
| 2003 | 9,985,337 | 9.9% |
| 2004 | 30,206,574 | 30.1% |
| 2005 | 53,250,073 | 53.0% |

The last 12 months hold more than half of all ratings. A window's length in months
therefore says little about its size, so results must also report rows used.

Monthly counts peak in 2005 and then fall sharply, from 5,156,753 in October to
2,405,578 in December. Ratings held back for the competition explain only 17.6% of that
gap. New users also fall 88.6% over those months, which fits a selection rule that left
out people who joined near the end, although the data cannot confirm this. November and
December 2005 are excluded from test periods for this reason.

## Distribution of ratings

| Stars | Ratings | Share |
|---|---|---:|
| 1 | 4,617,990 | 4.6% |
| 2 | 10,132,080 | 10.1% |
| 3 | 28,811,247 | 28.7% |
| 4 | 33,750,958 | 33.6% |
| 5 | 23,168,232 | 23.1% |

## Limitations

- **Self-selected ratings.** People choose what to watch and rate, so missing ratings
  are not random. Offline scores may not reflect what users would do with a real
  recommender.
- **One service, one era.** The data covers DVD-rental customers up to 2005, so
  findings may not transfer to other catalogs or to modern streaming.
- **Growing platform.** User behavior and the customer base change along with the
  volume of ratings, so time effects and growth effects can be hard to separate.
- **Perturbed values.** The organizers reported altering some ratings to protect
  privacy (Bennett and Lanning, 2007).
- **Dates, not sequences.** Ratings have dates but no time of day, so the order of
  ratings within a day is unknown.

## Privacy

Netflix removed names and replaced user IDs, but anonymization did not fully protect
users. Narayanan and Shmatikov (2008) showed that a small amount of outside
information, such as public movie reviews, could re-identify people in the data. Film
ratings can reveal sensitive information about a person.

This project therefore analyzes the data only in aggregate. It makes no attempt to
identify anyone, and it publishes no individual rating records.
