"""Measure the quality of one user's ranked list.

Both metrics divide by the most hits a list of length k could achieve, so a perfect
list scores 1.0 even when a user has more than k relevant films.
"""

import math
from collections.abc import Collection, Sequence


def recall_at_k(ranked: Sequence[int], relevant: Collection[int], k: int) -> float:
    hits = sum(film in relevant for film in ranked[:k])
    return hits / min(k, len(relevant))


def ndcg_at_k(ranked: Sequence[int], relevant: Collection[int], k: int) -> float:
    gain = sum(
        1 / math.log2(position + 2) for position, film in enumerate(ranked[:k]) if film in relevant
    )
    best_gain = sum(1 / math.log2(position + 2) for position in range(min(k, len(relevant))))
    return gain / best_gain
