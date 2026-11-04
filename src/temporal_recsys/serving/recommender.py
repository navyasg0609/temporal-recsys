"""Recommend films to one user from a serving bundle."""

from dataclasses import dataclass
from enum import StrEnum

from temporal_recsys.evaluation.ranking import rank_unseen
from temporal_recsys.serving.bundle import ServingBundle


class Source(StrEnum):
    MODEL = "model"
    """The user rated films in the training window, so the model has a vector for them."""

    POPULAR = "popular"
    """The model never saw the user, so they get the window's most rated films."""


@dataclass(frozen=True)
class Recommendation:
    user_id: int
    source: Source
    film_ids: list[int]


class Recommender:
    def __init__(self, bundle: ServingBundle) -> None:
        self.bundle = bundle

    def recommend(self, user_id: int, k: int) -> Recommendation:
        """Rank every film for this user, leaving out any film they rated before."""
        bundle = self.bundle
        row = bundle.user_row(user_id)
        if row is None:
            source, scores = Source.POPULAR, bundle.popularity.copy()
        else:
            source, scores = Source.MODEL, bundle.film_vectors @ bundle.user_vectors[row]
        (ranked,) = rank_unseen(scores[None, :], [bundle.seen_for(user_id)], k)
        return Recommendation(user_id, source, [index + 1 for index in ranked])
