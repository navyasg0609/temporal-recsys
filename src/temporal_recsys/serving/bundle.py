"""Everything the service needs to recommend, saved as one file of NumPy arrays.

The bundle mirrors how the main experiment scored a model. Users in the training window
get the model's vectors. Everyone else gets the window's most popular films. Films a
user rated at any time before the cutoff are never recommended to them.

This module needs only NumPy, so the serving image carries neither Spark nor PyTorch.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from temporal_recsys.models.factors import FactorModel

FORMAT_VERSION = 1


@dataclass(frozen=True)
class ServingBundle:
    """`user_ids` and `seen_user_ids` are sorted. Film index n is film ID n + 1.

    The films user `seen_user_ids[i]` rated are
    `seen_films[seen_starts[i] : seen_starts[i + 1]]`.
    """

    user_ids: NDArray[np.int32]
    user_vectors: NDArray[np.float64]
    film_vectors: NDArray[np.float64]
    popularity: NDArray[np.float64]
    seen_user_ids: NDArray[np.int32]
    seen_starts: NDArray[np.int64]
    seen_films: NDArray[np.int32]

    def __post_init__(self) -> None:
        n_films, dimensions = self.film_vectors.shape
        if self.user_vectors.shape != (len(self.user_ids), dimensions):
            raise ValueError(
                f"Expected user vectors of shape {(len(self.user_ids), dimensions)}, "
                f"got {self.user_vectors.shape}"
            )
        if self.popularity.shape != (n_films,):
            raise ValueError(f"Expected {n_films} popularity scores, got {self.popularity.shape}")
        for name in ("user_ids", "seen_user_ids"):
            ids = getattr(self, name)
            if np.any(np.diff(ids) <= 0):
                raise ValueError(f"{name} must be sorted with no repeats")
        if len(self.seen_starts) != len(self.seen_user_ids) + 1:
            raise ValueError("seen_starts needs one more entry than seen_user_ids")
        if self.seen_starts[0] != 0 or self.seen_starts[-1] != len(self.seen_films):
            raise ValueError("seen_starts must run from 0 to the number of seen films")
        if np.any(np.diff(self.seen_starts) < 0):
            raise ValueError("seen_starts must not decrease")
        if (
            len(self.seen_films)
            and not 0 <= self.seen_films.min() <= self.seen_films.max() < n_films
        ):
            raise ValueError(f"Seen film indices must lie in 0 to {n_films - 1}")

    @property
    def n_films(self) -> int:
        return len(self.popularity)

    def user_row(self, user_id: int) -> int | None:
        """The row of this user's vector, or None if the model never saw them."""
        return _row_of(self.user_ids, user_id)

    def seen_for(self, user_id: int) -> NDArray[np.int32]:
        row = _row_of(self.seen_user_ids, user_id)
        if row is None:
            return np.empty(0, dtype=np.int32)
        return self.seen_films[self.seen_starts[row] : self.seen_starts[row + 1]]


def build_bundle(
    factors: FactorModel,
    window_film_ids: NDArray[np.int32],
    history_user_ids: NDArray[np.int32],
    history_film_ids: NDArray[np.int32],
    n_films: int,
) -> ServingBundle:
    """Pack a trained model with the window's popularity and every user's rated films.

    The window is what the model trained on, and the history is every rating before the
    cutoff, as in the main experiment (decisions 030 and 031).
    """
    order = np.lexsort((history_film_ids, history_user_ids))
    users, films = history_user_ids[order], history_film_ids[order] - 1
    seen_user_ids, firsts = np.unique(users, return_index=True)
    return ServingBundle(
        user_ids=factors.user_ids.astype(np.int32),
        user_vectors=factors.user_factors,
        film_vectors=factors.film_factors,
        popularity=np.bincount(window_film_ids - 1, minlength=n_films).astype(np.float64),
        seen_user_ids=seen_user_ids.astype(np.int32),
        seen_starts=np.append(firsts, len(users)).astype(np.int64),
        seen_films=films.astype(np.int32),
    )


def save_bundle(bundle: ServingBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        np.savez(
            file,
            format_version=np.array(FORMAT_VERSION),
            user_ids=bundle.user_ids,
            user_vectors=bundle.user_vectors,
            film_vectors=bundle.film_vectors,
            popularity=bundle.popularity,
            seen_user_ids=bundle.seen_user_ids,
            seen_starts=bundle.seen_starts,
            seen_films=bundle.seen_films,
        )


def load_bundle(path: Path) -> ServingBundle:
    with np.load(path, allow_pickle=False) as arrays:
        version = int(arrays["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(f"Bundle format {version} is not the supported {FORMAT_VERSION}")
        return ServingBundle(
            user_ids=arrays["user_ids"],
            user_vectors=arrays["user_vectors"],
            film_vectors=arrays["film_vectors"],
            popularity=arrays["popularity"],
            seen_user_ids=arrays["seen_user_ids"],
            seen_starts=arrays["seen_starts"],
            seen_films=arrays["seen_films"],
        )


def _row_of(sorted_ids: NDArray[np.int32], user_id: int) -> int | None:
    row = int(np.searchsorted(sorted_ids, user_id))
    if row < len(sorted_ids) and sorted_ids[row] == user_id:
        return row
    return None
