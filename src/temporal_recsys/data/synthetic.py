"""Generate ratings with a planted change in user tastes.

Every user gets new tastes at a known month, so ratings from before that month carry
no information about ratings after it. Because the answer is known in advance, this
data can test whether the pipeline and evaluation detect a real temporal effect.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from temporal_recsys.data.parser import MAX_RATING, MIN_RATING, Rating

MEAN_STARS = 3.0
STARS_PER_UNIT_OF_AFFINITY = 1.2
LAST_SAFE_DAY = 28


@dataclass(frozen=True)
class SyntheticConfig:
    users: int = 2_000
    films: int = 300
    ratings: int = 50_000
    first_year: int = 2004
    months: int = 24
    change_month_index: int = 12
    taste_dimensions: int = 5
    noise: float = 0.5
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.change_month_index < self.months:
            raise ValueError("The taste change must fall inside the time range")
        if self.ratings > self.users * self.films:
            raise ValueError("Each user can rate each film only once")


@dataclass(frozen=True)
class SyntheticData:
    ratings: list[Rating]
    tastes_before_change: NDArray[np.float64]
    tastes_after_change: NDArray[np.float64]
    film_traits: NDArray[np.float64]


def generate_ratings(config: SyntheticConfig) -> SyntheticData:
    rng = np.random.default_rng(config.seed)
    scale = 1 / np.sqrt(config.taste_dimensions)
    taste_shape = (config.users, config.taste_dimensions)
    tastes_before = rng.normal(scale=scale, size=taste_shape)
    tastes_after = rng.normal(scale=scale, size=taste_shape)
    film_traits = rng.normal(size=(config.films, config.taste_dimensions))

    pairs = rng.choice(config.users * config.films, size=config.ratings, replace=False)
    users, films = np.divmod(pairs, config.films)
    months = rng.integers(0, config.months, size=config.ratings)
    days = rng.integers(1, LAST_SAFE_DAY + 1, size=config.ratings)

    is_before_change = months < config.change_month_index
    tastes = np.where(is_before_change[:, None], tastes_before[users], tastes_after[users])
    affinity = np.sum(tastes * film_traits[films], axis=1)
    noise = rng.normal(scale=config.noise, size=config.ratings)
    stars = np.clip(
        np.rint(MEAN_STARS + STARS_PER_UNIT_OF_AFFINITY * affinity + noise),
        MIN_RATING,
        MAX_RATING,
    ).astype(int)

    ratings = [
        Rating(
            film_id=film + 1,
            user_id=user + 1,
            stars=star,
            rated_on=date(config.first_year + month // 12, month % 12 + 1, day),
        )
        for film, user, star, month, day in zip(
            films.tolist(),
            users.tolist(),
            stars.tolist(),
            months.tolist(),
            days.tolist(),
            strict=True,
        )
    ]
    return SyntheticData(ratings, tastes_before, tastes_after, film_traits)


def write_raw_file(ratings: list[Rating], path: Path) -> None:
    """Write ratings in the Netflix Prize text format."""
    lines: list[str] = []
    current_film: int | None = None
    for rating in sorted(ratings):
        if rating.film_id != current_film:
            current_film = rating.film_id
            lines.append(f"{current_film}:\n")
        lines.append(f"{rating.user_id},{rating.stars},{rating.rated_on}\n")
    path.write_text("".join(lines), encoding="ascii", newline="\n")
