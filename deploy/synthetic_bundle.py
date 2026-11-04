"""Write a small serving bundle with a planted answer, for testing a deployment.

Run from the repository root: `uv run python deploy/synthetic_bundle.py artifacts`.
User 1 must get film 7 first: film 3 scores higher, but user 1 already rated it. A user
the model never saw must get film 5, the most rated film in the window.
"""

import sys
from pathlib import Path

import numpy as np

from temporal_recsys.models.factors import FactorModel
from temporal_recsys.serving.bundle import build_bundle, save_bundle

N_FILMS = 20
N_USERS = 50
KNOWN_USER, KNOWN_USER_FIRST_FILM = 1, 7
UNKNOWN_USER_FIRST_FILM = 5


def main(output_dir: Path) -> None:
    rng = np.random.default_rng(0)
    user_factors = rng.normal(scale=0.1, size=(N_USERS, 4))
    film_factors = rng.normal(scale=0.1, size=(N_FILMS, 4))
    user_factors[KNOWN_USER - 1] = [1.0, 0.0, 0.0, 0.0]
    film_factors[KNOWN_USER_FIRST_FILM - 1] = [1.0, 0.0, 0.0, 0.0]
    film_factors[3 - 1] = [2.0, 0.0, 0.0, 0.0]
    factors = FactorModel(np.arange(1, N_USERS + 1, dtype=np.int32), user_factors, film_factors)

    window_films = np.concatenate(
        [np.full(10, UNKNOWN_USER_FIRST_FILM), rng.integers(1, N_FILMS + 1, size=5)]
    ).astype(np.int32)
    history_users = np.array([KNOWN_USER, 2, 3], dtype=np.int32)
    history_films = np.array([3, 1, 2], dtype=np.int32)
    bundle = build_bundle(factors, window_films, history_users, history_films, N_FILMS)
    save_bundle(bundle, output_dir / "bundle.npz")
    print(f"wrote {output_dir / 'bundle.npz'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
