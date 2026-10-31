from pathlib import Path

import numpy as np
import pytest

from temporal_recsys.data.parser import parse_ratings
from temporal_recsys.data.synthetic import (
    SyntheticConfig,
    SyntheticData,
    generate_ratings,
    write_raw_file,
)

CONFIG = SyntheticConfig()


@pytest.fixture(scope="module")
def data() -> SyntheticData:
    return generate_ratings(CONFIG)


def test_same_seed_gives_identical_ratings(data: SyntheticData) -> None:
    assert generate_ratings(CONFIG).ratings == data.ratings


def test_different_seed_gives_different_ratings(data: SyntheticData) -> None:
    assert generate_ratings(SyntheticConfig(seed=1)).ratings != data.ratings


def test_each_user_rates_each_film_at_most_once(data: SyntheticData) -> None:
    pairs = {(rating.user_id, rating.film_id) for rating in data.ratings}

    assert len(pairs) == CONFIG.ratings


def test_stars_are_valid_and_varied(data: SyntheticData) -> None:
    assert {rating.stars for rating in data.ratings} == {1, 2, 3, 4, 5}


def _correlation_with_early_tastes(data: SyntheticData, *, before_change: bool) -> float:
    change_year = CONFIG.first_year + CONFIG.change_month_index // 12
    selected = [
        rating for rating in data.ratings if (rating.rated_on.year < change_year) == before_change
    ]
    predicted = [
        data.tastes_before_change[rating.user_id - 1] @ data.film_traits[rating.film_id - 1]
        for rating in selected
    ]
    return float(np.corrcoef(predicted, [rating.stars for rating in selected])[0, 1])


def test_early_tastes_predict_ratings_only_before_the_change(data: SyntheticData) -> None:
    assert _correlation_with_early_tastes(data, before_change=True) > 0.7
    assert abs(_correlation_with_early_tastes(data, before_change=False)) < 0.05


def test_raw_file_parses_back_to_the_same_ratings(data: SyntheticData, tmp_path: Path) -> None:
    path = tmp_path / "combined_data_1.txt"

    write_raw_file(data.ratings, path)

    with path.open(encoding="ascii") as lines:
        assert sorted(parse_ratings(lines)) == sorted(data.ratings)


@pytest.mark.parametrize("change_month_index", [0, 24])
def test_change_outside_the_time_range_raises(change_month_index: int) -> None:
    with pytest.raises(ValueError, match="inside the time range"):
        SyntheticConfig(change_month_index=change_month_index)


def test_more_ratings_than_pairs_raises() -> None:
    with pytest.raises(ValueError, match="only once"):
        SyntheticConfig(users=2, films=2, ratings=5)
