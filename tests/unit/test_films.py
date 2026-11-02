from pathlib import Path

import pytest

from temporal_recsys.data.films import (
    FilmInfo,
    build_film_features,
    load_film_info,
    parse_title_line,
)


def test_title_keeps_its_commas() -> None:
    assert parse_title_line("7,1999,Red, White, Blue") == FilmInfo(7, 1999, "Red, White, Blue")


def test_missing_year_becomes_none() -> None:
    assert parse_title_line("3,NULL,Some Film").year is None


def test_malformed_line_raises_without_repeating_the_title() -> None:
    with pytest.raises(ValueError, match="Malformed") as error:
        parse_title_line("not a title line")

    assert "not a title" not in str(error.value)


def test_load_reads_bytes_that_are_not_utf8(tmp_path: Path) -> None:
    (tmp_path / "movie_titles.csv").write_bytes(b"1,2001,Am\xe9lie\n")

    assert load_film_info(tmp_path) == [FilmInfo(1, 2001, "Am\xe9lie")]


def test_load_without_a_title_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="movie_titles"):
        load_film_info(tmp_path)


def test_features_bucket_years_and_keep_shared_words() -> None:
    films = [
        FilmInfo(2, None, "Blue Moon"),
        FilmInfo(1, 1895, "Blue Sky, part 2"),
        FilmInfo(3, 2005, "Sky"),
    ]

    features = build_film_features(films, n_films=3)

    assert features.year_buckets.tolist() == [2, 0, 24]
    assert features.n_year_buckets == 25
    # "blue" is word 1 and "sky" is word 2. The other words appear in only one film.
    assert features.n_words == 3
    assert features.title_words[0].tolist()[:3] == [1, 2, 0]
    assert features.title_words[1].tolist()[:2] == [1, 0]
    assert features.title_words[2].tolist()[:2] == [2, 0]


def test_features_need_every_film_once() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        build_film_features([FilmInfo(1, 2000, "A")], n_films=2)
