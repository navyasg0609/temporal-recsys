"""Read film titles and release years, and turn them into numbers for a model.

These features describe the catalog and use no ratings, so they cannot leak test data.
"""

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

TITLE_FILE_PATTERN = "movie_titles*"
TITLE_LINE = re.compile(r"^(\d+),(\d{4}|NULL),(.+)$")
WORD = re.compile(r"[a-z0-9]+")
MISSING_YEAR = "NULL"

FIRST_YEAR = 1890
YEARS_PER_BUCKET = 5
PADDING = 0
MIN_WORD_FILMS = 2
MAX_TITLE_WORDS = 8


@dataclass(frozen=True)
class FilmInfo:
    film_id: int
    year: int | None
    title: str


@dataclass(frozen=True)
class FilmFeatures:
    """One row per film, indexed by film ID minus 1. Zero always means padding."""

    year_buckets: NDArray[np.int64]
    title_words: NDArray[np.int64]
    n_year_buckets: int
    n_words: int


def parse_title_line(line: str) -> FilmInfo:
    match = TITLE_LINE.match(line)
    if match is None:
        raise ValueError(f"Malformed title line of {len(line)} characters")
    film_id, year, title = match.groups()
    return FilmInfo(int(film_id), None if year == MISSING_YEAR else int(year), title)


def load_film_info(raw_dir: Path) -> list[FilmInfo]:
    paths = sorted(raw_dir.rglob(TITLE_FILE_PATTERN))
    if not paths:
        raise FileNotFoundError(f"No {TITLE_FILE_PATTERN} file under {raw_dir}")
    # Some titles hold bytes that are not UTF-8, and this encoding accepts any byte.
    lines = paths[0].read_text(encoding="latin-1").splitlines()
    return [parse_title_line(line) for line in lines]


def build_film_features(films: Sequence[FilmInfo], n_films: int) -> FilmFeatures:
    """Group years into buckets, and keep words shared by at least two films.

    A word found in only one film says nothing that the film's own ID does not.
    """
    if sorted(film.film_id for film in films) != list(range(1, n_films + 1)):
        raise ValueError(f"Expected exactly one entry for each of {n_films} films")

    ordered = sorted(films, key=lambda film: film.film_id)
    words_by_film = [_title_words(film.title) for film in ordered]
    vocabulary = _word_ids(words_by_film)

    year_buckets = np.array([_year_bucket(film.year) for film in ordered], dtype=np.int64)
    title_words = np.full((n_films, MAX_TITLE_WORDS), PADDING, dtype=np.int64)
    for row, words in enumerate(words_by_film):
        known = [vocabulary[word] for word in words if word in vocabulary]
        title_words[row, : len(known[:MAX_TITLE_WORDS])] = known[:MAX_TITLE_WORDS]
    return FilmFeatures(year_buckets, title_words, int(year_buckets.max()) + 1, len(vocabulary) + 1)


def _title_words(title: str) -> list[str]:
    return WORD.findall(title.lower())


def _word_ids(words_by_film: Iterable[list[str]]) -> dict[str, int]:
    film_counts = Counter(word for words in words_by_film for word in set(words))
    shared = sorted(word for word, count in film_counts.items() if count >= MIN_WORD_FILMS)
    return {word: index + 1 for index, word in enumerate(shared)}


def _year_bucket(year: int | None) -> int:
    if year is None:
        return PADDING
    if year < FIRST_YEAR:
        raise ValueError(f"Release year {year} is before {FIRST_YEAR}")
    return (year - FIRST_YEAR) // YEARS_PER_BUCKET + 1
