from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st

from temporal_recsys.data.parser import Rating, RawFormatError, parse_ratings


def test_ratings_take_the_film_id_of_the_header_above_them() -> None:
    lines = ["1:\n", "10,3,2005-09-06\n", "11,5,2005-05-13\n", "2:\n", "10,1,2004-01-31\n"]

    assert list(parse_ratings(lines)) == [
        Rating(1, 10, 3, date(2005, 9, 6)),
        Rating(1, 11, 5, date(2005, 5, 13)),
        Rating(2, 10, 1, date(2004, 1, 31)),
    ]


def test_windows_line_endings_are_accepted() -> None:
    assert list(parse_ratings(["1:\r\n", "10,3,2005-09-06\r\n"])) == [
        Rating(1, 10, 3, date(2005, 9, 6))
    ]


def test_film_with_no_ratings_yields_nothing() -> None:
    assert list(parse_ratings(["1:\n", "2:\n"])) == []


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("0:", "film ID"),
        ("abc:", "film ID"),
        ("", "expected user,stars,date"),
        ("10,3", "expected user,stars,date"),
        ("10,3,2005-09-06,extra", "expected user,stars,date"),
        ("0,3,2005-09-06", "user ID"),
        ("x,3,2005-09-06", "user ID"),
        ("10,0,2005-09-06", "stars"),
        ("10,6,2005-09-06", "stars"),
        ("10,4.5,2005-09-06", "stars"),
        ("10,3,2005-13-01", "invalid date"),
        ("10,3,not-a-date", "invalid date"),
    ],
)
def test_malformed_lines_raise_with_their_line_number(line: str, reason: str) -> None:
    with pytest.raises(RawFormatError, match=reason) as caught:
        list(parse_ratings(["1:\n", f"{line}\n"]))

    assert caught.value.line_number == 2


def test_rating_before_any_header_raises() -> None:
    with pytest.raises(RawFormatError, match="before any film header"):
        list(parse_ratings(["10,3,2005-09-06\n"]))


ratings = st.builds(
    Rating,
    film_id=st.integers(min_value=1, max_value=17_770),
    user_id=st.integers(min_value=1, max_value=3_000_000),
    stars=st.integers(min_value=1, max_value=5),
    rated_on=st.dates(min_value=date(1999, 1, 1), max_value=date(2006, 12, 31)),
)


@given(st.lists(ratings))
def test_writing_ratings_as_raw_lines_and_parsing_returns_them(items: list[Rating]) -> None:
    lines: list[str] = []
    for rating in items:
        lines += [f"{rating.film_id}:\n", f"{rating.user_id},{rating.stars},{rating.rated_on}\n"]

    assert list(parse_ratings(lines)) == items
