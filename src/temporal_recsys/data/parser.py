"""Parse the Netflix Prize text format.

Ratings have no film column. A header line such as `123:` applies to every rating
after it, up to the next header, so lines must be read in order. See decision 012.
"""

from collections.abc import Iterable, Iterator
from datetime import date
from typing import NamedTuple

MIN_RATING = 1
MAX_RATING = 5


class Rating(NamedTuple):
    film_id: int
    user_id: int
    stars: int
    rated_on: date


class RawFormatError(ValueError):
    """A raw line does not match the expected format."""

    def __init__(self, line_number: int, line: str, reason: str) -> None:
        super().__init__(f"line {line_number}: {reason}: {line!r}")
        self.line_number = line_number


def parse_ratings(lines: Iterable[str]) -> Iterator[Rating]:
    """Yield each rating in `lines`, raising `RawFormatError` on any malformed line."""
    film_id: int | None = None
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\r\n")
        if line.endswith(":"):
            film_id = _parse_film_header(line, line_number)
        elif film_id is None:
            raise RawFormatError(line_number, line, "rating appears before any film header")
        else:
            yield _parse_rating(line, line_number, film_id)


def _parse_film_header(line: str, line_number: int) -> int:
    film_id = line[:-1]
    if not film_id.isdecimal() or int(film_id) < 1:
        raise RawFormatError(line_number, line, "film ID must be a positive integer")
    return int(film_id)


def _parse_rating(line: str, line_number: int, film_id: int) -> Rating:
    fields = line.split(",")
    if len(fields) != 3:
        raise RawFormatError(line_number, line, "expected user,stars,date")

    user_id, stars, rated_on = fields
    if not user_id.isdecimal() or int(user_id) < 1:
        raise RawFormatError(line_number, line, "user ID must be a positive integer")
    if not stars.isdecimal() or not MIN_RATING <= int(stars) <= MAX_RATING:
        raise RawFormatError(line_number, line, f"stars must be {MIN_RATING} to {MAX_RATING}")
    try:
        parsed_date = date.fromisoformat(rated_on)
    except ValueError as error:
        raise RawFormatError(line_number, line, "invalid date") from error

    return Rating(film_id, int(user_id), int(stars), parsed_date)
