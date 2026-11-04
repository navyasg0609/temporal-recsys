"""Every decimal number in the README and the report must appear in docs/CLAIMS.md.

Whole numbers are not checked: years, step sizes and counts would need a long list of
exceptions. Decimals are where results live, so a typo or an unrecorded number fails here.
"""

import re
from pathlib import Path

import pytest

LEDGER = Path("docs/CLAIMS.md")
PUBLISHED = [Path("README.md"), *sorted(Path("report").glob("*.tex"))]
# Dotted versions such as 2.3.0 have two dots, so they are a single match and skipped.
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)+(?![\w.]?\d)")

# Decimals that are settings or versions, not results.
NOT_RESULTS = {
    "3.13": "the Python version",
}


def unclaimed_numbers(text: str, ledger: str) -> list[str]:
    numbers = {match for match in NUMBER.findall(text) if match.count(".") == 1}
    claimed = set(NUMBER.findall(ledger))
    return sorted(numbers - claimed - NOT_RESULTS.keys())


def test_a_claimed_number_passes_and_an_unclaimed_one_fails() -> None:
    ledger = "| NDCG@10 | 0.0993 | ... | 1.32 times |"

    assert unclaimed_numbers("It scored 0.0993, and 1.32 times as many.", ledger) == []
    assert unclaimed_numbers("It scored 0.0994.", ledger) == ["0.0994"]


def test_part_of_a_longer_number_does_not_count_as_claimed() -> None:
    assert unclaimed_numbers("It scored 0.099.", "| 0.0993 |") == ["0.099"]


def test_versions_and_whole_numbers_are_not_checked() -> None:
    assert unclaimed_numbers("k6 2.3.0 ran 400 users on Python 3.13.", "") == []


@pytest.mark.parametrize("path", PUBLISHED, ids=str)
def test_every_published_decimal_is_in_the_claims_ledger(path: Path) -> None:
    ledger = LEDGER.read_text(encoding="utf-8")

    assert unclaimed_numbers(path.read_text(encoding="utf-8"), ledger) == []
