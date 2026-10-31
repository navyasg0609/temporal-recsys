from dataclasses import replace

from temporal_recsys.data.quality import ExpectedData, QualityReport, find_problems

CLEAN = QualityReport(
    rows=3,
    films=2,
    users=2,
    duplicate_ratings=0,
    first_date="2004-01-31",
    last_date="2005-09-06",
    rows_per_stars={1: 1, 3: 1, 5: 1},
    rows_per_month={"2004-01": 1, "2005-09": 2},
)
EXPECTED = ExpectedData(rows=3, films=2, first_date="2004-01-31", last_date="2005-09-06")


def test_clean_data_has_no_problems() -> None:
    assert find_problems(CLEAN, EXPECTED) == []


def test_expected_values_are_optional() -> None:
    assert find_problems(replace(CLEAN, rows_per_month={"2004-01": 3}, rows=3)) == []


def test_duplicates_are_reported() -> None:
    assert find_problems(replace(CLEAN, duplicate_ratings=2)) == ["2 duplicate film and user pairs"]


def test_invalid_stars_are_reported() -> None:
    assert find_problems(replace(CLEAN, rows_per_stars={0: 1, 3: 1, 6: 1})) == [
        "invalid star values: [0, 6]"
    ]


def test_months_that_do_not_add_up_are_reported() -> None:
    assert find_problems(replace(CLEAN, rows_per_month={"2004-01": 1})) == [
        "monthly row counts do not add up to the total"
    ]


def test_every_mismatch_with_the_expected_data_is_reported() -> None:
    report = replace(CLEAN, rows_per_month={"2004-01": 2}, rows=2, last_date="2005-09-07")

    assert find_problems(report, EXPECTED) == [
        "rows: found 2, expected 3",
        "last_date: found 2005-09-07, expected 2005-09-06",
    ]
