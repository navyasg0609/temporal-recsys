from datetime import date
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from temporal_recsys.data.convert import convert_raw_file, convert_raw_files
from temporal_recsys.data.parser import RawFormatError

RAW_TEXT = "1:\n10,3,2005-09-06\n11,5,2005-05-13\n2:\n10,1,2004-01-31\n"


@pytest.fixture
def raw_path(tmp_path: Path) -> Path:
    path = tmp_path / "combined_data_1.txt"
    path.write_text(RAW_TEXT, encoding="ascii")
    return path


def test_converted_rows_match_the_raw_file(raw_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "out.parquet"

    rows = convert_raw_file(raw_path, output, batch_size=2)

    assert rows == 3
    assert pq.read_table(output).to_pylist() == [
        {"film_id": 1, "user_id": 10, "stars": 3, "rated_on": date(2005, 9, 6)},
        {"film_id": 1, "user_id": 11, "stars": 5, "rated_on": date(2005, 5, 13)},
        {"film_id": 2, "user_id": 10, "stars": 1, "rated_on": date(2004, 1, 31)},
    ]


def test_restart_keeps_a_finished_file(raw_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "out.parquet"
    convert_raw_file(raw_path, output)
    finished_at = output.stat().st_mtime_ns

    rows = convert_raw_file(raw_path, output)

    assert rows == 3
    assert output.stat().st_mtime_ns == finished_at


def test_failed_conversion_leaves_no_finished_file(tmp_path: Path) -> None:
    raw_path = tmp_path / "bad.txt"
    raw_path.write_text("1:\n10,9,2005-09-06\n", encoding="ascii")
    output = tmp_path / "out.parquet"

    with pytest.raises(RawFormatError) as caught:
        convert_raw_file(raw_path, output)

    assert not output.exists()
    assert str(raw_path) in caught.value.__notes__[0]


def test_each_raw_file_becomes_one_parquet_file(raw_path: Path, tmp_path: Path) -> None:
    second = tmp_path / "combined_data_2.txt"
    second.write_text("3:\n12,4,2003-02-01\n", encoding="ascii")
    output_dir = tmp_path / "parquet"

    assert convert_raw_files([raw_path, second], output_dir) == 4
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "combined_data_1.parquet",
        "combined_data_2.parquet",
    ]
