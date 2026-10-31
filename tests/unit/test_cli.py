from pathlib import Path

import pytest

from temporal_recsys.cli import main
from temporal_recsys.data.pipeline import find_raw_files


def test_synthetic_command_writes_a_raw_file(tmp_path: Path) -> None:
    assert main(["synthetic", "--output-dir", str(tmp_path)]) == 0

    assert find_raw_files(tmp_path) == [tmp_path / "combined_data_1.txt"]


def test_missing_raw_files_raise(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="combined_data"):
        find_raw_files(tmp_path)
