from pathlib import Path

import pytest

from temporal_recsys.cli import main


def test_the_sample_run_finds_the_planted_answer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "run"

    assert main(["reproduce", "--sample", "--output-dir", str(output_dir)]) == 0

    printed = capsys.readouterr().out
    assert "Planted answer found" in printed
    # Two models: popularity on 5 windows, and ALS on 5 windows with one seed, at 3 cutoffs.
    for model in ("popularity", "numpy-als"):
        arms = (output_dir / model / "arms.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(arms) == 15


def test_the_sample_run_refuses_an_existing_output_dir(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError, match="already exists"):
        main(["reproduce", "--sample", "--output-dir", str(tmp_path)])
