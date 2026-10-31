from importlib.metadata import version

import temporal_recsys


def test_version_matches_installed_metadata() -> None:
    assert temporal_recsys.__version__ == version("temporal-recsys")
