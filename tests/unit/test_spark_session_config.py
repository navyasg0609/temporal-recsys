import pytest

from temporal_recsys.data.spark import create_spark_session


@pytest.mark.parametrize("cores", [0, -1])
def test_core_count_below_one_is_rejected(cores: int) -> None:
    with pytest.raises(ValueError, match="at least one core"):
        create_spark_session("test", cores=cores)
