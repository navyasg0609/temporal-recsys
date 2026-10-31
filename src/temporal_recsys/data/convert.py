"""Convert raw rating files to Parquet."""

import logging
from collections.abc import Sequence
from itertools import batched
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from temporal_recsys.data.parser import Rating, RawFormatError, parse_ratings

LOGGER = logging.getLogger(__name__)

RATING_SCHEMA = pa.schema(
    [
        pa.field("film_id", pa.int32(), nullable=False),
        pa.field("user_id", pa.int32(), nullable=False),
        pa.field("stars", pa.int8(), nullable=False),
        pa.field("rated_on", pa.date32(), nullable=False),
    ]
)
DEFAULT_BATCH_SIZE = 1_000_000


def convert_raw_file(
    raw_path: Path, parquet_path: Path, batch_size: int = DEFAULT_BATCH_SIZE
) -> int:
    """Convert one raw file and return its row count.

    A finished output is kept, so a run that stops partway can simply be restarted.
    Output appears at `parquet_path` only once complete.
    """
    if parquet_path.exists():
        LOGGER.info("Keeping finished file %s", parquet_path)
        return int(pq.read_metadata(parquet_path).num_rows)

    temporary_path = parquet_path.with_name(f".{parquet_path.name}.tmp")
    rows = 0
    try:
        with (
            raw_path.open(encoding="ascii") as lines,
            pq.ParquetWriter(temporary_path, RATING_SCHEMA) as writer,
        ):
            for batch in batched(parse_ratings(lines), batch_size, strict=False):
                writer.write_table(_to_table(batch))
                rows += len(batch)
                LOGGER.info("%s: %d rows", raw_path.name, rows)
    except RawFormatError as error:
        error.add_note(f"in {raw_path}")
        raise

    temporary_path.replace(parquet_path)
    return rows


def convert_raw_files(raw_paths: Sequence[Path], output_dir: Path) -> int:
    """Convert each raw file to one Parquet file and return the total row count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return sum(
        convert_raw_file(raw_path, output_dir / f"{raw_path.stem}.parquet")
        for raw_path in raw_paths
    )


def _to_table(batch: Sequence[Rating]) -> pa.Table:
    film_ids, user_ids, stars, rated_on = zip(*batch, strict=True)
    return pa.Table.from_arrays(
        [
            pa.array(film_ids, type=pa.int32()),
            pa.array(user_ids, type=pa.int32()),
            pa.array(stars, type=pa.int8()),
            pa.array(rated_on, type=pa.date32()),
        ],
        schema=RATING_SCHEMA,
    )
