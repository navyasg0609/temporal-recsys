"""Check a Kaggle GPU notebook before any neural model code depends on it.

Attach the Netflix Prize dataset, choose a GPU, and run this file. It prints only
machine details, counts and year ranges, never film titles.
"""

import os
import platform
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import torch

INPUT_ROOT = Path("/kaggle/input")
TITLE_FILE_PATTERN = "movie_titles*"
TITLE_LINE = r"^(\d+),(\d{4}|NULL),(.+)$"
EXPECTED_FILMS = 17_770
MATRIX_SIZE = 4_096


def print_machine() -> None:
    meminfo = Path("/proc/meminfo").read_text().splitlines()
    memory_kb = next(int(line.split()[1]) for line in meminfo if line.startswith("MemTotal"))
    print(f"python: {platform.python_version()}")
    print(f"cpu_count: {os.cpu_count()}")
    print(f"memory_gb: {memory_kb / 1024**2:.1f}")
    print(f"free_disk_gb: {shutil.disk_usage('/kaggle/working').free / 1024**3:.1f}")


def print_gpu() -> None:
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return
    print(f"gpu_count: {torch.cuda.device_count()}")
    print(f"gpu_name: {torch.cuda.get_device_name(0)}")
    memory = torch.cuda.get_device_properties(0).total_memory
    print(f"gpu_memory_gb: {memory / 1024**3:.1f}")

    # Listing a GPU is not enough, because an old GPU can fail on the first real work.
    matrix = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device="cuda:0")
    start = time.perf_counter()
    product = matrix @ matrix
    product.sum().item()
    print(f"gpu_matrix_product_seconds: {time.perf_counter() - start:.3f}")


def print_title_file() -> None:
    paths = sorted(INPUT_ROOT.rglob(TITLE_FILE_PATTERN))
    print(f"title_files_found: {len(paths)}")
    if not paths:
        return

    # The file is not UTF-8 in every copy, and this encoding accepts any byte.
    lines = paths[0].read_text(encoding="latin-1").splitlines()
    matches = [re.match(TITLE_LINE, line) for line in lines]
    parsed = [match for match in matches if match is not None]
    years = Counter(match[2] for match in parsed)
    known_years = sorted(int(year) for year in years if year != "NULL")
    print(f"title_file_name: {paths[0].name}")
    print(f"title_lines: {len(lines)}")
    print(f"title_lines_matched: {len(parsed)}")
    print(f"distinct_film_ids: {len({match[1] for match in parsed})}")
    print(f"expected_films: {EXPECTED_FILMS}")
    print(f"titles_containing_a_comma: {sum(',' in match[3] for match in parsed)}")
    print(f"missing_years: {years['NULL']}")
    print(f"first_year: {known_years[0] if known_years else None}")
    print(f"last_year: {known_years[-1] if known_years else None}")


def main() -> int:
    print_machine()
    print_gpu()
    print_title_file()
    return 0


if __name__ == "__main__":
    sys.exit(main())
