"""Write the user IDs a load test requests, into the git-ignored artifacts folder.

Run from the repository root: `uv run python deploy/load/sample_users.py artifacts`.
Nine in ten are drawn from users with any rating before the cutoff, so both the model and
the popular-films path are exercised. One in ten are IDs no user has. The list is derived
from rating data, so it stays out of git like the bundle (decision 038).
"""

import json
import sys
from pathlib import Path

import numpy as np

from temporal_recsys.serving.bundle import load_bundle

KNOWN_USERS = 9_000
UNKNOWN_USERS = 1_000
SEED = 2026


def main(artifacts: Path) -> None:
    bundle = load_bundle(artifacts / "bundle.npz")
    rng = np.random.default_rng(SEED)
    known = rng.choice(bundle.seen_user_ids, size=KNOWN_USERS, replace=False)
    first_unused = int(bundle.seen_user_ids.max()) + 1
    unknown = np.arange(first_unused, first_unused + UNKNOWN_USERS)
    users = rng.permutation(np.concatenate([known, unknown])).tolist()
    (artifacts / "users.json").write_text(json.dumps(users), encoding="utf-8")
    in_model = np.isin(known, bundle.user_ids).sum()
    print(f"wrote {len(users)} IDs: {in_model} with model vectors, {UNKNOWN_USERS} unknown")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
