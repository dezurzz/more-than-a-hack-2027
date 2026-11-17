"""Build the PFF Match and feature caches (work_data/overnight_cache/pff) for the first N games. Aggregates/timings only.

    python scripts/build_pff_cache.py --n 16 --jobs 3 [--extra smooth]
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.external import dataset, pff  # noqa: E402


def build(game_id, extra):
    started = time.time()
    fs, match = dataset.pff_feature_set(game_id, extra=extra)
    return game_id, time.time() - started, len(fs.x), fs.x.shape[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--extra", default=None)
    args = parser.parse_args()
    ids = pff.game_ids()[:args.n]
    decision = pff.damage_decision()          # computed once, before the workers start
    print("damage decision: applies {} (m {:.3f})".format(decision["applies"], decision["m"]), flush=True)
    started = time.time()
    with ProcessPoolExecutor(max_workers=min(3, args.jobs)) as pool:
        for game_id, seconds, rows, width in pool.map(build, ids, [args.extra] * len(ids)):
            print("game #{}: {:.0f} s, {} feature rows x {}".format(ids.index(game_id) + 1, seconds, rows, width), flush=True)
    total = sum(os.path.getsize(os.path.join(pff.CACHE_DIR, f)) for f in os.listdir(pff.CACHE_DIR))
    print("{} games in {:.0f} s; cache dir {:.0f} MB".format(len(ids), time.time() - started, total / 1e6))


if __name__ == "__main__":
    main()
