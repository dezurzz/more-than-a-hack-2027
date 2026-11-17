"""Run the rule-based pipeline on the provided match and write a submission.

    python -m pipeline.run                       # -> evaluation/results/dzaky_rules.csv
    python -m pipeline.run --name my_team --no-smart-tagging
    python evaluation/validate_submission.py evaluation/results/dzaky_rules.csv

Deterministic: no randomness anywhere, so the same data and the same Config give the same file,
which is what SUBMISSION.md requires. Everything printed is an aggregate -- counts and durations --
never a row of tracking data.
"""

import argparse
import collections
import os
import sys

import numpy as np

from pipeline.config import ROOT, Config
from pipeline.data import load_events, load_tracking
from pipeline.labels import label_match
from pipeline.runs import value_runs

sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402
from starter.baseline import write_submission  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def summarise(result, halves, fps):
    """Print what the file claims, half by half, plus how the restarts were decided."""
    for half, first, last in halves:
        window = result.labels[first:last + 1]
        spells = [(a, b, v) for a, b, v in value_runs(window) if v in ("H", "A")]
        stoppages = [(a, b) for a, b, v in value_runs(window) if v == "D"]
        mean = (np.mean([b - a for a, b, _ in spells]) / fps) if spells else 0.0
        print("half {}: H {:>6}  A {:>6}  D {:>6}   possessions {:>4} (mean {:.1f} s)"
              "   stoppages {:>4}".format(half, int(np.sum(window == "H")),
                                          int(np.sum(window == "A")),
                                          int(np.sum(window == "D")),
                                          len(spells), mean, len(stoppages)))
    kinds = collections.Counter(restart.kind for restart in result.restarts)
    sources = collections.Counter(restart.source for restart in result.restarts)
    print("restarts by kind:   {}".format(dict(sorted(kinds.items()))))
    print("restart team from:  {}".format(dict(sorted(sources.items()))))
    print("touches detected:   {}".format(len(result.touches)))
    live = sum(gap.live for gap in result.gaps)
    print("long ball gaps:     {}  ({} dead, {} vetoed as live play)".format(
        len(result.gaps), len(result.gaps) - live, live))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="dzaky_rules", help="submission file name, no .csv")
    parser.add_argument("--out", default=RESULTS_DIR, help="where to write")
    parser.add_argument("--no-smart-tagging", action="store_true",
                        help="ignore Smart Tagging entirely (ablation)")
    parser.add_argument("--no-cache", action="store_true", help="re-parse the raw ATD")
    args = parser.parse_args()

    cfg = Config(PAKAI_SMART_TAGGING=not args.no_smart_tagging)
    tracking = load_tracking(cfg=cfg, cache=not args.no_cache)
    events = [] if args.no_smart_tagging else load_events()
    halves = half_video_blocks(GAME)

    result = label_match(tracking, events, halves, cfg)
    path = write_submission(os.path.join(args.out, args.name + ".csv"), result.labels)
    summarise(result, halves, cfg.FPS)
    print("\nwrote {}".format(os.path.relpath(path, ROOT)))
    print("check it:  python evaluation/validate_submission.py {}".format(
        os.path.relpath(path, ROOT)))
