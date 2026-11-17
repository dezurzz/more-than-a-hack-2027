"""Tahap 3 -- score the rule-based pipeline on external matches, with the official scorer.

    python -m pipeline.external_eval                    # every downloaded DFL match, ATD-like damage
    python -m pipeline.external_eval --clean            # undamaged tracking, for comparison
    python -m pipeline.external_eval --seeds 3 --random # three damaged copies with varied NoiseSpec
    python -m pipeline.external_eval --set R_PZ=2.0 --set AMBANG_MATI=1.25

This is the loop thresholds (and later the MS-TCN) are chosen in: never on BAR-ATH. Labels come
from DFL's BallStatus / BallPossession (pipeline/dfl.py), so the numbers measure agreement with
DFL's possession definition, which is close to but not the competition's.
"""

import argparse
import os
import sys

import numpy as np

from pipeline.atd_noise import NoiseSpec, damage
from pipeline.config import ROOT
from pipeline.dfl import available_matches, load_match
from pipeline.labels import label_match

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

METRICS = ("spell_f1_mean", "macro_f1")


def score(gt_labels, pred_labels):
    """Official scorer on two label arrays: the subset of evaluate() the board ranks on."""
    result = pm.evaluate(pm.label_codes(gt_labels), pm.label_codes(pred_labels))
    return {metric: result[metric] for metric in METRICS}


def parse_overrides(pairs, cfg):
    """--set NAME=VALUE pairs applied to cfg, each value cast to the field's current type."""
    changes = {}
    for pair in pairs:
        name, value = pair.split("=", 1)
        current = getattr(cfg, name)
        changes[name] = (value.lower() in ("1", "true", "yes")) if isinstance(current, bool) \
            else type(current)(value)
    return cfg.replace(**changes)


def evaluate_match(match, clean=False, seeds=(0,), random_spec=False, overrides=()):
    """[(seed, scores, ball_missing_share)] for one match."""
    tracking, gt, halves, cfg = load_match(match)
    cfg = parse_overrides(overrides, cfg)
    graded = gt != "X"
    rows = []
    for seed in ([None] if clean else seeds):
        if seed is None:
            used = tracking
        else:
            spec = NoiseSpec.sample(np.random.default_rng(10_000 + seed)) if random_spec \
                else NoiseSpec()
            used = damage(tracking, gt, halves, spec, seed=seed)
        result = label_match(used, [], halves, cfg)
        rows.append((seed, score(gt, result.labels), float(np.isnan(used.ball[graded, 0]).mean())))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", nargs="*", help="DFL short ids; default every downloaded one")
    parser.add_argument("--clean", action="store_true", help="no damage")
    parser.add_argument("--seeds", type=int, default=1, help="damaged copies per match")
    parser.add_argument("--random", action="store_true", help="vary NoiseSpec per copy")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="override a Config field")
    args = parser.parse_args()

    matches = args.matches or available_matches()
    if not matches:
        sys.exit("no DFL match found in external/dfl/ -- see pipeline/dfl.py for the download")

    all_scores = []
    print("{:<8} {:>5} {:>10} {:>9} {:>9}".format("match", "seed", "ball miss", *METRICS))
    for match in matches:
        for seed, scores, missing in evaluate_match(match, args.clean, range(args.seeds),
                                                    args.random, args.set):
            all_scores.append(scores)
            print("{:<8} {:>5} {:>9.1%} {:>9.4f} {:>9.4f}".format(
                match, "-" if seed is None else seed, missing,
                *(scores[metric] for metric in METRICS)))
    print("{:<8} {:>5} {:>10} {:>9.4f} {:>9.4f}".format(
        "mean", "", "", *(np.mean([s[metric] for s in all_scores]) for metric in METRICS)))
