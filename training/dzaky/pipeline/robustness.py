"""MS-TCN stage -- how much of the win over the rules survives damage the model was not trained on.

    python -m pipeline.robustness                    # every scenario, fold models of run tcn1
    python -m pipeline.robustness --name tcn1 --scenarios live_only heavy_missing

Each leave-one-match-out fold model (models/<name>_<match>.pt, from pipeline.train_tcn) is scored
on its own held-out match, damaged per scenario with seeds 0..2; the rules are scored on the very
same copies. Training used NoiseSpec.sample(), so the scenarios below that leave its ranges are
the real test -- above all live_only, where the missing ball says nothing about a stoppage, and
which is what the model would face if the ATD tracker loses the ball for reasons unrelated to
play stopping.
"""

import argparse
import json
import os

import numpy as np

from pipeline import features
from pipeline.atd_noise import NoiseSpec, damage
from pipeline.config import ROOT
from pipeline.dfl import available_matches
from pipeline.dfl_events import LABELS, loader
from pipeline.external_eval import score
from pipeline.labels import label_match
from pipeline.tcn import Predictor, deterministic
from pipeline.train_tcn import MODELS_DIR, VALIDATION_SEEDS, fold_model_path

# name -> (NoiseSpec or "sample" or None for clean, what it tests). Training ranges, for reference:
# BALL_MISSING 0.30-0.50, SHARE_ON_DEAD 0.3-0.8, BALL_JITTER 0.2-0.8, PLAYER_JITTER 0.1-0.5,
# PLAYER_DROPS_PER_MIN 0-1.5.
DEFAULT = NoiseSpec()
SCENARIOS = {
    "default": (DEFAULT, "the validation damage of train_tcn (in range)"),
    "varied": ("sample", "a fresh NoiseSpec.sample() per copy (in range)"),
    "clean": (None, "no damage at all (out of range: ball always seen)"),
    "live_only": (DEFAULT.replace(SHARE_ON_DEAD=0.0),
                  "every long gap in live play; none anchored on stoppages (out of range)"),
    "dead_only": (DEFAULT.replace(SHARE_ON_DEAD=1.0, DEAD_SPILL_S=0.5),
                  "long gaps only on stoppages, tight (out of range: missing ~ dead)"),
    "light_missing": (DEFAULT.replace(BALL_MISSING=0.15), "ball missing 15% (out of range)"),
    "heavy_missing": (DEFAULT.replace(BALL_MISSING=0.60), "ball missing 60% (out of range)"),
    "jitter": (DEFAULT.replace(BALL_JITTER=1.5, PLAYER_JITTER=0.8),
               "positions much noisier (out of range)"),
    "player_drops": (DEFAULT.replace(PLAYER_DROPS_PER_MIN=4.0, PLAYER_DROP_MAX_S=10.0),
                     "players out of view far more often (out of range)"),
}

# The damage pipeline.atd_stats measured on the ATD, if it has been run; plus the same with half
# the stoppage share, in case that estimate -- the least certain one that matters -- is off.
_FITTED = os.path.join(ROOT, "models", "noise_atd.json")
if os.path.isfile(_FITTED):
    with open(_FITTED) as _handle:
        _ATD = NoiseSpec(**json.load(_handle)["estimates"])
    SCENARIOS["atd_fit"] = (_ATD, "the NoiseSpec pipeline.atd_stats fitted to the ATD")
    SCENARIOS["atd_fit_live"] = (_ATD.replace(SHARE_ON_DEAD=_ATD.SHARE_ON_DEAD / 2),
                                 "atd_fit with half the gaps moved from stoppages to live play")
    SCENARIOS["atd_fit_carried"] = (_ATD.replace(CARRIED_ON_DEAD=0.5),
                                    "atd_fit with half the visible stoppages' ball carried")


def copies(tracking, gt, halves, scenario):
    """[(seed, damaged Tracking)] for one scenario."""
    spec = SCENARIOS[scenario][0]
    if spec is None:
        return [(None, tracking)]
    out = []
    for seed in VALIDATION_SEEDS:
        used = NoiseSpec.sample(np.random.default_rng(10_000 + seed)) if spec == "sample" else spec
        out.append((seed, damage(tracking, gt, halves, used, seed=seed)))
    return out


def run(name, matches, scenarios, labels="dfl"):
    """{scenario: {match: {model, rules, ball_missing}}} with spellF1 / macroF1 means over seeds."""
    deterministic()
    report = {scenario: {} for scenario in scenarios}
    for match in matches:
        predictor = Predictor.load(fold_model_path(name, match))
        tracking, gt, halves, cfg = loader(labels)(match)
        graded = gt != "X"
        for scenario in scenarios:
            model, rules, missing = [], [], []
            for _, damaged in copies(tracking, gt, halves, scenario):
                rule_labels = label_match(damaged, [], halves, cfg).labels
                table = features.compute(damaged, halves, cfg, rule_labels=rule_labels)
                model.append(score(gt, predictor.labels(table, halves, cfg)))
                rules.append(score(gt, rule_labels))
                missing.append(float(np.isnan(damaged.ball[graded, 0]).mean()))
            report[scenario][match] = {
                "model": {k: float(np.mean([s[k] for s in model])) for k in model[0]},
                "rules": {k: float(np.mean([s[k] for s in rules])) for k in rules[0]},
                "ball_missing": float(np.mean(missing))}
        print("scored {}".format(match), flush=True)
    return report


def print_table(report):
    print("\n{:<14} {:>6} | {:>8} {:>8} {:>7} {:>5} | {:>8} {:>8} | {}".format(
        "scenario", "miss", "TCN", "rules", "gap", "wins", "TCN", "rules",
        "spellF1 ... | macroF1"))
    for scenario, per_match in report.items():
        values = list(per_match.values())
        model = [v["model"]["spell_f1_mean"] for v in values]
        rules = [v["rules"]["spell_f1_mean"] for v in values]
        wins = sum(m > r for m, r in zip(model, rules))
        print("{:<14} {:>5.0%} | {:>8.4f} {:>8.4f} {:>+7.3f} {:>2}/{:<2} | {:>8.4f} {:>8.4f} | {}"
              .format(scenario, np.mean([v["ball_missing"] for v in values]), np.mean(model),
                      np.mean(rules), np.mean(model) - np.mean(rules), wins, len(values),
                      np.mean([v["model"]["macro_f1"] for v in values]),
                      np.mean([v["rules"]["macro_f1"] for v in values]),
                      SCENARIOS[scenario][1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="tcn1", help="run name given to pipeline.train_tcn")
    parser.add_argument("--matches", nargs="*", help="held-out matches; default every one")
    parser.add_argument("--scenarios", nargs="*", choices=sorted(SCENARIOS),
                        help="default: all, in the order listed in SCENARIOS")
    parser.add_argument("--labels", choices=LABELS, default="dfl",
                        help="score against DFL's possession or the challenge's")
    args = parser.parse_args()

    report = run(args.name, args.matches or available_matches(), args.scenarios or list(SCENARIOS),
                 args.labels)
    print_table(report)
    path = os.path.join(MODELS_DIR, "robustness_{}{}.json".format(
        args.name, "" if args.labels == "dfl" else "_" + args.labels))
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)
    print("\nwrote {}".format(os.path.relpath(path, ROOT)))
