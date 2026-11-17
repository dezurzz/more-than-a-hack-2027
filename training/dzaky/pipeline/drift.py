"""MS-TCN stage -- do the models see on the ATD what they saw on damaged DFL, and act the same?

    python -m pipeline.drift --build-reference   # DFL only: models/drift_reference.json
    python -m pipeline.drift                     # BAR-ATH, no ground truth: aggregates only

Written when the models with the touch features (tcn4, tcn6) beat tcn2 on held-out DFL but cut
the ATD into more, shorter possessions. Two questions, both answerable without the truth:

    A. features: per column, how far the ATD's mean sits from damaged DFL's (in DFL sd) and the
       share of ATD frames outside DFL's p1-p99. The touch group is always listed.
    B. behaviour: per model, after the same flips + Viterbi as run_tcn, the H/A/D shares,
       possessions per 10 min, median possession, direct H<->A switches per 10 min and the
       model's confidence -- on damaged held-out DFL (fold models) and on the ATD (final model).

A model whose behaviour moves between the two columns while its inputs moved is reacting to
damage it was not trained on. Like atd_stats, the ATD run prints aggregates only and reads no
ground truth, not even the public window.
"""

import argparse
import json
import os

import numpy as np

from pipeline import features
from pipeline.atd_stats import feature_stats, graded_mask
from pipeline.config import ROOT, Config
from pipeline.decode import decode, flipped_probabilities
from pipeline.dfl import available_matches, load_match
from pipeline.labels import label_match
from pipeline.runs import value_runs
from pipeline.train_tcn import MODELS_DIR, fold_model_path

REFERENCE = os.path.join(MODELS_DIR, "drift_reference.json")
MODELS = ("tcn2", "tcn4", "tcn6")
SEED = 0
DECODE_SWITCH, DECODE_MIN_S = 4.0, 2.0   # as pipeline.run_tcn
HESITANT = 0.6                           # top probability below this counts as hesitant
TOUCH = features.FEATURES[features.N_BASE:]


def behaviour(probabilities, halves, graded, fps=25):
    """Aggregates of the decoded labels and of the probabilities, over the graded frames."""
    labels = decode(probabilities, halves, DECODE_SWITCH, DECODE_MIN_S, fps)
    minutes = graded.sum() / fps / 60.0
    spells, switches = [], 0
    for _, first, last in halves:
        runs = value_runs(labels[first:last + 1])
        spells += [b - a for a, b, v in runs if v in ("H", "A")]
        switches += sum(1 for (_, _, u), (_, _, v) in zip(runs, runs[1:])
                        if {u, v} == {"H", "A"})
    top = probabilities[graded].max(1)
    return {"share_H": float(np.mean(labels[graded] == "H")),
            "share_A": float(np.mean(labels[graded] == "A")),
            "share_D": float(np.mean(labels[graded] == "D")),
            "poss_per_10min": len(spells) / minutes * 10,
            "median_poss_s": float(np.median(spells)) / fps if spells else 0.0,
            "switch_per_10min": switches / minutes * 10,
            "mean_top": float(top.mean()),
            "share_hesitant": float(np.mean(top < HESITANT))}


def _mean(rows):
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def build_reference(matches, models=MODELS):
    """Feature distributions and model behaviour on held-out DFL damaged like the ATD."""
    from pipeline.robustness import SCENARIOS
    from pipeline.atd_noise import damage
    from pipeline.tcn import Predictor, deterministic
    deterministic()
    spec = SCENARIOS["atd_fit"][0]   # needs models/noise_atd.json
    tables, behaviours = [], {name: [] for name in models}
    for match in matches:
        tracking, gt, halves, cfg = load_match(match)
        copy = damage(tracking, gt, halves, spec, seed=SEED)
        cfg = cfg.replace(PAKAI_SMART_TAGGING=False)
        table = features.compute(copy, halves, cfg, label_match(copy, [], halves, cfg).labels)
        graded = gt != "X"
        tables.append(table[graded])
        for name in models:
            predictor = Predictor.load(fold_model_path(name, match))
            behaviours[name].append(behaviour(flipped_probabilities(predictor, table, halves),
                                              halves, graded, cfg.FPS))
        print("reference: {}".format(match), flush=True)
    rows = np.vstack(tables)
    return {"matches": list(matches), "features": feature_stats(rows, np.ones(len(rows), bool)),
            "behaviour": {name: _mean(rows) for name, rows in behaviours.items()}}


def run_on_atd(reference):
    """(feature rows, behaviour per model) on BAR-ATH. Imports BAR-ATH loaders only here."""
    import sys
    from pipeline.data import load_tracking
    from pipeline.tcn import Predictor, deterministic
    sys.path.insert(0, ROOT)
    from games import GAME, half_video_blocks

    deterministic()
    cfg = Config().replace(PAKAI_SMART_TAGGING=False)
    tracking = load_tracking(cfg=cfg)
    halves = half_video_blocks(GAME)
    graded = graded_mask(len(tracking), halves)
    table = features.compute(tracking, halves, cfg,
                             rule_labels=label_match(tracking, [], halves, cfg).labels)
    atd = feature_stats(table, graded)

    rows = []
    for i, name in enumerate(features.FEATURES):
        ref = reference["features"][name]
        column = table[graded, i]
        outside = float(np.mean((column < ref["p"][0] - 1e-6) | (column > ref["p"][4] + 1e-6)))
        rows.append((name, atd[name]["mean"], ref["mean"],
                     (atd[name]["mean"] - ref["mean"]) / (ref["std"] + 1e-6), outside))
    models = {}
    for name in reference["behaviour"]:
        path = os.path.join(MODELS_DIR, name + ".pt")
        if os.path.isfile(path):
            probabilities = flipped_probabilities(Predictor.load(path), table, halves)
            models[name] = behaviour(probabilities, halves, graded, cfg.FPS)
    return rows, models


def print_report(rows, models, reference, shift_limit=0.5, outside_limit=0.10):
    print("\nA. features (ATD vs damaged DFL)   * = shifted or often outside DFL's p1-p99")
    print("  {:<22} {:>9} {:>9} {:>9} {:>9}".format("", "ATD", "DFL", "shift sd", "outside"))
    for name, atd, dfl, shift, outside in sorted(rows, key=lambda row: -abs(row[3])):
        flagged = abs(shift) > shift_limit or outside > outside_limit
        if flagged or name in TOUCH:
            print("  {:<22} {:>9.3f} {:>9.3f} {:>+9.2f} {:>9.3f} {}{}".format(
                name, atd, dfl, shift, outside, "*" if flagged else " ",
                "  touch" if name in TOUCH else ""))
    print("\nB. behaviour after flips + Viterbi ({:g}, {:g} s)".format(DECODE_SWITCH, DECODE_MIN_S))
    keys = list(next(iter(reference["behaviour"].values())))
    print("  {:<18}".format("") + "".join("{:>18}".format(name + " DFL / ATD") for name in models))
    for key in keys:
        print("  {:<18}".format(key) + "".join(
            "{:>9.3f} {:>8.3f}".format(reference["behaviour"][name][key], models[name][key])
            for name in models))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build-reference", action="store_true",
                        help="DFL only: write models/drift_reference.json")
    parser.add_argument("--models", nargs="+", default=list(MODELS),
                        help="run names to compare (fold models for DFL, final ones for the ATD)")
    args = parser.parse_args()

    if args.build_reference:
        reference = build_reference(available_matches(), args.models)
        with open(REFERENCE, "w") as handle:
            json.dump(reference, handle, indent=1)
        print("wrote {}".format(os.path.relpath(REFERENCE, ROOT)))
        raise SystemExit(0)

    with open(REFERENCE) as handle:
        reference = json.load(handle)
    rows, models = run_on_atd(reference)
    print_report(rows, models, reference)
