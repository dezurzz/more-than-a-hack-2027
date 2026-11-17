"""MS-TCN stage -- where two models disagree, and which one DFL says to trust there.

    python -m pipeline.disagree --build-reference   # DFL only: models/disagree_reference.json
    python -m pipeline.disagree                     # BAR-ATH, no ground truth: counts only

pipeline.drift found the models with the touch features (tcn4, tcn6) calling about 2 points of
the ATD's frames possession where tcn2 calls them dead ball. This finds where. Every graded frame
goes in one situation bucket, read off the tracking alone:

    ball_out    the ball is seen beyond a line
    gap_edge    the ball is missing, in a gap of GAP_LONG_S or more, within EDGE_S of either end
                (where a stoppage's start and end are timed)
    gap_long    the rest of such a gap (mostly the middle of a stoppage on the ATD)
    gap_short   the ball is missing, in a shorter gap
    rule_D      the ball is seen and the rules call it dead
    open        everything else

and, per bucket, the frames where model A and model B disagree are counted by kind (A says D and
B says H/A, the reverse, or H against A), plus each model's dead-ball share and its stoppages.
On damaged held-out DFL (challenge labels, the ATD-fitted damage) the same counts carry the truth
too: in each bucket and kind, how often A is right, B is right, or neither. The ATD run prints
counts and shares only and reads no ground truth, not even the public window.
"""

import argparse
import json
import os

import numpy as np

from pipeline import features
from pipeline.atd_stats import graded_mask
from pipeline.config import ROOT, Config
from pipeline.decode import decode, flipped_probabilities
from pipeline.labels import label_match
from pipeline.runs import value_runs
from pipeline.train_tcn import MODELS_DIR, VALIDATION_SEEDS, fold_model_path

REFERENCE = os.path.join(MODELS_DIR, "disagree_reference.json")
MODEL_A, MODEL_B = "tcn2", "tcn6"
DECODE_SWITCH, DECODE_MIN_S = 4.0, 2.0   # as pipeline.run_tcn
GAP_LONG_S = 2.0
EDGE_S = 2.0
BUCKETS = ("open", "rule_D", "gap_short", "gap_edge", "gap_long", "ball_out")
KINDS = ("A_D_B_poss", "A_poss_B_D", "H_vs_A")


def buckets(table, halves, fps=25):
    """Bucket index (into BUCKETS) per frame, from the feature table."""
    seen = table[:, features.FEATURES.index("ball_seen")] > 0
    out = np.zeros(len(table), dtype=np.int8)
    out[table[:, features.FEATURES.index("rule_D")] > 0] = BUCKETS.index("rule_D")
    for start, stop, value in value_runs(seen):
        if not value:
            if stop - start < GAP_LONG_S * fps:
                out[start:stop] = BUCKETS.index("gap_short")
                continue
            edge = int(EDGE_S * fps)
            out[start:stop] = BUCKETS.index("gap_edge")
            out[start + edge:max(start + edge, stop - edge)] = BUCKETS.index("gap_long")
    out[seen & (table[:, features.FEATURES.index("ball_out")] > 0)] = BUCKETS.index("ball_out")
    return out


def kinds(a, b):
    """Kind index (into KINDS) per frame where a != b, -1 where they agree."""
    out = np.full(len(a), -1, dtype=np.int8)
    poss_a, poss_b = np.isin(a, ("H", "A")), np.isin(b, ("H", "A"))
    out[(a == "D") & poss_b] = 0
    out[poss_a & (b == "D")] = 1
    out[poss_a & poss_b & (a != b)] = 2
    return out


def stoppages(labels, halves):
    """D spells inside the halves, as lengths in frames."""
    out = []
    for _, first, last in halves:
        out += [stop - start for start, stop, v in value_runs(labels[first:last + 1]) if v == "D"]
    return out


def tally(a, b, halves, graded, bucket, truth=None, fps=25):
    """Counts of one match; with truth, also who is right in each bucket and kind."""
    kind = kinds(a, b)
    out = {"frames": int(graded.sum()), "minutes": float(graded.sum()) / fps / 60.0,
           "bucket_frames": [int(np.sum(graded & (bucket == i))) for i in range(len(BUCKETS))],
           "D_frames": {}, "stoppages": {}}
    for name, labels in (("A", a), ("B", b)) + ((("truth", truth),) if truth is not None else ()):
        out["D_frames"][name] = [int(np.sum(graded & (bucket == i) & (labels == "D")))
                                 for i in range(len(BUCKETS))]
        out["stoppages"][name] = stoppages(labels, halves)
    cells = {}
    for i, bucket_name in enumerate(BUCKETS):
        for k, kind_name in enumerate(KINDS):
            mask = graded & (bucket == i) & (kind == k)
            cell = {"n": int(mask.sum())}
            if truth is not None:
                cell["A_right"] = int(np.sum(a[mask] == truth[mask]))
                cell["B_right"] = int(np.sum(b[mask] == truth[mask]))
            cells[bucket_name + "|" + kind_name] = cell
    out["cells"] = cells
    return out


def merge(tallies):
    """Sum of several tallies."""
    out = {"frames": sum(t["frames"] for t in tallies),
           "minutes": sum(t["minutes"] for t in tallies),
           "bucket_frames": np.sum([t["bucket_frames"] for t in tallies], axis=0).tolist(),
           "D_frames": {}, "stoppages": {}, "cells": {}}
    for name in tallies[0]["D_frames"]:
        out["D_frames"][name] = np.sum([t["D_frames"][name] for t in tallies], axis=0).tolist()
        out["stoppages"][name] = sum((t["stoppages"][name] for t in tallies), [])
    for key in tallies[0]["cells"]:
        out["cells"][key] = {field: sum(t["cells"][key][field] for t in tallies)
                             for field in tallies[0]["cells"][key]}
    return out


def _labels(predictor, table, halves, fps):
    return decode(flipped_probabilities(predictor, table, halves), halves,
                  DECODE_SWITCH, DECODE_MIN_S, fps)


def build_reference(matches, model_a=MODEL_A, model_b=MODEL_B):
    """Tally of fold model A against fold model B on damaged held-out DFL, with the truth."""
    from pipeline.atd_noise import damage
    from pipeline.dfl_events import load_relabelled
    from pipeline.robustness import SCENARIOS
    from pipeline.tcn import Predictor, deterministic
    deterministic()
    spec = SCENARIOS["atd_fit"][0]   # needs models/noise_atd.json
    tallies = []
    for match in matches:
        tracking, gt, halves, cfg = load_relabelled(match)
        cfg = cfg.replace(PAKAI_SMART_TAGGING=False)
        a_model = Predictor.load(fold_model_path(model_a, match))
        b_model = Predictor.load(fold_model_path(model_b, match))
        for seed in VALIDATION_SEEDS:
            copy = damage(tracking, gt, halves, spec, seed=seed)
            rules = label_match(copy, [], halves, cfg).labels
            table = features.compute(copy, halves, cfg, rule_labels=rules)
            tallies.append(tally(_labels(a_model, table, halves, cfg.FPS),
                                 _labels(b_model, table, halves, cfg.FPS), halves, gt != "X",
                                 buckets(table, halves, cfg.FPS), truth=gt, fps=cfg.FPS))
        print("reference: {}".format(match), flush=True)
    return {"matches": list(matches), "A": model_a, "B": model_b, "tally": merge(tallies)}


def run_on_atd(model_a, model_b):
    """Tally of final model A against final model B on BAR-ATH. Imports BAR-ATH loaders only here."""
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
    a = _labels(Predictor.load(os.path.join(MODELS_DIR, model_a + ".pt")), table, halves, cfg.FPS)
    b = _labels(Predictor.load(os.path.join(MODELS_DIR, model_b + ".pt")), table, halves, cfg.FPS)
    return tally(a, b, halves, graded, buckets(table, halves, cfg.FPS), fps=cfg.FPS)


def _share(n, total):
    return n / float(total) if total else 0.0


def print_report(atd, reference):
    dfl = reference["tally"]
    a, b = reference["A"], reference["B"]
    print("\nA. situations: share of graded frames, and share each model calls dead ball")
    print("  {:<10} {:>13} {:>13} {:>13} {:>13}".format(
        "", "frames DFL/ATD", a + " D DFL/ATD", b + " D DFL/ATD", "truth D DFL"))
    for i, name in enumerate(BUCKETS):
        n_dfl, n_atd = dfl["bucket_frames"][i], atd["bucket_frames"][i]
        print("  {:<10} {:>6.3f} {:>6.3f} {:>6.3f} {:>6.3f} {:>6.3f} {:>6.3f} {:>13.3f}".format(
            name, _share(n_dfl, dfl["frames"]), _share(n_atd, atd["frames"]),
            _share(dfl["D_frames"]["A"][i], n_dfl), _share(atd["D_frames"]["A"][i], n_atd),
            _share(dfl["D_frames"]["B"][i], n_dfl), _share(atd["D_frames"]["B"][i], n_atd),
            _share(dfl["D_frames"]["truth"][i], n_dfl)))

    print("\nB. where they disagree (share of all graded frames), and who DFL says is right there")
    print("  {:<10} {:<12} {:>8} {:>8}   {:>8} {:>8} {:>8}".format(
        "", "kind", "DFL", "ATD", a + " ok", b + " ok", "neither"))
    labels = {"A_D_B_poss": "{} D, {} H/A".format(a, b), "A_poss_B_D": "{} H/A, {} D".format(a, b),
              "H_vs_A": "H vs A"}
    for bucket_name in BUCKETS:
        for kind_name in KINDS:
            d = dfl["cells"][bucket_name + "|" + kind_name]
            n_atd = atd["cells"][bucket_name + "|" + kind_name]["n"]
            if not d["n"] and not n_atd:
                continue
            neither = d["n"] - d["A_right"] - d["B_right"]
            print("  {:<10} {:<16} {:>6.4f} {:>8.4f}   {:>8.2f} {:>8.2f} {:>8.2f}".format(
                bucket_name, labels[kind_name], _share(d["n"], dfl["frames"]),
                _share(n_atd, atd["frames"]), _share(d["A_right"], d["n"]),
                _share(d["B_right"], d["n"]), _share(neither, d["n"])))

    print("\nC. stoppages: per 10 min, median and p90 length (s)")
    rows = [("DFL", dfl, ("A", "B", "truth")), ("ATD", atd, ("A", "B"))]
    for where, t, names in rows:
        for name in names:
            lengths = np.array(t["stoppages"][name], dtype=float) / 25.0
            label = {"A": a, "B": b, "truth": "truth"}[name]
            print("  {:<4} {:<6} {:>7.2f} /10min   median {:>5.1f}   p90 {:>5.1f}".format(
                where, label, len(lengths) / t["minutes"] * 10,
                np.median(lengths) if len(lengths) else 0.0,
                np.percentile(lengths, 90) if len(lengths) else 0.0))


if __name__ == '__main__':
    from pipeline.dfl import available_matches
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build-reference", action="store_true",
                        help="DFL only: write models/disagree_reference.json")
    parser.add_argument("--a", default=MODEL_A, help="first run name")
    parser.add_argument("--b", default=MODEL_B, help="second run name")
    args = parser.parse_args()

    if args.build_reference:
        reference = build_reference(available_matches(), args.a, args.b)
        with open(REFERENCE, "w") as handle:
            json.dump(reference, handle)
        print("wrote {}".format(os.path.relpath(REFERENCE, ROOT)))
        raise SystemExit(0)

    with open(REFERENCE) as handle:
        reference = json.load(handle)
    print_report(run_on_atd(reference["A"], reference["B"]), reference)
