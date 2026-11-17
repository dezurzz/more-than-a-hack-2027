"""Run the trained TCN on the provided match and write a submission.

    python -m pipeline.run_tcn                         # tcn11 -> evaluation/results/dzaky_tcn11.csv
    python -m pipeline.run_tcn --model models/tcn6.pt --dead-model models/tcn2.pt --name dzaky_hybrid
    python -m pipeline.run_tcn --decode tidy           # one model, argmax + spell tidying
    python evaluation/validate_submission.py evaluation/results/dzaky_tcn11.csv

The model was trained on external data only (pipeline/train_tcn.py): Sportec Open DFL and, since
tcn9, SkillCorner Open Data; nothing from BAR-ATH was used to train or choose it. This is the only place the model meets the match.

The rule-based labels that go in as features are computed WITHOUT Smart Tagging, because that is
how they were computed in training -- the DFL matches have no Smart Tagging. Feeding the model
rule labels built with more evidence than it ever saw would shift its inputs.

Decoding (pipeline/decode.py) is chosen on held-out DFL, challenge-style labels, ATD-fitted
damage: probabilities averaged over the four team-swap / mirror flips, then Viterbi with a switch
price of DECODE_SWITCH and spells of at least DECODE_MIN_S -- +0.021 spellF1 over argmax + tidy,
better on all 7 matches, the same setting picked in every leave-one-match-out fold.

By default one model, tcn11 (4 stages, DFL + SkillCorner, 3000 steps), labels the match. Against
the old default, tcn6 + tcn2 combined before decoding (decode.combine), its predecessor tcn10 won
on held-out SkillCorner (+0.096, 20/20 matches) and stayed within the pre-set 0.01 on held-out DFL
(atd_fit -0.006); tcn11 then beat tcn10 (SkillCorner +0.003, 15/20; DFL atd_fit +0.006, 6/7).
Every comparison used a rule written before the run (kaggle/score_vs_hybrid.py); BAR-ATH decided
nothing. --dead-model still combines two models, e.g. to reproduce the old hybrid.

Deterministic: CPU, fixed threads, deterministic kernels, pinned torch (requirements-model.txt).
Everything printed is an aggregate -- counts and durations -- never a row of tracking data.
"""

import argparse
import os
import sys

import numpy as np

from pipeline import features
from pipeline.decode import combine, decode, flipped_probabilities
from pipeline.config import ROOT, Config
from pipeline.data import load_tracking
from pipeline.labels import label_match
from pipeline.runs import value_runs
from pipeline.tcn import Predictor, deterministic

sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402
from starter.baseline import write_submission  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
DEFAULT_MODEL = os.path.join(ROOT, "models", "tcn11.pt")
DEFAULT_DEAD_MODEL = "none"
DECODE_SWITCH, DECODE_MIN_S = 4.0, 2.0


def label_with_tcn(tracking, halves, predictor, cfg=Config(), decoding="viterbi",
                   dead_predictor=None):
    """(TCN labels, rule labels), each one letter per frame, X outside the halves.

    decoding "viterbi" (flips + Viterbi, the default) or "tidy" (argmax + spell tidying).
    dead_predictor: if given, dead ball comes from it and the team from predictor (viterbi only).
    """
    cfg = cfg.replace(PAKAI_SMART_TAGGING=False)
    rules = label_match(tracking, [], halves, cfg).labels
    table = features.compute(tracking, halves, cfg, rule_labels=rules)
    if decoding == "tidy":
        return predictor.labels(table, halves, cfg, tidy=True), rules
    probabilities = flipped_probabilities(predictor, table, halves)
    if dead_predictor is not None:
        probabilities = combine(probabilities, flipped_probabilities(dead_predictor, table, halves))
    return decode(probabilities, halves, DECODE_SWITCH, DECODE_MIN_S, cfg.FPS), rules


def summarise(labels, rules, halves, fps):
    """Per half: label counts, possessions and stoppages, plus agreement with the rules."""
    for half, first, last in halves:
        window = labels[first:last + 1]
        spells = [b - a for a, b, v in value_runs(window) if v in ("H", "A")]
        print("half {}: H {:>6}  A {:>6}  D {:>6}   possessions {:>4} (mean {:.1f} s)"
              "   stoppages {:>4}".format(half, int(np.sum(window == "H")),
                                          int(np.sum(window == "A")),
                                          int(np.sum(window == "D")), len(spells),
                                          np.mean(spells) / fps if spells else 0.0,
                                          sum(1 for _, _, v in value_runs(window) if v == "D")))
    graded = labels != "X"
    print("frames where TCN and rules agree: {:.1%}".format(np.mean(labels[graded] == rules[graded])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="trained model (.pt)")
    parser.add_argument("--dead-model", default=DEFAULT_DEAD_MODEL,
                        help="model whose dead ball is used (.pt), or 'none' for --model alone")
    parser.add_argument("--name", default="dzaky_tcn11", help="submission file name, no .csv")
    parser.add_argument("--out", default=RESULTS_DIR, help="where to write")
    parser.add_argument("--no-cache", action="store_true", help="re-parse the raw ATD")
    parser.add_argument("--decode", choices=("viterbi", "tidy"), default="viterbi",
                        help="flips + Viterbi (default) or argmax + spell tidying")
    args = parser.parse_args()

    deterministic()
    cfg = Config()
    predictor = Predictor.load(args.model)
    dead = None if args.dead_model.lower() == "none" else Predictor.load(args.dead_model)
    if dead is not None and args.decode == "tidy":
        parser.error("--decode tidy works on one model: add --dead-model none")
    tracking = load_tracking(cfg=cfg, cache=not args.no_cache)
    halves = half_video_blocks(GAME)

    labels, rules = label_with_tcn(tracking, halves, predictor, cfg, args.decode, dead)
    path = write_submission(os.path.join(args.out, args.name + ".csv"), labels)
    summarise(labels, rules, halves, cfg.FPS)
    print("\nwrote {}".format(os.path.relpath(path, ROOT)))
    print("check it:  python evaluation/validate_submission.py {}".format(
        os.path.relpath(path, ROOT)))
