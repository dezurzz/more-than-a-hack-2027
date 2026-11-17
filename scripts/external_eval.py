"""One evaluation for every model on the damaged IDSSE matches.

    python scripts/external_eval.py --model aji
    python scripts/external_eval.py --model smooth50 --variants clean match severe
    python scripts/external_eval.py --model folder:path/to/csvs --variants match   # <match_id>.csv per match
    python scripts/external_eval.py --model aji --matches J03WMX J03WOH --seed 1

For every match and variant (clean = no damage; public / match / severe = the ATD-like damage of
mtah/external/damage.py) the model predicts from the tracking alone, then the prediction is scored
with the official possession_metrics functions against the mapped labels (contest resolution
T = 1.5 s: the main score) and against the raw DFL flags (reference). Prints, per variant: mean over
the 7 matches and over the 2 Bundesliga matches of spellF1, macroF1, team accuracy and the alive /
dead F1, the number of predicted and true spells, and the breakdown of what happened to the true
spells (found / boundary / split / merged / dead / wrong team) with the recall on short ones.
Aggregates only. Nothing here is a BAR-ATH label or a BAR-ATH score.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.external import convert, damage, files, idsse, labels as label_map, models, scoring  # noqa: E402
import possession_metrics as pm  # noqa: E402

BUNDESLIGA = ("J03WMX", "J03WN1")
VARIANTS = ("clean",) + tuple(damage.VARIANTS)


def tracking_for(match, variant, seed, profile):
    if variant == "clean":
        return convert.to_tracking(match)
    return damage.damage(match, variant, seed, profile=profile).tracking


def evaluate_model(model, matches, variants, seed, mapping, log=print):
    """{variant: {match_id: result}}; result has the metrics for mapped and raw labels."""
    profile = damage.load_profile()
    out = {}
    for variant in variants:
        out[variant] = {}
        for match_id in matches:
            started = time.time()
            match = idsse.load(match_id)
            pred = np.asarray(model(match_id, tracking_for(match, variant, seed, profile), match.runs))
            truth = {"mapped": match.labels(mapping=mapping), "raw": match.labels(mapping=None)}
            result = {}
            for name, gt in truth.items():
                metrics = scoring.score(gt, pred, match.runs)
                result[name] = {
                    "spell_f1": metrics["spell_f1_mean"], "macro_f1": metrics["macro_f1"],
                    "team_acc": metrics["team_acc_gt_alive"], "alive_dead_f1": metrics["alive_dead_f1"],
                    "n_gt": metrics["n_gt_spells"], "n_pred": metrics["n_pred_spells"],
                    "d_recall": metrics["per_class"]["D"]["recall"],
                    "d_precision": metrics["per_class"]["D"]["precision"],
                    "d_share": float((pred == pm.D).mean()), "d_share_gt": float((gt == pm.D).mean()),
                }
                if name == "mapped":
                    result["breakdown"] = scoring.spell_breakdown(gt, pred, match.runs)
            out[variant][match_id] = result
            log("  {} {:7s} {:6s} spellF1 {:.3f} (raw {:.3f}) macroF1 {:.3f}  [{:.0f} s]".format(
                variant, match_id, "", result["mapped"]["spell_f1"], result["raw"]["spell_f1"],
                result["mapped"]["macro_f1"], time.time() - started))
    return out


def summarise(results):
    """Print the tables for evaluate_model's output."""
    for variant, per_match in results.items():
        print("\n== {} ==".format(variant))
        print("{:<10s} {:>7s} {:>7s} {:>7s} {:>7s} {:>8s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s}".format(
            "match", "spellF1", "macroF1", "team", "a/d F1", "raw spF1", "n_pred", "n_true", "D rec", "D prec",
            "D pred", "D true"))
        for match_id, r in per_match.items():
            m = r["mapped"]
            print("{:<10s} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:8.3f} {:6d} {:6d} {:6.2f} {:6.2f} {:6.1%} {:6.1%}".format(
                match_id, m["spell_f1"], m["macro_f1"], m["team_acc"], m["alive_dead_f1"],
                r["raw"]["spell_f1"], m["n_pred"], m["n_gt"], m["d_recall"], m["d_precision"],
                m["d_share"], m["d_share_gt"]))
        for name, subset in (("mean of {}".format(len(per_match)), list(per_match)),
                             ("Bundesliga pair", [m for m in per_match if m in BUNDESLIGA]),
                             ("2. Bundesliga rest", [m for m in per_match if m not in BUNDESLIGA])):
            if not subset:
                continue
            rows = [per_match[m] for m in subset]
            mean = lambda key, which="mapped": np.mean([r[which][key] for r in rows])      # noqa: E731
            print("{:<10s} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:8.3f} {:6.0f} {:6.0f} {:6.2f} {:6.2f} {:6.1%} {:6.1%}".format(
                name[:10], mean("spell_f1"), mean("macro_f1"), mean("team_acc"), mean("alive_dead_f1"),
                mean("spell_f1", "raw"), mean("n_pred"), mean("n_gt"), mean("d_recall"), mean("d_precision"),
                mean("d_share"), mean("d_share_gt")))
        total = {kind: sum(r["breakdown"]["counts"][kind] for r in per_match.values()) for kind in scoring.KINDS}
        n_gt = sum(r["breakdown"]["n_gt"] for r in per_match.values())
        print("true spells ({}): ".format(n_gt) + ", ".join("{} {:.0%}".format(k, v / n_gt) for k, v in total.items()))
        for seconds in (2, 4):
            n = sum(r["breakdown"]["short_%ds" % seconds]["n"] for r in per_match.values())
            found = sum(r["breakdown"]["short_%ds" % seconds]["found"] for r in per_match.values())
            print("true spells under {} s: {} , found {:.0%}".format(seconds, n, found / max(1, n)))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="aji, smooth50 or folder:<path>")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--matches", nargs="+", default=list(files.MATCHES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mapping", type=float, default=label_map.MAPPING_SECONDS,
                        help="contest-resolution seconds for the main labels (0 = raw)")
    args = parser.parse_args()
    results = evaluate_model(models.get(args.model), args.matches, args.variants, args.seed, args.mapping or None)
    print("\nmodel: {}  seed {}  main labels: {}".format(
        args.model, args.seed, "mapped T={} s".format(args.mapping) if args.mapping else "raw"))
    summarise(results)


if __name__ == "__main__":
    main()
