"""Our models on SkillCorner's real broadcast tracking, no training. Aggregates only.

    python scripts/skillcorner_eval.py                       # sc, three labellings, then sc_detected on the primary one
    python scripts/skillcorner_eval.py --variants sc --labels phase
    python scripts/skillcorner_eval.py --primary group_t25

Models (all trained on IDSSE only, so the 20 A-League matches are unseen):
    aji   our rules (slot teams of the tracking, Smart Tagging off)
    t20   the five TCNs trained on T = 2.0 s labels, mean probabilities, decoder switch 2, dead 2
    t25   the five TCNs trained on T = 2.5 s labels, mean probabilities, decoder switch 2, dead 4
Weights: cache/external/final_tcn_<T>_s<seed>.pt.

Per match and mean: spellF1, macroF1, team accuracy, alive/dead F1, D recall / precision, spells
predicted / true, and what happened to the true spells. Then 95% bootstrap over matches (2000
resamples, seed 0) of each model's mean spellF1 and of the paired differences. Rule fixed in the plan:
the candidate stays ext_t25 unless, on `sc` with the primary labelling, the interval of t20 - t25 lies
entirely above 0.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah import config  # noqa: E402
from mtah.external import dataset, features, learn, scoring, skillcorner as sc, skillcorner_files as files  # noqa: E402
import possession_metrics as pm  # noqa: E402

LABELLINGS = {"phase": ("phase", None), "group": ("group", None), "group_t25": ("group", 2.5)}
SEEDS = (0, 1, 2, 3, 4)
SETTINGS = {"t20": ("2.0", 2.0, 2.0), "t25": ("2.5", 2.0, 4.0)}      # weights tag, switch, dead cost
BOOTSTRAP = 2000


def weights_dir():
    return os.path.join(config.CACHE_DIR, "external")


def load_networks(tag):
    return [learn.TCN.from_state(os.path.join(weights_dir(), "final_tcn_{}_s{}.pt".format(tag, s))) for s in SEEDS]


def predictions(match_id, variant, networks):
    """{model: codes (n,)} on one SkillCorner match (cached)."""
    cache = os.path.join(weights_dir(), "sc_pred_v1_{}_{}.npz".format(match_id, variant))
    match = sc.load(match_id, variant, "phase")
    if os.path.isfile(cache):
        with np.load(cache) as cached:
            return {k: cached[k] for k in cached.files}, match
    fs, match, codes = dataset.sc_feature_set(match_id, variant)
    out = {"aji": codes}
    for name, (tag, switch, dead) in SETTINGS.items():
        proba = np.mean([net.predict_proba(fs.x, fs) for net in networks[tag]], axis=0)
        rows = learn.decode_rows(proba, fs, switch, dead, dead)
        out[name] = features.expand(rows, fs, match.n_frames)
    np.savez(cache, **out)
    return out, match


EXTRA_SWITCH, EXTRA_DEAD = 2.0, 4.0           # the extra model is decoded like t25


def load_prefix_networks(prefix):
    return [learn.TCN.from_state("{}_s{}.pt".format(prefix, s)) for s in SEEDS]


def extra_predictions(match_id, variant, name, prefix, extra):
    """codes (n,) of the extra-feature model (mean probabilities over seeds 0-4), cached under its own name."""
    cache = os.path.join(dataset.JITTER_DIR, "sc_pred_x1_{}_{}_{}.npz".format(name, match_id, variant))
    if os.path.isfile(cache):
        with np.load(cache) as cached:
            return cached["codes"]
    extra = None if extra in ("none", "") else extra          # "none": the 92 base columns (e.g. --wira-aug models)
    fs, match, _ = dataset.sc_feature_set(match_id, variant, extra=extra)
    proba = np.mean([net.predict_proba(fs.x, fs) for net in load_prefix_networks(prefix)], axis=0)
    codes = features.expand(learn.decode_rows(proba, fs, EXTRA_SWITCH, EXTRA_DEAD, EXTRA_DEAD), fs, match.n_frames)
    os.makedirs(dataset.JITTER_DIR, exist_ok=True)
    partial = cache + ".partial"
    with open(partial, "wb") as handle:
        np.savez(handle, codes=codes)
    os.replace(partial, cache)
    return codes


def evaluate(variants_labels, matches, extra_model=None):
    networks = {tag: load_networks(tag) for tag, _, _ in SETTINGS.values()}
    results = {}                                   # (variant, label) -> {model: {match: result}}
    for variant, labels in variants_labels:
        for match_id in matches:
            preds, match = predictions(match_id, variant, networks)
            if extra_model:
                preds = dict(preds, **{extra_model[0]: extra_predictions(match_id, variant, *extra_model)})
            for label in labels:
                source, mapping = LABELLINGS[label]
                truth = sc.load(match_id, variant, source).labels(mapping=mapping)
                for model, pred in preds.items():
                    m = scoring.score(truth, pred, match.runs)
                    results.setdefault((variant, label), {}).setdefault(model, {})[match_id] = {
                        "spell_f1": m["spell_f1_mean"], "macro_f1": m["macro_f1"], "team_acc": m["team_acc_gt_alive"],
                        "alive_dead_f1": m["alive_dead_f1"], "n_gt": m["n_gt_spells"], "n_pred": m["n_pred_spells"],
                        "d_recall": m["per_class"]["D"]["recall"], "d_precision": m["per_class"]["D"]["precision"],
                        "d_share": float((pred == pm.D).mean()), "d_share_gt": float((truth == pm.D).mean()),
                        "breakdown": scoring.spell_breakdown(truth, pred, match.runs)}
            print("  predicted {} {}".format(variant, match_id), flush=True)
    return results


def bootstrap(values_by_match, rng):
    """95% interval of a mean over matches."""
    values = np.asarray(values_by_match)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(BOOTSTRAP)]
    return np.percentile(means, [2.5, 97.5])


def report(variant, label, per_model, extra_name=None):
    print("\n==== variant {} | labels {} ====".format(variant, label))
    ids = list(next(iter(per_model.values())))
    for model, per_match in per_model.items():
        print("\n-- {} --".format(model))
        print("{:<9s} {:>7s} {:>7s} {:>7s} {:>7s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s}".format(
            "match", "spellF1", "macroF1", "team", "a/d F1", "n_pred", "n_true", "D rec", "D prec", "D pred", "D true"))
        for match_id in ids:
            r = per_match[match_id]
            print("{:<9s} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:6d} {:6d} {:6.2f} {:6.2f} {:6.1%} {:6.1%}".format(
                match_id, r["spell_f1"], r["macro_f1"], r["team_acc"], r["alive_dead_f1"], r["n_pred"], r["n_gt"],
                r["d_recall"], r["d_precision"], r["d_share"], r["d_share_gt"]))
        mean = lambda key: np.mean([per_match[m][key] for m in ids])      # noqa: E731
        print("{:<9s} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:6.0f} {:6.0f} {:6.2f} {:6.2f} {:6.1%} {:6.1%}".format(
            "mean", mean("spell_f1"), mean("macro_f1"), mean("team_acc"), mean("alive_dead_f1"), mean("n_pred"),
            mean("n_gt"), mean("d_recall"), mean("d_precision"), mean("d_share"), mean("d_share_gt")))
        total = {k: sum(per_match[m]["breakdown"]["counts"][k] for m in ids) for k in scoring.KINDS}
        n_gt = sum(per_match[m]["breakdown"]["n_gt"] for m in ids)
        print("true spells ({}): ".format(n_gt) + ", ".join("{} {:.0%}".format(k, v / n_gt) for k, v in total.items()))
        for seconds in (2, 4):
            n = sum(per_match[m]["breakdown"]["short_%ds" % seconds]["n"] for m in ids)
            found = sum(per_match[m]["breakdown"]["short_%ds" % seconds]["found"] for m in ids)
            print("true spells under {} s: {}, found {:.0%}".format(seconds, n, found / max(1, n)))
    rng = np.random.default_rng(0)
    spell = {model: np.array([per_model[model][m]["spell_f1"] for m in ids]) for model in per_model}
    print("\nbootstrap over {} matches (95%, {} resamples):".format(len(ids), BOOTSTRAP))
    for model, v in spell.items():
        lo, hi = bootstrap(v, np.random.default_rng(0))
        print("  {:<6s} spellF1 mean {:.3f} [{:.3f}, {:.3f}]".format(model, v.mean(), lo, hi))
    out = {}
    pairs = [("t25", "aji"), ("t20", "aji"), ("t25", "t20"), ("t20", "t25")]
    if extra_name:
        pairs.append((extra_name, "t25"))
    for a, b in pairs:
        d = spell[a] - spell[b]
        lo, hi = bootstrap(d, np.random.default_rng(0))
        out[(a, b)] = (d.mean(), lo, hi)
        print("  {} - {}: {:+.3f} [{:+.3f}, {:+.3f}]  better in {}/{} matches".format(a, b, d.mean(), lo, hi, int((d > 0).sum()), len(d)))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", nargs="+", default=["sc", "sc_detected"], choices=sc.VARIANTS)
    parser.add_argument("--labels", nargs="+", default=None, choices=list(LABELLINGS),
                        help="labellings for sc (default all three); sc_detected always uses --primary only")
    parser.add_argument("--primary", default="phase", choices=list(LABELLINGS))
    parser.add_argument("--extra-model", nargs=3, default=None, metavar=("NAME", "WEIGHTS_PREFIX", "EXTRA"),
                        help="add a model trained with extra features (weights '<prefix>_s<seed>.pt', seeds 0-4), "
                             "decoded at switch 2 dead 4, and print NAME - t25")
    parser.add_argument("--matches", nargs="+", default=list(files.MATCHES))
    args = parser.parse_args()
    plan = []
    for variant in args.variants:
        labels = (args.labels or list(LABELLINGS)) if variant == "sc" else [args.primary]
        plan.append((variant, labels))
    extra_model = tuple(args.extra_model) if args.extra_model else None
    results = evaluate(plan, args.matches, extra_model)
    verdict = None
    for (variant, label), per_model in results.items():
        out = report(variant, label, per_model, extra_model[0] if extra_model else None)
        if variant == "sc" and label == args.primary:
            verdict = out[("t20", "t25")]
    if extra_model and ("sc", args.primary) in results:
        name = extra_model[0]
        d = np.array([results[("sc", args.primary)][name][m]["spell_f1"] - results[("sc", args.primary)]["t25"][m]["spell_f1"]
                      for m in args.matches])
        lo, hi = bootstrap(d, np.random.default_rng(0))
        print()
        print("G2 reading (sc, {}): {} - t25 = {:+.3f} [{:+.3f}, {:+.3f}], better in {}/{} -> {}".format(
            args.primary, name, d.mean(), lo, hi, int((d > 0).sum()), len(d),
            "PASS (mean > 0 and lower end > -0.02)" if d.mean() > 0 and lo > -0.02 else "FAIL"))
    if verdict:
        mean, lo, hi = verdict
        print("\nRULE (sc, {}): t20 - t25 = {:+.3f} [{:+.3f}, {:+.3f}] -> {}".format(
            args.primary, mean, lo, hi,
            "SWITCH the candidate to t20 (interval above 0)" if lo > 0 else "candidate stays ext_t25"))


if __name__ == "__main__":
    main()
