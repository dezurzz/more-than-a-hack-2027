"""Leave-one-match-out for the learned models on damaged IDSSE.

    python scripts/external_learn.py --model hgb
    python scripts/external_learn.py --model tcn --steps 2500
    python scripts/external_learn.py --model hgb --folds J03WMX J03WOH      # a subset of folds
    python scripts/external_learn.py --model hgb --indep                    # damage without the dead-ball link

Each fold trains on six matches (clean + public + match x2 + severe copies, team-swap augmentation,
targets = mapped labels) and predicts the held-out match as damaged in the `match` variant with
a seed no training copy used. The decoder (switch and dead-ball costs of the 5 Hz Viterbi) is chosen
by mean held-out spellF1 over the folds, macroF1 breaking ties, and the table shows all folds with the
`aji` rules on the very same damaged copies for reference. Aggregates only. Predictions are cached in
cache/external/ so a decoder can be re-chosen without retraining.
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from mtah import config  # noqa: E402
from mtah.external import dataset, features, files, learn, models, scoring, skillcorner_files as files_sc  # noqa: E402
import external_eval  # noqa: E402

TRAIN_COPIES = (("clean", 0), ("public", 1), ("match", 1), ("match", 2), ("severe", 1))
EVAL_SEED = 100
SWITCH = (1.0, 2.0, 4.0, 8.0)
DEAD = (1.0, 2.0, 4.0, 8.0)


def copies(indep, jitter=False, wira=False, views=None):
    """Training copies; jitter: the "_jit" variants (ball noise), the held-out copy is not one.
    views="clean": only the clean copy, repeated once per default copy so the match keeps its weight."""
    suffix = "_wp" if wira else ("_jit" if jitter else "")
    if views == "clean":
        return [("clean" + suffix, 0)] * len(TRAIN_COPIES)
    if views is not None:
        raise ValueError("unknown views {!r}".format(views))
    return [(("clean" if v == "clean" else (v + "_indep" if indep else v)) + suffix, s) for v, s in TRAIN_COPIES]


def training_data(match_ids, indep, mapping, stride, extra=None, pff_ids=(), flat=True, jitter=False, wira=False, sc_ids=(), with_meta=False, with_stats=False, transform=None, metrica_ids=(), views=None):
    """views: None (all damage views) or "clean" (see copies).
    flat=False skips the strided (x, y) copy that only the HGB needs (returns None, None, seqs).
    pff_ids: PFF games added, one copy each.
    metrica_ids: Metrica games (metrica.GAMES) added with the same copies as IDSSE; domain "metrica", group
    "metrica:<game>:<variant>:<seed>" (training only, never a test domain).
    with_meta=True returns a 4th item: meta[i] = (domain, group) of seqs[i], domain in sc / pff / idsse,
    group "sc:<id>", "pff:<id>" or "idsse:<match>:<variant>:<seed>".
    transform: callable applied to each match's whole fs.x before slicing and before the statistics (R8).
    with_stats=True (needs with_meta) adds a 5th item {group: (mean, std)} of the continuous columns of each
    match's whole (transformed) fs.x, labelled or not (R6)."""
    if with_stats and not with_meta:
        raise ValueError("with_stats needs with_meta")
    xs, ys, seqs, meta, stats = [], [], [], [], {}

    def prepare(fs, group):
        if transform is not None:
            fs.x = transform(fs.x)
        if with_stats:
            stats[group] = learn._continuous_stats(fs.x, learn.column_groups(fs.x.shape[1])["continuous"])
    for sc_id in sc_ids:                                       # SkillCorner matches: real damage, group labels, one copy
        from mtah.external import skillcorner as sc
        fs, _, _ = dataset.sc_feature_set(sc_id, "sc", extra=extra)
        prepare(fs, "sc:{}".format(sc_id))
        y = dataset.row_labels(sc.load(sc_id, "sc", "group"), fs, mapping)
        for a, b in fs.runs:                                   # the network takes no X rows: keep the labelled stretches
            ok = np.flatnonzero(np.diff(np.concatenate([[0], (y[a:b] < 3).astype(np.int8), [0]])))
            for lo, hi in zip(ok[::2], ok[1::2]):
                if hi - lo < 50:
                    continue
                seqs.append((fs.x[a + lo:a + hi], y[a + lo:a + hi]))
                meta.append(("sc", "sc:{}".format(sc_id)))
                if flat:
                    xs.append(fs.x[a + lo:a + hi:stride])
                    ys.append(y[a + lo:a + hi:stride])
        del fs
    for game_id in pff_ids:
        fs, match = dataset.pff_feature_set(game_id, extra=extra)
        prepare(fs, "pff:{}".format(game_id))
        y = dataset.row_labels(match, fs, mapping)
        for a, b in fs.runs:
            labelled = np.flatnonzero(y[a:b] < 3)             # PFF rows before the kickoff / after END are X
            if not len(labelled):
                continue
            a, b = a + int(labelled[0]), a + int(labelled[-1]) + 1
            seqs.append((fs.x[a:b], y[a:b]))
            meta.append(("pff", "pff:{}".format(game_id)))
            if flat:
                xs.append(fs.x[a:b:stride])
                ys.append(y[a:b:stride])
        del fs, match
    for domain, ids in (("idsse", match_ids), ("metrica", metrica_ids)):
        for match_id in ids:
            for variant, seed in copies(indep, jitter, wira, views):
                group = "{}:{}:{}:{}".format(domain, match_id, variant, seed)
                fs, match = dataset.feature_set(match_id, variant, seed, extra=extra)
                prepare(fs, group)
                y = dataset.row_labels(match, fs, mapping)
                for a, b in fs.runs:
                    if domain == "metrica":      # ungraded rows (X) before the first / after the last event of a half
                        keep = np.nonzero(y[a:b] != 3)[0]
                        if not len(keep):
                            continue
                        a, b = a + int(keep[0]), a + int(keep[-1]) + 1
                        assert not (y[a:b] == 3).any(), "{}: ungraded rows inside a half".format(group)
                    seqs.append((fs.x[a:b], y[a:b]))
                    meta.append((domain, group))
                    if flat:
                        xs.append(fs.x[a:b:stride])
                        ys.append(y[a:b:stride])
                del fs, match
    out = (None, None, seqs) if not flat else (np.concatenate(xs), np.concatenate(ys), seqs)
    if with_stats:
        return out + (meta, stats)
    return out + (meta,) if with_meta else out


def subset_ids(train_ids, held_out, match_ids, size, draw):
    """P9 (overnight 8 Nov): `size` of the training matches, drawn with numpy default_rng([draw, fold index])."""
    if not size:
        return train_ids
    rng = np.random.default_rng([int(draw), list(match_ids).index(held_out)])
    pick = sorted(rng.choice(len(train_ids), size=size, replace=False))
    return [train_ids[i] for i in pick]


_PFF_ONLY_MODELS = {}      # (kind, seed) -> fitted model: the training set does not depend on the fold


def pff_games(n):
    from mtah.external import pff
    return pff.game_ids()[:n]


def fit_model(kind, train_ids, pff_ids, args, seed):
    x, y, seqs = training_data(train_ids, args.indep, args.mapping, args.stride, getattr(args, "extra", None),
                               pff_ids=pff_ids, flat=(kind == "hgb"), jitter=getattr(args, "jitter", False), wira=getattr(args, "wira_aug", False),
                               sc_ids=list(files_sc.MATCHES) if getattr(args, "add_sc", False) else ())
    if kind == "hgb":
        return learn.HGB(max_iter=args.iterations, drop_rules=args.drop_rules).fit(x, y)
    return learn.TCN(steps=args.steps, seed=seed, verbose=args.verbose, drop_rules=args.drop_rules,
                     attention=getattr(args, "attention", False)).fit(seqs)


def fit_predict(kind, held_out, match_ids, args, seed=None):
    seed = args.seed if seed is None else seed
    extra = getattr(args, "extra", None)
    variant = "match_indep" if args.indep else "match"
    only = getattr(args, "pff_only", None)
    if only:                                  # trained on PFF alone, once per seed, scored on every fold
        key = (kind, seed)
        if key not in _PFF_ONLY_MODELS:
            _PFF_ONLY_MODELS.clear()
            _PFF_ONLY_MODELS[key] = fit_model(kind, [], pff_games(only), args, seed)
        model = _PFF_ONLY_MODELS[key]
    else:
        train_ids = [m for m in match_ids if m != held_out]
        train_ids = subset_ids(train_ids, held_out, match_ids, getattr(args, "train_subset", None),
                               getattr(args, "subset_draw", 0))
        model = fit_model(kind, train_ids, pff_games(args.add_pff) if getattr(args, "add_pff", None) else (),
                          args, seed)
    fs, match = dataset.feature_set(held_out, variant, EVAL_SEED, extra=extra)
    proba = model.predict_proba(fs.x) if kind == "hgb" else model.predict_proba(fs.x, fs)
    return proba, fs, match


def score_setting(held, setting, mapping, gate=False):
    """(mean spellF1, mean macroF1) of one decoder setting over the folds in `held`."""
    switch, dead = setting
    spell, macro = [], []
    for fs, match, proba in held.values():
        rows = learn.decode_rows(proba, fs, switch, dead, dead, gate=gate)
        pred = features.expand(rows, fs, match.n_frames)
        metrics = scoring.score(match.labels(mapping=mapping or None), pred, match.runs)
        spell.append(metrics["spell_f1_mean"])
        macro.append(metrics["macro_f1"])
    return float(np.mean(spell)), float(np.mean(macro))


def detail_setting(held, setting, mapping, gate):
    """Aggregates of one decoder setting over the folds: the numbers of the results table."""
    switch, dead = setting
    rows = []
    for fs, match, proba in held.values():
        pred = features.expand(learn.decode_rows(proba, fs, switch, dead, dead, gate=gate), fs, match.n_frames)
        gt = match.labels(mapping=mapping or None)
        m = scoring.score(gt, pred, match.runs)
        b = scoring.spell_breakdown(gt, pred, match.runs)
        rows.append((m["spell_f1_mean"], m["macro_f1"], m["per_class"]["D"]["recall"],
                     m["per_class"]["D"]["precision"], b["n_pred"], b["n_gt"], b["counts"]["found"],
                     b["short_2s"]["n"], b["short_2s"]["found"]))
    r = np.array(rows, dtype=np.float64)
    return {"spellF1": r[:, 0].mean(), "macroF1": r[:, 1].mean(), "D recall": r[:, 2].mean(),
            "D precision": r[:, 3].mean(), "spells predicted": int(r[:, 4].sum()),
            "spells true": int(r[:, 5].sum()), "true found": r[:, 6].sum() / r[:, 5].sum(),
            "under 2 s n": int(r[:, 7].sum()), "under 2 s found": r[:, 8].sum() / max(1, r[:, 7].sum()),
            "per fold spellF1": r[:, 0]}


def tag_suffix(args):
    """Cache-tag suffix of the optional extras (empty by default, so existing cache names are unchanged)."""
    out = ""
    if args.extra:
        out += "_x" + args.extra.replace("+", "-")
    if args.attention:
        out += "_attn"
    if getattr(args, "add_pff", None):
        out += "_pff{}".format(args.add_pff)
    if getattr(args, "pff_only", None):
        out += "_pffonly{}".format(args.pff_only)
    if getattr(args, "jitter", False):
        out += "_jit"
    if getattr(args, "wira_aug", False):
        out += "_wp"
    if getattr(args, "add_sc", False):
        out += "_sc20"
    return out


def gate_comparison(args):
    """Wira's evidence gate on the cached five-seed probabilities, no retraining."""
    seeds = args.seeds or [0, 1, 2, 3, 4]
    variant = "match_indep" if args.indep else "match"
    caches = {m: [os.path.join(config.CACHE_DIR, "external", "lomo_tcn_{}_{}_{}_{}{}_{}.npy".format(
        "indep" if args.indep else "dep", args.steps, s, args.mapping, tag_suffix(args), m)) for s in seeds]
        for m in args.folds}
    missing = [c for cs in caches.values() for c in cs if not os.path.isfile(c)]
    if missing:
        print("missing cache files, not retraining:\n  " + "\n  ".join(missing))
        sys.exit(1)
    held = {}
    for match_id in args.folds:
        fs, match = dataset.feature_set(match_id, variant, EVAL_SEED, extra=args.extra)
        held[match_id] = (fs, match, np.mean([np.load(c) for c in caches[match_id]], axis=0))
    grid = [(s, d) for s, d in itertools.product(SWITCH, DEAD) if s < 2 * d]
    print("tcn ensemble seeds {}, T={}, {} folds, {} settings".format(seeds, args.mapping, len(held), len(grid)))
    for gate in (False, True):
        results = {g: score_setting(held, g, args.mapping, gate) for g in grid}
        ranked = sorted(results.items(), key=lambda kv: (-round(kv[1][0], 4), -kv[1][1]))
        for label, setting in (("best", ranked[0][0]), ("default", (2.0, 4.0))):
            d = detail_setting(held, setting, args.mapping, gate)
            print("\ngate {} / {}: switch {:g}, dead {:g}".format("ON" if gate else "OFF", label, *setting))
            for key, value in d.items():
                if key == "per fold spellF1":
                    print("  per fold spellF1: " + " ".join("{:.3f}".format(v) for v in value))
                elif isinstance(value, float):
                    print("  {}: {:.3f}".format(key, value))
                else:
                    print("  {}: {}".format(key, value))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("hgb", "tcn"), default="tcn")
    parser.add_argument("--gate", action="store_true",
                        help="compare the decoder with and without Wira's evidence gate on the cached "
                             "tcn probabilities (use --mapping 2.5); never retrains")
    parser.add_argument("--folds", nargs="+", default=list(files.MATCHES))
    parser.add_argument("--mapping", type=float, default=1.5)
    parser.add_argument("--indep", action="store_true")
    parser.add_argument("--stride", type=int, default=2, help="hgb: every n-th training row")
    parser.add_argument("--iterations", type=int, default=200, help="hgb boosting rounds")
    parser.add_argument("--steps", type=int, default=2500, help="tcn training steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="tcn: ensemble, a fold's probabilities are the mean over these seeds")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--drop-rules", action="store_true",
                        help="drop the feature columns that come from the rule model (features.RULE_COLUMNS)")
    parser.add_argument("--train-subset", type=int, default=None,
                        help="P9: train each fold on this many of its training matches (drawn, see --subset-draw)")
    parser.add_argument("--subset-draw", type=int, default=0, help="P9: draw seed of the training subset")
    parser.add_argument("--skip-report", action="store_true",
                        help="only train / cache the fold probabilities; no decoder table, no aji reference")
    parser.add_argument("--extra", default=None,
                        help="extra feature set(s) appended to the 92 columns, e.g. smooth, touch, near5, smooth+touch")
    parser.add_argument("--add-pff", type=int, default=None, metavar="N",
                        help="each fold also trains on the first N PFF games (one copy each); tag suffix _pffN")
    parser.add_argument("--pff-only", type=int, default=None, metavar="N",
                        help="train on the first N PFF games only (once per seed), predict every fold; suffix _pffonlyN")
    parser.add_argument("--jitter", action="store_true",
                        help="training copies use the '_jit' variants (Gaussian ball noise, sigma 0.1-0.5 m per 20 s); "
                             "the held-out copy is unchanged; tag suffix _jit")
    parser.add_argument("--wira-aug", action="store_true",
                        help="training copies use the '_wp' variants (Wira's ball perturbations at 25 Hz, "
                             "dataset.add_wira_perturbation); the held-out copy is unchanged; tag suffix _wp")
    parser.add_argument("--add-sc", action="store_true",
                        help="each fold also trains on all 20 SkillCorner matches (real damage, group labels); suffix _sc20")
    parser.add_argument("--attention", action="store_true", help="tcn: one residual self-attention layer")
    args = parser.parse_args()
    if args.add_pff and args.pff_only:
        parser.error("--add-pff and --pff-only are exclusive")
    if args.drop_rules:
        print("dropping rule columns: {}".format(", ".join(features.RULE_COLUMNS)))
    if args.gate:
        return gate_comparison(args)
    all_ids = list(files.MATCHES)
    variant = "match_indep" if args.indep else "match"

    held = {}
    seeds = args.seeds if (args.seeds and args.model == "tcn") else [args.seed]
    for match_id in args.folds:
        started = time.time()
        fs, match = dataset.feature_set(match_id, variant, EVAL_SEED, extra=args.extra)
        probas = []
        for seed in seeds:
            tag = "{}_{}_{}_{}_{}".format(args.model, "indep" if args.indep else "dep",
                                          args.iterations if args.model == "hgb" else args.steps, seed, args.mapping)
            if args.drop_rules:
                tag += "_norules"
            if args.train_subset:
                tag += "_sub{}d{}".format(args.train_subset, args.subset_draw)
            tag += tag_suffix(args)
            name = "lomo_{}_{}.npy".format(tag, match_id)
            cache = os.path.join(config.CACHE_DIR, "external", name)
            if not os.path.isfile(cache) and os.environ.get("MTAH_LOMO_DIR"):
                cache = os.path.join(os.environ["MTAH_LOMO_DIR"], name)     # overnight 8 Nov: off-disk outputs
            if os.path.isfile(cache):
                probas.append(np.load(cache))
            else:
                proba, fs, match = fit_predict(args.model, match_id, all_ids, args, seed)
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                partial = cache + ".partial"
                with open(partial, "wb") as handle:          # atomic: a shutdown never leaves a half file
                    np.save(handle, proba)
                os.replace(partial, cache)
                probas.append(proba)
        proba = np.mean(probas, axis=0)
        held[match_id] = (fs, match, proba)
        print("fold {} done in {:.0f} s".format(match_id, time.time() - started), flush=True)
    if args.skip_report:
        return

    grid = [(s, d) for s, d in itertools.product(SWITCH, DEAD) if s < 2 * d]
    results = {g: score_setting(held, g, args.mapping) for g in grid}
    ranked = sorted(results.items(), key=lambda kv: (-round(kv[1][0], 4), -kv[1][1]))
    print("\nmodel {} (mapped labels T={}, damage {}, eval seed {}): decoder settings by mean spellF1".format(
        args.model, args.mapping, "without dead-ball link" if args.indep else "default", EVAL_SEED))
    for (switch, dead), (spell, macro) in ranked[:6]:
        print("  switch {:g} dead {:g}: spellF1 {:.3f} macroF1 {:.3f}".format(switch, dead, spell, macro))
    best = ranked[0][0]
    print("chosen: switch {:g}, dead costs {:g}".format(*best))

    preds = {}
    for match_id, (fs, match, proba) in held.items():
        rows = learn.decode_rows(proba, fs, best[0], best[1], best[1])
        preds[match_id] = features.expand(rows, fs, match.n_frames)

    def learned_model(match_id, tracking, runs):
        return preds[match_id]

    original = external_eval.tracking_for
    external_eval.tracking_for = lambda match, v, seed, profile: dataset.tracking_for(match, variant, seed, profile)
    try:
        learned = external_eval.evaluate_model(learned_model, list(held), ["match"], EVAL_SEED,
                                               args.mapping or None, log=lambda *a: None)
        reference = external_eval.evaluate_model(models.aji, list(held), ["match"], EVAL_SEED,
                                                 args.mapping or None, log=lambda *a: None)
    finally:
        external_eval.tracking_for = original
    print("\n-- {} (decoder chosen on these same folds) --".format(args.model))
    external_eval.summarise({args.model: learned["match"]})
    print("\n-- aji rules on the same damaged copies --")
    external_eval.summarise({"aji": reference["match"]})


if __name__ == "__main__":
    main()
