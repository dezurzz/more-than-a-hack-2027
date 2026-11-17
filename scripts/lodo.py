"""Leave-one-domain-out (LODO) yardstick.

    python scripts/lodo.py --recipe R0 [--runs L-SC L-PFF L-IDSSE] [--seeds 0 1 2 3 4]
    python scripts/lodo.py --recipe R1 --score-only [--reference R0]
    python scripts/lodo.py --recipe R0 --decoder-table
    python scripts/lodo.py --ensemble R0 R4 --score-only --reference R0 --reference2 R4

Runs (train / test, the test domain is never seen in training):
    L-SC     IDSSE 7 + PFF games[:32]       -> SkillCorner 20 (group_t25 labels)
    L-PFF    IDSSE 7 + SkillCorner 20       -> PFF games[32:64]
    L-IDSSE  SkillCorner 20 + PFF games[:32] -> IDSSE 7, copy `match`, seed 100
Ablation runs (same test domain, one source dropped; select with --runs, never in the default):
    L-PFF-noSC IDSSE -> PFF[32:64]; L-IDSSE-noSC PFF32 -> IDSSE; L-SC-noPFF IDSSE -> SC20; L-IDSSE-noPFF SC20 -> IDSSE
Recipe parts NW (Wira off), CV (clean IDSSE copy x5), R8s32 (32 channels): e.g. R8+NW, R8+CV.
Recipes are learn.TCN switches: R0 {}, R1 per_match, R2 balance_domains, R3 input_aug, R4 domain_heads,
R5 channels 64 / steps 4000, R6 R4 + whole-match per-match statistics, R7 R4 + no rule columns, R8 R4 + rank of
rule_pdead, R9 R4 + test-time self-training, DT dedicated dead TCN on 92 + deadcue columns
(`--extra SET` adds any extra feature set to any recipe, tag suffix _x<set>); combine with '+', e.g. R1+R3, R6+R9. Probabilities are cached per (recipe, run, seed,
match) in work_data/overnight_cache/lodo/ (resumable, atomic). Decoder 2 / 4 / 4 unless --decoder-table,
labels T = 2.5 s, scoring.score as sc_train_cv.py. Aggregates only.
"""
import argparse
import gc
import itertools
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

CACHE_DIR = os.path.join(os.path.dirname(ROOT), "work_data", "overnight_cache", "lodo")
RUNS = ("L-SC", "L-PFF", "L-IDSSE")
# Ablation runs: the same test domain, one training source dropped (never part of the default --runs)
ABLATION_RUNS = {"L-PFF-noSC": "L-PFF", "L-IDSSE-noSC": "L-IDSSE", "L-SC-noPFF": "L-SC", "L-IDSSE-noPFF": "L-IDSSE"}
# L-PFFH: the fresh PFF hold-out, same training spec as L-PFF (IDSSE + SkillCorner, no PFF game), test games[0:32]. Predictions
# only: it is never scored (needs --no-score, refused by every scoring path), a pre-registered test scores it later.
HOLDOUT_RUNS = {"L-PFFH": "L-PFF"}
ALL_RUNS = RUNS + tuple(ABLATION_RUNS) + tuple(HOLDOUT_RUNS)
DROPPED = {"L-PFF-noSC": "sc", "L-IDSSE-noSC": "sc", "L-SC-noPFF": "pff", "L-IDSSE-noPFF": "pff"}
RECIPES = {
    "R0": {},
    "R1": {"per_match": True},
    "R2": {"balance_domains": True},
    "R3": {"input_aug": True},
    "R4": {"domain_heads": True},
    "R5": {"channels": 64, "steps": 4000},
    "R6": {"domain_heads": True, "per_match": "whole"},
    "R7": {"domain_heads": True, "drop_rules": True},
    "R8": {"domain_heads": True, "rank_pdead": True},
    "R9": {"domain_heads": True, "adapt_steps": 300},
    # Metrica games added to the training list of all three runs (never a test set)
    "M": {"metrica": True},
    "R4M": {"domain_heads": True, "metrica": True},
    "R1M": {"per_match": True, "metrica": True},
    "R8M": {"domain_heads": True, "rank_pdead": True, "metrica": True},
    # Dedicated dead-ball TCN, R4 heads, rank_pdead, dead loss only with weight 3 on short stretches
    # and dead onsets, inputs 92 + the deadcue columns (`extra` is a data switch, not a TCN kwarg)
    # A4 Wira perturbation off, A3 IDSSE clean copy only (repeated), C 32 channels
    "NW": {"wira": False},
    "CV": {"idsse_views": "clean"},
    "R8s32": {"domain_heads": True, "rank_pdead": True, "channels": 32},
    # R8 with a transformer encoder instead of the TCN (width 96, 60 s windows)
    "TF": {"domain_heads": True, "rank_pdead": True, "arch": "transformer", "channels": 96, "crop": 300, "lr": 1e-3},
    "DT": {"domain_heads": True, "rank_pdead": True, "dead_only": True, "short_dead_weight": 3.0, "extra": "deadcue"},
}
_TRUTH_FORBIDDEN = False     # True while train() runs an R9-type recipe: no truth label object may be built
DEFAULT_DECODER = (2.0, 4.0)
SWITCHES = (1.0, 2.0, 3.0, 4.0)
DEADS = (2.0, 4.0, 8.0)
# decode.transition_costs rejects switch >= 2 * dead (switch 4 / dead 2): that cell of the 4 x 3 grid is infeasible
GRID = tuple((s, d) for s, d in itertools.product(SWITCHES, DEADS) if s < 2 * d)


# -- pure helpers (unit-tested) ---------------------------------------------------------------

def base_run(run):
    """Test domain of a run id: L-PFF-noSC -> L-PFF; the three base runs map to themselves."""
    if run in RUNS:
        return run
    if run in ABLATION_RUNS:
        return ABLATION_RUNS[run]
    if run in HOLDOUT_RUNS:
        return HOLDOUT_RUNS[run]
    raise ValueError("unknown run {!r}".format(run))


FUSERS = ("DH", "RB")        # scripts/lodo_deadfuse.py: a tag 'R0+DH' is cached fused probabilities, never trained


def is_fused(tag):
    """'R0+DH' / 'R8+RB': probabilities written by scripts/lodo_deadfuse.py (score-only tags)."""
    parts = tag.split("+")
    return len(parts) > 1 and parts[-1] in FUSERS


def recipe_kwargs(recipe):
    out = {}
    for part in recipe.split("+"):
        if part in FUSERS:
            continue                               # the fuser adds no training switch
        if part not in RECIPES:
            raise ValueError("unknown recipe part {!r} in {!r}".format(part, recipe))
        out.update(RECIPES[part])
    return out


def cache_tag(recipe, steps=None, adapt_steps=None, extra=None):
    """Cache tag; an extra feature set adds _x<set> ('+' written '-'), so default tags are unchanged."""
    return (recipe + ("_st{}".format(steps) if steps else "") + ("_ad{}".format(adapt_steps) if adapt_steps else "")
            + ("_x{}".format(extra.replace("+", "-")) if extra else ""))


def extra_set(recipe, cli_extra=None):
    """The extra feature set of a run: --extra if given, else the one the recipe carries (DT), else None."""
    return cli_extra or recipe_kwargs(recipe).get("extra")


def ensemble_tag(tags):
    """Name of an equal-weight ensemble of cached recipe tags: ['R0', 'R4'] -> 'E_R0+R4'."""
    if len(tags) < 2:
        raise ValueError("an ensemble needs at least two recipes")
    return "E_" + "+".join(tags)


def component_tags(name):
    """Cache tags behind a name: 'E_R0+R4' -> ['R0', 'R4'], any other name -> [name]."""
    return name[2:].split("+") if name.startswith("E_") else [name]


def dead_f1(recall, precision):
    """2PR / (P + R), 0 when both are 0."""
    return 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0


def stratified_bootstrap(diffs_by_domain, n=2000, seed=0):
    """95% interval of the mean of the domain means; matches are resampled inside each domain."""
    rng = np.random.default_rng(seed)
    domains = sorted(diffs_by_domain)
    arrays = [np.asarray(diffs_by_domain[d], dtype=np.float64) for d in domains]
    draws = np.empty(n)
    for i in range(n):
        draws[i] = np.mean([a[rng.integers(0, len(a), len(a))].mean() for a in arrays])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def gate_gr(spell_diff, spell_lo, domain_spell_diffs, macro_diff):
    """Gate GR -> (passed, [one line per condition])."""
    c1 = spell_diff >= 0.010 and spell_lo > 0
    worst = min(domain_spell_diffs.values())
    c2 = worst >= -0.005
    c3 = macro_diff >= -0.005
    reasons = [
        "1 LODO spell diff {:+.4f} >= +0.0100 and lower bound {:+.4f} > 0: {}".format(spell_diff, spell_lo, "PASS" if c1 else "FAIL"),
        "2 worst domain spell diff {:+.4f} >= -0.0050: {}".format(worst, "PASS" if c2 else "FAIL"),
        "3 LODO macro diff {:+.4f} >= -0.0050: {}".format(macro_diff, "PASS" if c3 else "FAIL"),
    ]
    return bool(c1 and c2 and c3), reasons


def gate_g2(spell_diff, spell_lo, domain_spell_diffs, macro_diff):
    """Gate G2 (against R4) -> (passed, [one line per condition])."""
    c1 = spell_diff >= 0.005 and spell_lo > 0
    worst = min(domain_spell_diffs.values())
    c2 = worst >= -0.005
    c3 = macro_diff >= -0.005
    reasons = [
        "1 LODO spell diff {:+.4f} >= +0.0050 and lower bound {:+.4f} > 0: {}".format(spell_diff, spell_lo, "PASS" if c1 else "FAIL"),
        "2 worst domain spell diff {:+.4f} >= -0.0050: {}".format(worst, "PASS" if c2 else "FAIL"),
        "3 LODO macro diff {:+.4f} >= -0.0050: {}".format(macro_diff, "PASS" if c3 else "FAIL"),
    ]
    return bool(c1 and c2 and c3), reasons


def assert_no_validation():
    """The validation answers are never imported by a training run."""
    assert "mtah.validation" not in sys.modules, "mtah.validation was imported during a training run"


def gate_dg(dead_diff, dead_lo, domain_dead_f1_diffs, domain_dead_prec_diffs, found_new, found_base, false_new, false_base,
            spell_diff, domain_spell_diffs, macro_diff, macro_lo, spell_new, ratio_new, ratio_base, r4_spell=0.5659):
    """Gate DG (X+B against its base B) -> (passed, [one line per condition]).

    dead_diff, dead_lo     LODO dead-frame F1 (per-match mean) difference and the lower end of its 95% interval
    domain_dead_f1_diffs   {run: dead F1 difference};  domain_dead_prec_diffs {run: dead precision difference}
    found_new, found_base  pooled found share of the ball-tracked true dead stretches under 15 s (D1), X+B and B
    false_new, false_base  pooled number of false dead stretches (D1, all buckets), X+B and B
    spell_diff, domain_spell_diffs, macro_diff, macro_lo   as gate_gr; spell_new is the LODO spellF1 of X+B
    ratio_new, ratio_base  {run: predicted / true spells per alive minute}, X+B and B."""
    c1 = dead_diff >= 0.015 and dead_lo > 0
    worst_f1 = min(domain_dead_f1_diffs.values())
    worst_prec = min(domain_dead_prec_diffs.values())
    c2 = worst_f1 >= -0.010 and worst_prec >= -0.03
    found_gain = found_new - found_base
    false_rel = (false_new - false_base) / false_base if false_base else (0.0 if false_new == 0 else float("inf"))
    c3 = found_gain >= 0.05 and false_rel <= 0.10
    worst_spell = min(domain_spell_diffs.values())
    ratio_worse = max(abs(ratio_new[r] - 1.0) - abs(ratio_base[r] - 1.0) for r in ratio_new)
    c4 = (spell_diff >= -0.003 and worst_spell >= -0.008 and macro_diff >= 0 and macro_lo > -0.003
          and spell_new >= r4_spell and ratio_worse <= 0.05)
    ok = lambda c: "PASS" if c else "FAIL"      # noqa: E731
    reasons = [
        "DG1 dead F1 diff {:+.4f} >= +0.0150 and lower bound {:+.4f} > 0: {}".format(dead_diff, dead_lo, ok(c1)),
        "DG2 worst domain dead F1 diff {:+.4f} >= -0.0100 and worst dead precision diff {:+.4f} >= -0.0300: {}".format(
            worst_f1, worst_prec, ok(c2)),
        "DG3 tracked short found share {:+.4f} >= +0.0500 and false stretches {:+.1%} <= +10.0%: {}".format(
            found_gain, false_rel, ok(c3)),
        "DG4 spell diff {:+.4f} >= -0.0030, worst domain {:+.4f} >= -0.0080, macro diff {:+.4f} >= 0 (lower {:+.4f} > -0.0030), "
        "spellF1 {:.4f} >= {:.4f}, spells ratio worse by {:+.4f} <= +0.0500: {}".format(
            spell_diff, worst_spell, macro_diff, macro_lo, spell_new, r4_spell, ratio_worse, ok(c4)),
    ]
    return bool(c1 and c2 and c3 and c4), reasons


def assert_no_leak(run, test_ids, idsse_ids, sc_ids, pff_ids, meta=None, metrica_ids=()):
    """Fail hard if any test match id is in the training list of the run (ids and, when given, the group keys).
    Metrica is a training-only domain: its games may never be a test id, and only the games of metrica.GAMES
    may carry the domain "metrica"."""
    from mtah.external import metrica, pff
    test = {str(t) for t in test_ids}
    train = {"idsse": {str(i) for i in idsse_ids}, "sc": {str(i) for i in sc_ids}, "pff": {str(i) for i in pff_ids}}
    source = {"L-SC": "sc", "L-PFF": "pff", "L-IDSSE": "idsse"}[base_run(run)]
    if run in DROPPED:
        assert not train[DROPPED[run]], "{}: the dropped source {} is in the training list".format(run, DROPPED[run])
    assert not (test & train[source]), "{}: test ids in the training list: {}".format(run, sorted(test & train[source]))
    held_only = {str(g) for g in pff.game_ids()[32:64]}
    met = {str(i) for i in metrica_ids}
    assert met <= set(metrica.GAMES), "{}: not Metrica games: {}".format(run, sorted(met - set(metrica.GAMES)))
    assert not (test & set(metrica.GAMES)), "{}: a Metrica game is in the test list: {}".format(run, sorted(test & set(metrica.GAMES)))
    assert not (test & met), "{}: test ids in the Metrica training list: {}".format(run, sorted(test & met))
    assert not (held_only & train["pff"]), "{}: PFF games [32:64] (test-only) are in the PFF training list".format(run)
    if meta is not None:
        for domain, group in meta:
            token = group.split(":")[1]
            if domain == "metrica":
                assert token in met and token not in test, "{}: group {} is not a Metrica training game".format(run, group)
                continue
            assert token not in metrica.GAMES, "{}: group {} is a Metrica game under domain {}".format(run, group, domain)
            assert token not in test or domain != source, "{}: group {} is a test match".format(run, group)
            assert token not in held_only or domain != "pff", "{}: group {} is a test-only PFF game".format(run, group)
            assert domain != source, "{}: a {} sequence ({}) in the training data of the run that tests {}".format(run, domain, group, source)


def domain_means(per_match):
    """{run: {match: (spell, macro, ...)}} -> ({run: mean spell}, {run: mean macro})."""
    spell = {r: float(np.mean([v[0] for v in d.values()])) for r, d in per_match.items()}
    macro = {r: float(np.mean([v[1] for v in d.values()])) for r, d in per_match.items()}
    return spell, macro


def paired(new, ref):
    """Paired per-match differences {run: array} (spell, macro) over the matches both have, sorted by match id."""
    ds, dm = {}, {}
    for run in sorted(new):
        keys = sorted(set(new[run]) & set(ref[run]))
        ds[run] = np.array([new[run][k][0] - ref[run][k][0] for k in keys])
        dm[run] = np.array([new[run][k][1] - ref[run][k][1] for k in keys])
    return ds, dm


def best_setting(per_setting, runs):
    """Setting with the best mean (over `runs`) of the domain mean spellF1, macroF1 breaking ties."""
    def key(s):
        sp = np.mean([np.mean([v[0] for v in per_setting[s][r].values()]) for r in runs])
        ma = np.mean([np.mean([v[1] for v in per_setting[s][r].values()]) for r in runs])
        return (-round(float(sp), 4), -float(ma))
    return min(per_setting, key=key)


def nested_choice(per_setting):
    """{run: chosen setting}, nested per_match dict: the setting is picked on the other two runs."""
    runs = sorted(next(iter(per_setting.values())))
    chosen, nested = {}, {}
    for held in runs:
        chosen[held] = best_setting(per_setting, [r for r in runs if r != held])
        nested[held] = per_setting[chosen[held]][held]
    return chosen, nested


# -- run definitions -----------------------------------------------------------------------------

def run_spec(run):
    """(train kwargs for external_learn.training_data, test ids)."""
    from mtah.external import files, pff, skillcorner_files as sfiles
    idsse = list(files.MATCHES)
    sc_ids = sorted(sfiles.MATCHES)
    games = list(pff.game_ids())
    base = base_run(run)
    if base == "L-SC":
        spec, test = dict(idsse=idsse, sc=[], pff=games[:32]), sc_ids
    elif base == "L-PFF":
        spec, test = dict(idsse=idsse, sc=sc_ids, pff=[]), games[32:64]
    else:
        spec, test = dict(idsse=[], sc=sc_ids, pff=games[:32]), idsse
    if run in DROPPED:
        spec[DROPPED[run]] = []                      # the named source is not in the training list
    if run in HOLDOUT_RUNS:
        test = games[:32]                            # same training spec as L-PFF, the other 32 games as test
    return spec, test


def refuse_holdout_scoring(runs):
    """Hard stop: a hold-out run id is never scored (reserved for one pre-registered test)."""
    bad = [r for r in runs if r in HOLDOUT_RUNS]
    if bad:
        raise SystemExit("{}: hold-out run, never scored here (predictions only: use --no-score, no --score-only / --ensemble / "
                         "--decoder-table)".format(", ".join(bad)))


def load_test(run, match_id):
    """(feature set, match, truth labels) of one test match, patterns of sc_train_cv.py / overnight_compare.py."""
    assert not _TRUTH_FORBIDDEN, "load_test (truth labels) called while an R9-type recipe is training"
    from mtah.external import dataset, skillcorner as sc
    run = base_run(run)
    if run == "L-SC":
        import skillcorner_eval
        fs, match, _ = dataset.sc_feature_set(match_id, "sc")
        source, mapping = skillcorner_eval.LABELLINGS["group_t25"]
        return fs, match, sc.load(match_id, "sc", source).labels(mapping=mapping)
    if run == "L-PFF":
        fs, match = dataset.pff_feature_set(match_id)
        return fs, match, match.labels(mapping=2.5)
    fs, match = dataset.feature_set(match_id, "match", 100)
    return fs, match, match.labels(mapping=2.5)


def load_inputs(run, match_id, extra=None):
    """The feature set of one test match and nothing else: no truth labels are built (R9 leakage guard).
    extra: set name of mtah/external/extra_features appended to the 92 columns (None = the base matrix)."""
    from mtah.external import dataset
    run = base_run(run)
    if run == "L-SC":
        fs, _, _ = dataset.sc_feature_set(match_id, "sc", extra=extra)
    elif run == "L-PFF":
        fs, _ = dataset.pff_feature_set(match_id, extra=extra)
    else:
        fs, _ = dataset.feature_set(match_id, "match", 100, extra=extra)
    return fs


def save_proba(path, proba):
    with open(path + ".partial", "wb") as handle:     # atomic: a shutdown never leaves a half file
        np.save(handle, proba)
    os.replace(path + ".partial", path)


def proba_path(cache_dir, tag, run, seed, match_id):
    return os.path.join(cache_dir, "{}_{}_s{}_{}.npy".format(tag, run, seed, match_id))


# -- training ------------------------------------------------------------------------------------

def train(args, recipe, tag):
    global _TRUTH_FORBIDDEN
    kwargs = recipe_kwargs(recipe)
    kwargs["extra"] = extra_set(recipe, getattr(args, "extra", None))
    adapt_steps = kwargs.pop("adapt_steps", 0)
    if adapt_steps and args.adapt_steps is not None:
        adapt_steps = args.adapt_steps
    _TRUTH_FORBIDDEN = bool(adapt_steps)
    try:
        _train(args, tag, kwargs, adapt_steps)
    finally:
        _TRUTH_FORBIDDEN = False


def _train(args, tag, kwargs, adapt_steps):
    import external_learn
    from mtah.external import learn, metrica
    if args.steps:
        kwargs["steps"] = args.steps
    kwargs.setdefault("steps", 2500)
    use_metrica = bool(kwargs.pop("metrica", False))         # a training-data switch, not a TCN kwarg
    extra_cols = kwargs.pop("extra", None)                   # likewise: the extra feature set of the matrices
    metrica_ids = list(metrica.GAMES) if use_metrica else []
    wira = bool(kwargs.pop("wira", True))                    # NW: Wira perturbation off
    views = kwargs.pop("idsse_views", None)                  # CV: clean IDSSE copy only, repeated (A3)
    whole = kwargs.get("per_match") == "whole"
    ranked = bool(kwargs.get("rank_pdead"))
    for run in args.runs:
        spec, test_ids = run_spec(run)
        todo = [s for s in args.seeds
                if not all(os.path.isfile(proba_path(args.cache_dir, tag, run, s, m)) for m in test_ids)]
        if not todo:
            print("{} {}: all seeds cached".format(tag, run), flush=True)
            continue
        assert_no_leak(run, test_ids, spec["idsse"], spec["sc"], spec["pff"], metrica_ids=metrica_ids)
        started = time.time()
        extra = {}
        if ranked:
            extra["transform"] = learn.rank_pdead
        if whole:
            extra["with_stats"] = True
        out = external_learn.training_data(spec["idsse"], False, 2.5, 2, flat=False, wira=wira, views=views,
                                           sc_ids=spec["sc"], pff_ids=spec["pff"], with_meta=True,
                                           metrica_ids=metrica_ids, extra=extra_cols, **extra)
        _, _, seqs, meta = out[:4]
        stats = out[4] if whole else None
        del out
        assert_no_leak(run, test_ids, spec["idsse"], spec["sc"], spec["pff"], meta, metrica_ids=metrica_ids)
        print("{} {}: {} training sequences in {:.0f} s, leakage check passed".format(
            tag, run, len(seqs), time.time() - started), flush=True)
        domains = [d for d, _ in meta]
        groups = [g for _, g in meta]
        fit_kwargs = {}
        if whole:
            fit_kwargs["group_stats"] = stats
        if ranked:
            fit_kwargs["pre_transformed"] = True
        os.makedirs(args.cache_dir, exist_ok=True)
        for seed in todo:
            started = time.time()
            model = learn.TCN(seed=seed, device=args.device, **kwargs).fit(
                seqs, domains=domains, groups=groups, **fit_kwargs)
            shares = []
            for m in list(test_ids) + sanity_ids(args, run):
                path = proba_path(args.cache_dir, tag, run, seed, m)
                if os.path.isfile(path):
                    continue                                     # resumable per (seed, match)
                fs = load_inputs(run, m, extra_cols)
                if adapt_steps:
                    adapted = model.adapted(fs.x, fs, steps=adapt_steps)
                    shares.append(adapted.last_confident)
                    save_proba(path, adapted.predict_proba(fs.x, fs))
                    del adapted
                else:
                    save_proba(path, model.predict_proba(fs.x, fs))
                del fs
                gc.collect()
            model.release_source()
            del model
            gc.collect()
            if _cuda_available():
                import torch
                torch.cuda.empty_cache()
            note = " | confident rows {:.3f} (mean over {} matches)".format(np.mean(shares), len(shares)) if shares else ""
            print("{} {} seed {} done in {:.0f} s{}".format(tag, run, seed, time.time() - started, note), flush=True)
        del seqs, meta, stats
        assert_no_validation()
        gc.collect()


def sanity_ids(args, run):
    """Extra games predicted by a hold-out run only, to compare with the cached L-PFF probabilities (games 32-63, never the hold-out)."""
    ids = list(getattr(args, "sanity_game", None) or [])
    if ids:
        assert run in HOLDOUT_RUNS, "--sanity-game is only for hold-out run ids"
        from mtah.external import pff
        assert all(i in [str(g) for g in list(pff.game_ids())[32:64]] for i in ids), "--sanity-game must be a game of 32-63"
    return ids


def _cuda_available():
    import torch
    return torch.cuda.is_available()


# -- scoring -------------------------------------------------------------------------------------

def missing_files(args, tag):
    out = []
    for part in component_tags(tag):
        for run in args.runs:
            _, test_ids = run_spec(run)
            out += [proba_path(args.cache_dir, part, run, s, m) for s in args.seeds for m in test_ids
                    if not os.path.isfile(proba_path(args.cache_dir, part, run, s, m))]
    return out


def match_proba(cache_dir, tag, run, seeds, match_id):
    """Mean over the component recipes (equal weight) of the mean over seeds of the cached probabilities."""
    return np.mean([np.mean([np.load(proba_path(cache_dir, part, run, s, match_id)) for s in seeds], axis=0)
                    for part in component_tags(tag)], axis=0)


def collect(args, tag, settings):
    """{setting: {run: {match: (spell, macro, n_pred, n_gt, dead recall, dead precision, alive min)}}}."""
    import possession_metrics as pm
    from mtah.external import features, learn, scoring
    refuse_holdout_scoring(args.runs)
    out = {s: {r: {} for r in args.runs} for s in settings}
    for run in args.runs:
        _, test_ids = run_spec(run)
        for m in test_ids:
            proba = match_proba(args.cache_dir, tag, run, args.seeds, m)
            fs, match, truth = load_test(run, m)
            g, _ = scoring.joined(truth, match.runs)
            alive = float(((g == pm.H) | (g == pm.A)).sum()) / 25 / 60
            for setting in settings:
                switch, dead = setting
                pred = features.expand(learn.decode_rows(proba, fs, switch, dead, dead), fs, match.n_frames)
                r = scoring.score(truth, pred, match.runs)
                p, _ = scoring.joined(pred, match.runs)
                ok = g != pm.X
                tp = ((g == pm.D) & (p == pm.D) & ok).sum()
                out[setting][run][str(m)] = (r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"],
                                             tp / max((g == pm.D).sum(), 1), tp / max(((p == pm.D) & ok).sum(), 1), alive)
            del fs, match, truth, proba
        gc.collect()
    return out


def print_runs(per_match):
    for run in sorted(per_match):
        v = np.array(list(per_match[run].values()), dtype=np.float64)
        alive = v[:, 6].sum()
        print("{:8s} n {:2d} | spellF1 {:.4f} macroF1 {:.4f} | spells/alive-min pred {:.3f} true {:.3f} | dead recall {:.3f} precision {:.3f} F1 {:.3f}".format(
            run, len(v), v[:, 0].mean(), v[:, 1].mean(), v[:, 2].sum() / alive, v[:, 3].sum() / alive, v[:, 4].mean(), v[:, 5].mean(),
            np.mean([dead_f1(r, p) for r, p in v[:, 4:6]])))


def print_dead(tag, per_match):
    """One line: dead-frame F1 (recall, precision) per domain, means over matches of the per-match values."""
    parts = []
    for run in sorted(per_match):
        v = np.array(list(per_match[run].values()), dtype=np.float64)
        parts.append("{} {:.3f} (R {:.3f} P {:.3f})".format(
            run, np.mean([dead_f1(r, p) for r, p in v[:, 4:6]]), v[:, 4].mean(), v[:, 5].mean()))
    print("dead F1 per domain {}: {}".format(tag, " ".join(parts)))

def print_lodo(per_match):
    spell, macro = domain_means(per_match)
    runs = sorted(spell)
    names = " ".join("{} {:.4f}".format(r, spell[r]) for r in runs)
    print("LODO spellF1 {:.4f} | {} | min {:.4f} | LODO macroF1 {:.4f}".format(
        np.mean(list(spell.values())), names, min(spell.values()), np.mean(list(macro.values()))))
    return spell, macro


def print_diff(label, new, ref, gate="GR"):
    """Print the paired difference and apply the gate ('GR' or 'G2'); True / False, None if not evaluated."""
    ds, dm = paired(new, ref)
    lo, hi = stratified_bootstrap(ds)
    dom_spell = {r: float(d.mean()) for r, d in ds.items()}
    lodo = float(np.mean(list(dom_spell.values())))
    macro = float(np.mean([d.mean() for d in dm.values()]))
    better = sum(int((d > 0).sum()) for d in ds.values())
    total = sum(len(d) for d in ds.values())
    print("{}: LODO spell diff {:+.4f} [{:+.4f}, {:+.4f}] | per domain {} | LODO macro diff {:+.4f} | matches better {}/{}".format(
        label, lodo, lo, hi, " ".join("{} {:+.4f}".format(r, dom_spell[r]) for r in sorted(dom_spell)), macro, better, total))
    if len(ds) == 3:
        ok, reasons = (gate_gr if gate == "GR" else gate_g2)(lodo, lo, dom_spell, macro)
        print("  {} {}".format(gate, "PASS" if ok else "FAIL"))
        for line in reasons:
            print("   " + line)
        return ok
    print("  {} needs all three runs; not evaluated".format(gate))
    return None


def score_only(args, recipe, tag):
    refuse_holdout_scoring(args.runs)
    missing = missing_files(args, tag)
    if missing:
        print("missing {} cache files, not scoring:\n  ".format(len(missing)) + "\n  ".join(missing))
        sys.exit(1)
    ref_recipe = args.reference or ("R0" if recipe != "R0" else None)
    ref_tag = cache_tag(ref_recipe, args.steps) if ref_recipe else None
    ref2_tag = cache_tag(args.reference2, args.steps) if args.reference2 else None
    for extra in (ref_tag, ref2_tag):
        gone = missing_files(args, extra) if extra else []
        if gone:
            print("missing {} reference cache files ({}):\n  ".format(len(gone), extra) + "\n  ".join(gone))
            sys.exit(1)
    print("recipe {} (tag {}), seeds {}, decoder {:g} / {:g}, T = 2.5".format(
        recipe, tag, " ".join(map(str, args.seeds)), *DEFAULT_DECODER))
    settings = list(GRID) if args.decoder_table else [DEFAULT_DECODER]
    res = collect(args, tag, settings)
    base = res[DEFAULT_DECODER]
    print_runs(base)
    if len(args.runs) == 3:
        print_lodo(base)
    ref_res = collect(args, ref_tag, settings) if ref_tag else None
    ref2_res = collect(args, ref2_tag, settings) if ref2_tag else None
    print()
    print_dead(tag, base)
    for name, res_x in ((ref_tag, ref_res), (ref2_tag, ref2_res)):
        if res_x:
            print_dead(name, res_x[DEFAULT_DECODER])
    gr = g2 = None
    if ref_res:
        print()
        gr = print_diff("{} vs {}".format(tag, ref_tag), base, ref_res[DEFAULT_DECODER], "GR")
    if ref2_res:
        print()
        g2 = print_diff("{} vs {}".format(tag, ref2_tag), base, ref2_res[DEFAULT_DECODER], "G2")
    word = {True: "PASS", False: "FAIL", None: "n/a"}
    print("\nVERDICT {}: GR {}, G2 {}".format(tag, word[gr], word[g2]))
    if args.decoder_table:
        decoder_table(tag, res)
        if ref_res:
            decoder_table(ref_tag, ref_res)


def decoder_table(tag, res):
    print("\n-- decoder table {} (to_dead = from_dead = dead) --".format(tag))
    for switch, dead in GRID:
        spell, macro = domain_means(res[(switch, dead)])
        print("switch {:g} dead {:g}: LODO spellF1 {:.4f} macroF1 {:.4f} | min {:.4f}".format(
            switch, dead, np.mean(list(spell.values())), np.mean(list(macro.values())), min(spell.values())))
    if len(next(iter(res.values()))) != 3:
        return
    chosen, nested = nested_choice(res)
    print("nested choice: " + " ".join("{}->switch {:g} dead {:g}".format(r, *chosen[r]) for r in sorted(chosen)))
    print_lodo(nested)
    print_diff("nested vs 2 / 4", nested, res[DEFAULT_DECODER])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recipe", default="R0", help="R0..R5, combinations with '+', e.g. R1+R3")
    ap.add_argument("--runs", nargs="+", default=list(RUNS), choices=ALL_RUNS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--reference", default=None, help="recipe to compare against (default R0 unless the recipe is R0)")
    ap.add_argument("--ensemble", nargs="+", default=None, help="2+ cached recipes averaged with equal weight (replaces --recipe)")
    ap.add_argument("--reference2", default=None, help="second reference (e.g. R4), compared with gate G2")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--decoder-table", action="store_true", help="decoder grid and nested choice (implies --score-only)")
    ap.add_argument("--steps", type=int, default=None, help="override the training steps (smoke tests); tag gets _st<steps>")
    ap.add_argument("--extra", default=None,
                    help="extra feature set(s) appended to the 92 columns (e.g. deadcue); tag gets _x<set>. DT carries deadcue itself")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    ap.add_argument("--adapt-steps", type=int, default=None, help=argparse.SUPPRESS)    # smoke tests of R9; tag gets _ad<n>
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--no-score", action="store_true", help="stop after writing the probability caches (required for L-PFFH)")
    ap.add_argument("--sanity-game", nargs="+", default=None,
                    help="L-PFFH only: extra game(s) of PFF 32-63 to predict, to compare with the cached L-PFF probabilities")
    args = ap.parse_args()
    if any(r in HOLDOUT_RUNS for r in args.runs):
        if not args.no_score or args.score_only or args.decoder_table or args.ensemble:
            refuse_holdout_scoring(args.runs)
    if args.ensemble:
        for part in args.ensemble:
            recipe_kwargs(part)
        args.recipe = tag = ensemble_tag([cache_tag(r, args.steps) for r in args.ensemble])
        args.score_only = True                       # an ensemble never trains
    else:
        adapts = "adapt_steps" in recipe_kwargs(args.recipe)
        tag = cache_tag(args.recipe, args.steps, args.adapt_steps if adapts else None, extra_set(args.recipe, args.extra))
    if is_fused(args.recipe):
        args.score_only = True                       # fused tags are written by scripts/lodo_deadfuse.py, never trained here
    if not (args.score_only or args.decoder_table):
        train(args, args.recipe, tag)
        if not args.ensemble and "adapt_steps" in recipe_kwargs(args.recipe):
            print("R9-type recipe: training done; score with --score-only in a new process")
            return
    if args.no_score:
        print("{} (tag {}): probability caches written, --no-score: nothing scored".format(args.recipe, tag))
        return
    score_only(args, args.recipe, tag)


if __name__ == "__main__":
    main()
