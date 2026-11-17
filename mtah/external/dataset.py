"""Feature sets of damaged IDSSE matches with their labels, cached.

A copy is identified by (match, variant, seed); variant "clean" is the undamaged conversion. The
features do not depend on the label mapping, the labels do: they are the mapped (or raw) code at
each row's middle frame. Cached in cache/external/ (derived from public IDSSE data, not committed).
"""

import dataclasses
import os

import numpy as np

from mtah import config
from mtah.external import convert, damage, features, idsse, metrica, models, pff, skillcorner

CACHE_VERSION = 2          # 2: IDSSE speeds converted from km/h, ball speed from positions


JITTER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "work_data", "overnight_cache", "feat")
# caches of the "_jit" variants and of SkillCorner extras live off C: (JITTER_DIR)
JITTER_SIGMA_M = (0.1, 0.5)                 # sigma ~ U(0.1, 0.5) m per block
JITTER_BLOCK_S = 20


def _cache_dir(variant):
    return JITTER_DIR if ("_jit" in variant or "_wp" in variant) else os.path.join(config.CACHE_DIR, "external")


def _path(match_id, variant, seed):
    return os.path.join(_cache_dir(variant),
                        "feat_v{}_{}_{}_s{}.npz".format(CACHE_VERSION, match_id, variant, seed))


def add_ball_jitter(tracking, match_id, variant, seed):
    """Copy of `tracking` with i.i.d. Gaussian noise (metres, drawn per 20 s block) on the observed ball x / y."""
    block = JITTER_BLOCK_S * features.FPS
    ball = tracking.ball.copy()
    n = len(ball)
    n_blocks = -(-n // block)
    rng = damage.rng_for(match_id, variant, seed, 7)
    sigma = np.repeat(rng.uniform(*JITTER_SIGMA_M, size=n_blocks), block)[:n]
    noise = rng.normal(size=(n, 2)) * sigma[:, None]
    seen = ~np.isnan(ball[:, 0])
    length, width = tracking.pitch
    ball[seen, 0] += (noise[seen, 0] / length).astype(ball.dtype)
    ball[seen, 1] += (noise[seen, 1] / width).astype(ball.dtype)
    return dataclasses.replace(tracking, ball=ball)


WP_BLOCK_S = 20            # his 100 samples at 5 Hz
WP_P = 0.5                  # chance that a block is perturbed (his 0.5)
WP_DRIFT_M = (0.5, 3.0)


def add_wira_perturbation(tracking, match_id, variant, seed):
    """Copy of `tracking` with Wira's ball perturbations (wira_plus.augment kinds 0-3, ported to 25 Hz).

    Per 20 s block, with probability 0.5, one of: (0) coordinates rounded to 1% of the pitch (players and ball),
    (1) a ball drift of 0.5-3 m in a random direction, ramped over 1 s at both ends, (2) the ball linearly
    interpolated between its own positions every 1 / 2 / 3 s, (3) the ball shifted in time by +-0.2 / +-0.4 s.
    Frames where the ball was missing stay missing. Never applied to the held-out copy.
    """
    rng = damage.rng_for(match_id, variant, seed, 11)
    fps = features.FPS
    block = WP_BLOCK_S * fps
    ball = tracking.ball.copy()
    players = tracking.players.copy()
    original = ball.copy()
    length, width = tracking.pitch
    scale = np.array([length, width], dtype=np.float64)
    n = len(ball)
    for a in range(0, n, block):
        b = min(n, a + block)
        if rng.random() >= WP_P:
            continue
        kind = int(rng.integers(4))
        if kind == 0:
            players[a:b, :, :2] = np.round(players[a:b, :, :2] * 100.0) / 100.0
            ball[a:b, :2] = np.round(ball[a:b, :2] * 100.0) / 100.0
        elif kind == 1:
            theta = rng.uniform(0, 2 * np.pi)
            v = rng.uniform(*WP_DRIFT_M) * np.array([np.cos(theta), np.sin(theta)]) / scale
            i = np.arange(b - a)
            ramp = np.minimum(1.0, np.minimum(i / (fps), i[::-1] / (fps)))
            ball[a:b, :2] += (ramp[:, None] * v).astype(ball.dtype)
        elif kind == 2:
            step = int(rng.choice([1, 2, 3])) * fps
            knots = np.unique(np.r_[np.arange(a, b, step), b - 1])
            for left, right in zip(knots[:-1], knots[1:]):
                if np.isfinite(original[[left, right], :2]).all():
                    t = np.linspace(0.0, 1.0, right - left + 1)[:, None]
                    ball[left:right + 1, :2] = original[left, :2] * (1 - t) + original[right, :2] * t
        else:
            lag = int(rng.choice([-2, -1, 1, 2])) * (fps // 5) * 1
            dest = np.arange(a, b)
            src = dest + lag
            ok = (src >= a) & (src < b)
            ball[dest[ok], :2] = original[src[ok], :2]
    ball[~np.isfinite(original[:, 0]), :2] = np.nan
    return dataclasses.replace(tracking, ball=ball, players=players)


def tracking_for(match, variant, seed, profile=None):
    """Tracking of a copy; "<variant>_indep" is the damage without the link to dead balls."""
    if variant.endswith("_wp"):          # the same copy, then Wira's ball perturbations (training copies only)
        return add_wira_perturbation(tracking_for(match, variant[:-len("_wp")], seed, profile), match.match_id, variant, seed)
    if variant.endswith("_jit"):         # the same copy, then jitter on the observed ball (never in the held-out copy)
        return add_ball_jitter(tracking_for(match, variant[:-len("_jit")], seed, profile), match.match_id, variant, seed)
    if variant == "clean":
        return convert.to_tracking(match)
    base, independent = (variant[:-len("_indep")], True) if variant.endswith("_indep") else (variant, False)
    return damage.damage(match, base, seed, dead_dependence=not independent, profile=profile).tracking


def _with_extra(fs, extra, make_tracking, match, path_for, use_cache):
    """fs with the extra columns of set `extra` appended; each component set is cached on its own."""
    from mtah.external import extra_features
    features.register_extra(extra_features.names(extra))
    tracking, blocks = [], []
    for part in extra_features.components(extra):
        path = path_for(part)
        if use_cache and os.path.isfile(path):
            with np.load(path) as cached:
                matrix = cached["x"]
        else:
            if not tracking:
                tracking.append(make_tracking())
            matrix = extra_features.compute_extra(tracking[0], match.runs, tuple(a for a, _ in match.runs), part)
            if use_cache:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                partial = path + ".partial"
                with open(partial, "wb") as handle:
                    np.savez(handle, x=matrix)
                os.replace(partial, path)
        assert len(matrix) == len(fs.x), "extra rows are not aligned with the base rows"
        blocks.append(matrix)
    return features.FeatureSet(x=np.concatenate([fs.x] + blocks, axis=1), frame=fs.frame, block=fs.block,
                               runs=fs.runs)


def load_match(match_id):
    """The undamaged Match of an IDSSE id or of a Metrica game ("MET1" / "MET2": a fourth training source, never a test set)."""
    return metrica.load(match_id) if str(match_id) in metrica.GAMES else idsse.load(match_id)


def feature_set(match_id, variant, seed, use_cache=True, extra=None):
    """(FeatureSet, Match) of one copy; extra: set name(s) of extra_features appended to the 92 columns.
    match_id is an IDSSE id or a Metrica game (metrica.GAMES): damaged exactly alike, cached as feat_v2_MET1_<variant>_s<seed>."""
    if extra is not None:
        from mtah.external import extra_features
        fs, match = feature_set(match_id, variant, seed, use_cache)
        return _with_extra(
            fs, extra, lambda: tracking_for(match, variant, seed), match,
            lambda part: os.path.join(_cache_dir(variant), "featx_v{}_{}_{}_{}_s{}.npz".format(
                extra_features.EXTRA_VERSION, part, match_id, variant, seed)), use_cache), match
    match = load_match(match_id)
    path = _path(match_id, variant, seed)
    if use_cache and os.path.isfile(path):
        with np.load(path) as cached:
            runs = [tuple(r) for r in cached["runs"]]
            return features.FeatureSet(x=cached["x"], frame=cached["frame"], block=cached["block"], runs=runs), match
    tracking = tracking_for(match, variant, seed)
    codes, p_dead = models.aji_rules(tracking, match.runs)
    fs = features.compute(tracking, match.runs, tuple(a for a, _ in match.runs), codes, p_dead)
    if use_cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        partial = path + ".partial"
        with open(partial, "wb") as handle:
            np.savez(handle, x=fs.x, frame=fs.frame, block=fs.block, runs=np.array(fs.runs))
        os.replace(partial, path)
    return fs, match


def row_labels(match, fs, mapping=1.5):
    """(rows,) label code at each row's middle frame; mapping None = raw DFL flags."""
    return match.labels(mapping=mapping)[fs.frame]


def sc_feature_set(match_id, variant="sc", use_cache=True, extra=None):
    """(FeatureSet, Match, rule codes) of one SkillCorner match: its real damage, slots assigned with a seed
    from the id; the rule codes are `aji`'s labels on the same tracking."""
    if extra is not None:
        from mtah.external import extra_features
        fs, match, codes = sc_feature_set(match_id, variant, use_cache)
        return _with_extra(
            fs, extra, lambda: convert.to_tracking(match, rng=np.random.default_rng(int(match_id))), match,
            lambda part: os.path.join(JITTER_DIR, "featx_sc_v{}_{}_{}_{}.npz".format(
                extra_features.EXTRA_VERSION, part, match_id, variant)), use_cache), match, codes
    match = skillcorner.load(match_id, variant, "phase")
    path = os.path.join(config.CACHE_DIR, "external", "feat_sc_v1_{}_{}.npz".format(match_id, variant))
    if use_cache and os.path.isfile(path):
        with np.load(path) as cached:
            runs = [tuple(r) for r in cached["runs"]]
            return (features.FeatureSet(x=cached["x"], frame=cached["frame"], block=cached["block"], runs=runs),
                    match, cached["codes"])
    tracking = convert.to_tracking(match, rng=np.random.default_rng(int(match_id)))
    codes, p_dead = models.aji_rules(tracking, match.runs)
    fs = features.compute(tracking, match.runs, tuple(a for a, _ in match.runs), codes, p_dead)
    if use_cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        partial = path + ".partial"
        with open(partial, "wb") as handle:
            np.savez(handle, x=fs.x, frame=fs.frame, block=fs.block, runs=np.array(fs.runs), codes=codes)
        os.replace(partial, path)
    return fs, match, codes


def pff_tracking(match, game_id, decision=None):
    """Tracking of one PFF copy (seed 1): as delivered (slots seeded by the game id) unless the PFF_PLAN section 4
    decision adds damage at its level."""
    decision = decision or pff.damage_decision()
    if not decision["applies"]:
        return convert.to_tracking(match, rng=np.random.default_rng(int(game_id)))
    return damage.damage(match, "match", 1, level=decision["level"]).tracking


def pff_feature_set(game_id, extra=None, use_cache=True):
    """(FeatureSet, Match) of one PFF game (one copy, seed 1), cached in the PFF cache dir on D:."""
    game_id = str(game_id)
    decision = pff.damage_decision()
    tag = "d{:.3f}".format(decision["level"]) if decision["applies"] else "asis"
    match = pff.load(game_id)

    def atomic(path, **arrays):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        partial = path + ".partial"
        with open(partial, "wb") as handle:
            np.savez(handle, **arrays)
        os.replace(partial, path)

    path = os.path.join(pff.CACHE_DIR, "feat_v{}_{}_{}.npz".format(pff.FORMAT_VERSION, game_id, tag))
    if use_cache and os.path.isfile(path):
        with np.load(path) as cached:
            fs = features.FeatureSet(x=cached["x"], frame=cached["frame"], block=cached["block"],
                                     runs=[tuple(r) for r in cached["runs"]])
    else:
        tracking = pff_tracking(match, game_id, decision)
        codes, p_dead = models.aji_rules(tracking, match.runs)
        fs = features.compute(tracking, match.runs, tuple(a for a, _ in match.runs), codes, p_dead)
        if use_cache:
            atomic(path, x=fs.x, frame=fs.frame, block=fs.block, runs=np.array(fs.runs))
    if extra is None:
        return fs, match
    from mtah.external import extra_features
    return _with_extra(
        fs, extra, lambda: pff_tracking(match, game_id, decision), match,
        lambda part: os.path.join(pff.CACHE_DIR, "featx_v{}_{}_{}_{}.npz".format(
            extra_features.EXTRA_VERSION, part, game_id, tag)), use_cache), match
