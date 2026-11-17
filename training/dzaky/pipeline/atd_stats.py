"""MS-TCN stage -- how damaged is the real ATD, and do the model's inputs look familiar on it?

    python -m pipeline.atd_stats --build-reference   # DFL only: models/atd_reference.json
    python -m pipeline.atd_stats                     # the ATD: report + models/noise_atd.json

The first command reads nothing but the external DFL matches. It stores, as reference:

    the same statistics on clean DFL and on DFL damaged with the default NoiseSpec
    calibration tables (knob value -> statistic) for the knobs that cannot be read off directly
    a small logistic model telling a stoppage gap from a live-play gap, fitted on damaged DFL
    the 39 features' distributions and the fold models' confidence on damaged held-out DFL
    a recovery check: the estimator run on damaged DFL with known knobs, leave-one-match-out

The second command is the only one that touches BAR-ATH, and uses no ground truth -- not even the
public window. It prints aggregates only (shares, percentiles, histograms, one number per
feature), never a frame, a timestamp or an event, estimates every NoiseSpec knob, flags the
features whose ATD values sit outside what the model was trained on, and writes the suggested
training ranges to models/noise_atd.json, which `pipeline.train_tcn --noise` reads.

Whether stoppage gaps are stoppages cannot be known without the truth; it is estimated from what
a stoppage looks like in the tracking (players slow down, the ball was last seen near a line),
learned on DFL. Measured jitter includes real movement, so on its own it is an upper bound; the
calibration tables measure DFL the same way, which takes most of that out.
"""

import argparse
import json
import os

import numpy as np

from pipeline import features
from pipeline.atd_noise import FPS, NoiseSpec, RANGES, damage
from pipeline.config import ROOT, Config
from pipeline.dfl import available_matches, load_match
from pipeline.labels import label_match
from pipeline.runs import true_runs
from pipeline.train_tcn import MODELS_DIR, fold_model_path

REFERENCE = os.path.join(MODELS_DIR, "atd_reference.json")
SUGGESTED = os.path.join(MODELS_DIR, "noise_atd.json")
DEFAULT_MODEL = os.path.join(MODELS_DIR, "tcn1.pt")

LONG_S = 5.0                  # a "long" ball gap, as in NoiseSpec.LONG_GAP_MIN_S
SHORT_S = 0.5                 # a short drop-out, as Bagian 1 interpolates
GAP_BINS_S = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0, np.inf)
FULL_TEAM = 11
FULL_BLOCK_S = 60             # a team's "full" count is the most it shows within such a block
WOBBLE_HALF = 4               # frames either side of the centred mean jitter is measured against
LINE_CAP = 15.0               # metres: distance-to-line descriptor saturates here
CONFIDENT = 0.6               # top probability below this = the model hesitates
DESCRIPTORS = ("log_length_s", "speed_in_gap", "speed_before", "speed_first_3s",
               "speed_last_3s", "last_seen_line_m", "last_seen_out")

# Calibration grids: the knob is varied alone, everything else at the NoiseSpec default.
CAL_MATCHES = 3
CAL_GRID = {"SHARE_ON_DEAD": (0.0, 0.25, 0.5, 0.75, 1.0),
            "BALL_JITTER": (0.0, 0.2, 0.4, 0.8, 1.5),
            "PLAYER_JITTER": (0.0, 0.1, 0.25, 0.5, 1.0)}
CLASSIFIER_COPIES = 2
RECOVERY_SEED = 6000
REFERENCE_SEED = 5000


# --- measurements ------------------------------------------------------------------------------

def graded_mask(n, halves):
    out = np.zeros(n, dtype=bool)
    for _, first, last in halves:
        out[first:last + 1] = True
    return out


def ball_gaps(tracking, halves):
    """[(start, stop)] runs of missing ball, cut at the half boundaries."""
    out = []
    missing = np.isnan(tracking.ball[:, 0])
    for _, first, last in halves:
        out.extend((first + a, first + b) for a, b in true_runs(missing[first:last + 1]))
    return out


def frame_speed(tracking):
    """Mean reported player speed per frame (m/s), NaN where nobody is seen."""
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(tracking.speeds, axis=1)


def ball_stats(tracking, halves):
    graded = graded_mask(len(tracking), halves)
    minutes = graded.sum() / (60.0 * FPS)
    gaps = ball_gaps(tracking, halves)
    lengths = np.array([b - a for a, b in gaps], dtype=float) / FPS
    bins = np.histogram(lengths, bins=GAP_BINS_S)[0] if len(lengths) else np.zeros(8, int)
    frames = [float(lengths[(lengths >= lo) & (lengths < hi)].sum() * FPS / graded.sum())
              for lo, hi in zip(GAP_BINS_S[:-1], GAP_BINS_S[1:])]
    long = lengths[lengths >= LONG_S]
    return {"missing_share": float(np.isnan(tracking.ball[graded, 0]).mean()),
            "long_gap_share": float(long.sum() * FPS / graded.sum()),
            "gaps_per_90": [float(c * 90.0 / minutes) for c in bins],
            "gap_frame_share": frames,
            "long_gaps_per_90": float(len(long) * 90.0 / minutes),
            "long_gap_s_p25_50_75_90": ([float(np.percentile(long, q)) for q in (25, 50, 75, 90)]
                                        if len(long) else [0.0] * 4),
            "short_drops_per_min": float(np.sum(lengths <= SHORT_S) / minutes)}


def gap_descriptors(tracking, halves, cfg):
    """(gaps, lengths in frames, (k, len(DESCRIPTORS)) array) for the long ball gaps."""
    speed = frame_speed(tracking)
    long = [(a, b) for a, b in ball_gaps(tracking, halves) if b - a >= LONG_S * FPS]
    rows = []
    three = 3 * FPS
    for a, b in long:
        before = tracking.ball[max(0, a - 1)]
        if np.isnan(before[0]):
            line, out = LINE_CAP, 0.0
        else:
            x, y = before
            line = min(LINE_CAP, abs(x), abs(cfg.LAPANGAN_P - x), abs(y), abs(cfg.LAPANGAN_L - y))
            out = float(x < 0 or y < 0 or x > cfg.LAPANGAN_P or y > cfg.LAPANGAN_L)
        rows.append([np.log(max(1, b - a) / FPS), np.nanmean(speed[a:b]),
                     np.nanmean(speed[max(0, a - three):a]), np.nanmean(speed[a:a + three]),
                     np.nanmean(speed[max(a, b - three):b]), line, out])
    table = np.nan_to_num(np.array(rows, dtype=float).reshape(-1, len(DESCRIPTORS)))
    return long, np.array([b - a for a, b in long], dtype=float), table


def team_counts(tracking, halves):
    """{team: visible players per graded frame}."""
    graded = graded_mask(len(tracking), halves)
    seen = ~np.isnan(tracking.players[:, :, 0])
    return {code: seen[:, tracking.team == code].sum(1)[graded] for code in (0, 1)}


def player_stats(tracking, halves):
    """Visible players per team, and the deficit below each minute's full count."""
    counts = team_counts(tracking, halves)
    block = FULL_BLOCK_S * FPS
    out = {"visible_p5_50_95": [], "share_below_10": [], "share_above_11": [],
           "missing_fraction": [], "deficit_runs_per_min": [], "deficit_run_mean_s": [],
           "drops_per_player_min": [],
           "track_slots": int(len(tracking.team))}
    every_run = []
    for v in counts.values():
        full = np.minimum(FULL_TEAM, np.concatenate([np.full(len(v[i:i + block]), v[i:i + block].max())
                                                     for i in range(0, len(v), block)]))
        deficit = np.maximum(0, full - v)
        runs = [b - a for a, b in true_runs(deficit > 0)]
        minutes = len(v) / (60.0 * FPS)
        out["visible_p5_50_95"].append([float(np.percentile(v, q)) for q in (5, 50, 95)])
        out["share_below_10"].append(float(np.mean(v < 10)))
        out["share_above_11"].append(float(np.mean(v > FULL_TEAM)))
        out["missing_fraction"].append(float(deficit.sum() / max(1, full.sum())))
        out["deficit_runs_per_min"].append(float(len(runs) / minutes))
        every_run.extend(runs)
        # Every player dropping out raises the deficit by one, even while others are out, so
        # the rises count drops where the runs of deficit would have merged.
        rises = np.maximum(0, np.diff(deficit)).sum()
        out["drops_per_player_min"].append(float(rises / max(1.0, full.mean() * minutes)))
    for key in ("share_below_10", "share_above_11", "missing_fraction", "deficit_runs_per_min",
                "drops_per_player_min"):
        out[key] = float(np.mean(out[key]))
    out["deficit_run_mean_s"] = float(np.mean(every_run) / FPS) if every_run else 0.0
    out["visible_p5_50_95"] = np.mean(out["visible_p5_50_95"], axis=0).tolist()
    return out


def wobble(xy, half=WOBBLE_HALF, chunk=32):
    """Median |position - centred mean| (m) over points whose whole window is seen."""
    if xy.ndim == 2:
        xy = xy[:, None, :]
    width = 2 * half + 1
    residuals = []
    for start in range(0, xy.shape[1], chunk):
        part = xy[:, start:start + chunk].astype(np.float64)
        seen = ~np.isnan(part[..., 0])
        zero = np.zeros((1,) + part.shape[1:])
        sums = np.cumsum(np.concatenate([zero, np.nan_to_num(part)]), axis=0)
        counts = np.cumsum(np.concatenate([zero[..., 0], seen]), axis=0)
        window = sums[width:] - sums[:-width]
        full = (counts[width:] - counts[:-width]) == width
        centre = part[half:len(part) - half]
        residuals.append(np.abs(centre - window / width)[full].ravel())
    values = np.concatenate(residuals)
    return float(np.median(values)) if len(values) else 0.0


def speed_ratio(tracking, step=FPS // 2):
    """Median reported speed / speed from positions, where a player moves over 3 m/s.

    Displacement over half a second, so position jitter barely moves it; curved runs make it
    read slightly above 1 on clean data.
    """
    p = tracking.players
    moved = np.hypot(*(p[step:] - p[:-step]).transpose(2, 0, 1)) * FPS / step
    reported = (tracking.speeds[step:] + tracking.speeds[:-step]) / 2.0
    ok = np.isfinite(moved) & np.isfinite(reported) & (moved > 3.0)
    return float(np.median(reported[ok] / moved[ok])) if ok.any() else float("nan")


def measure(tracking, halves):
    """Every non-model statistic of one match, all aggregates."""
    return {"ball": ball_stats(tracking, halves), "players": player_stats(tracking, halves),
            "ball_wobble_m": wobble(tracking.ball), "player_wobble_m": wobble(tracking.players),
            "speed_ratio": speed_ratio(tracking)}


def feature_stats(table, graded):
    rows = table[graded]
    return {name: {"mean": float(rows[:, i].mean()), "std": float(rows[:, i].std()),
                   "p": [float(np.percentile(rows[:, i], q)) for q in (1, 5, 50, 95, 99)]}
            for i, name in enumerate(features.FEATURES)}


def confidence(probabilities, graded):
    top = probabilities[graded].max(1)
    shares = np.bincount(probabilities[graded].argmax(1), minlength=3) / graded.sum()
    return {"mean_top": float(top.mean()), "share_hesitant": float(np.mean(top < CONFIDENT)),
            "class_share_HAD": shares.tolist()}


# --- logistic model and calibration ------------------------------------------------------------

def fit_logistic(x, y, l2=1e-2, iterations=30):
    """Newton-fitted logistic regression on standardised x."""
    mean, std = x.mean(0), x.std(0) + 1e-6
    z = np.hstack([np.ones((len(x), 1)), (x - mean) / std])
    w = np.zeros(z.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-z @ w))
        gradient = z.T @ (p - y) + l2 * w
        hessian = (z * (p * (1 - p))[:, None]).T @ z + l2 * np.eye(len(w))
        w -= np.linalg.solve(hessian, gradient)
    return {"w": w.tolist(), "mean": mean.tolist(), "std": std.tolist()}


def predict_logistic(model, x):
    if not len(x):
        return np.zeros(0)
    z = np.hstack([np.ones((len(x), 1)), (x - np.array(model["mean"])) / np.array(model["std"])])
    return 1.0 / (1.0 + np.exp(-z @ np.array(model["w"])))


def dead_score(model, lengths, x):
    """Frame-weighted mean probability that a long gap is a stoppage."""
    p = predict_logistic(model, x)
    return float(np.sum(p * lengths) / lengths.sum()) if len(lengths) else 0.0


def invert(table, value):
    """Knob value whose statistic is value, by linear interpolation in a monotonic table."""
    knobs, stats = np.array(table["knob"]), np.array(table["stat"])
    order = np.argsort(stats)
    return float(np.interp(value, stats[order], knobs[order]))


def gap_targets(gaps, gt):
    """1 where most of a gap is a stoppage in the truth."""
    return np.array([np.mean(gt[a:b] == "D") >= 0.5 for a, b in gaps], dtype=float)


def build_estimator(matches):
    """Classifier and calibration tables, from damaged copies of matches only."""
    xs, ys = [], []
    for m_index, match in enumerate(matches):
        tracking, gt, halves, cfg = load_match(match)
        for copy in range(CLASSIFIER_COPIES):
            seed = REFERENCE_SEED + 100 * m_index + copy
            spec = NoiseSpec.sample(np.random.default_rng(seed),
                                    {"SHARE_ON_DEAD": (0.0, 1.0)})
            gaps, _, x = gap_descriptors(damage(tracking, gt, halves, spec, seed=seed), halves, cfg)
            xs.append(x)
            ys.append(gap_targets(gaps, gt))
    classifier = fit_logistic(np.vstack(xs), np.concatenate(ys))

    calibration = {knob: {"knob": list(values), "stat": []} for knob, values in CAL_GRID.items()}
    loaded = [load_match(match) for match in matches[:CAL_MATCHES]]
    for knob, values in CAL_GRID.items():
        for value in values:
            stats = []
            for tracking, gt, halves, cfg in loaded:
                damaged = damage(tracking, gt, halves, NoiseSpec().replace(**{knob: value}), seed=0)
                if knob == "SHARE_ON_DEAD":
                    _, lengths, x = gap_descriptors(damaged, halves, cfg)
                    stats.append(dead_score(classifier, lengths, x))
                elif knob == "BALL_JITTER":
                    stats.append(wobble(damaged.ball))
                else:
                    stats.append(wobble(damaged.players))
            calibration[knob]["stat"].append(float(np.mean(stats)))
    return {"classifier": classifier, "calibration": calibration}


def estimate(measured, descriptors, estimator):
    """NoiseSpec knob -> value, from one match's measurements."""
    lengths, x = descriptors
    ball, players = measured["ball"], measured["players"]
    p_dead = predict_logistic(estimator["classifier"], x)
    live = lengths[p_dead < 0.5] / FPS
    rate = players["drops_per_player_min"]
    drop_s = 60.0 * players["missing_fraction"] / rate if rate > 0 else 0.0
    cal = estimator["calibration"]
    return {
        "BALL_MISSING": ball["long_gap_share"],
        "SHARE_ON_DEAD": invert(cal["SHARE_ON_DEAD"], dead_score(estimator["classifier"],
                                                                 lengths, x)),
        "LIVE_GAP_MEDIAN_S": float(np.median(live)) if len(live) else NoiseSpec.LIVE_GAP_MEDIAN_S,
        "LIVE_GAP_SIGMA": float(np.std(np.log(live))) if len(live) > 2 else NoiseSpec.LIVE_GAP_SIGMA,
        # Short drops landing inside a long gap cannot be seen; assume they fall anywhere.
        "SHORT_DROPS_PER_MIN": ball["short_drops_per_min"] / max(0.1, 1.0 - ball["missing_share"]),
        "BALL_JITTER": invert(cal["BALL_JITTER"], measured["ball_wobble_m"]),
        "PLAYER_JITTER": invert(cal["PLAYER_JITTER"], measured["player_wobble_m"]),
        # Drops of mean length l at r per player-minute hide r * l / 60 of the players;
        # lengths are uniform up to the max, so the max is about twice the mean.
        "PLAYER_DROPS_PER_MIN": rate,
        "PLAYER_DROP_MAX_S": 2.0 * drop_s,
    }


# Absolute floor on a suggested range's half-width, and hard bounds, per knob.
RANGE_RULES = {"BALL_MISSING": (0.08, 0.0, 0.9), "SHARE_ON_DEAD": (0.2, 0.0, 1.0),
               "DEAD_SPILL_S": (1.5, 0.0, 10.0), "LIVE_GAP_MEDIAN_S": (3.0, 5.0, 60.0),
               "LIVE_GAP_SIGMA": (0.2, 0.1, 1.5), "SHORT_DROPS_PER_MIN": (2.0, 0.0, 60.0),
               "BALL_JITTER": (0.15, 0.0, 3.0), "PLAYER_JITTER": (0.1, 0.0, 3.0),
               "PLAYER_DROPS_PER_MIN": (0.5, 0.0, 20.0), "PLAYER_DROP_MAX_S": (2.0, 0.5, 60.0)}


def suggest_ranges(estimates):
    """Training ranges centred on the estimates, +/- 30% or the floor, whichever is wider."""
    out = {}
    for knob, value in estimates.items():
        floor, low, high = RANGE_RULES[knob]
        half = max(0.3 * abs(value), floor)
        out[knob] = [round(max(low, value - half), 3), round(min(high, value + half), 3)]
    return out


# --- reference (DFL only) ----------------------------------------------------------------------

def _mean_dicts(dicts):
    """Element-wise mean of equally shaped nested dicts / lists / numbers."""
    first = dicts[0]
    if isinstance(first, dict):
        return {key: _mean_dicts([d[key] for d in dicts]) for key in first}
    if isinstance(first, list):
        return np.mean(np.array(dicts, dtype=float), axis=0).tolist()
    return float(np.mean(dicts))


def build_reference(matches):
    from pipeline.tcn import Predictor, deterministic
    deterministic()
    reference = {"matches": list(matches), "estimator": build_estimator(matches)}

    clean, damaged, tables, conf = [], [], [], []
    for m_index, match in enumerate(matches):
        tracking, gt, halves, cfg = load_match(match)
        clean.append(measure(tracking, halves))
        damaged.append(measure(damage(tracking, gt, halves, NoiseSpec(), seed=0), halves))
        seed = REFERENCE_SEED + 50 + m_index
        copy = damage(tracking, gt, halves, NoiseSpec.sample(np.random.default_rng(seed)), seed=seed)
        table = features.compute(copy, halves, cfg, label_match(copy, [], halves, cfg).labels)
        graded = gt != "X"
        tables.append(table[graded])
        path = fold_model_path("tcn1", match)
        if os.path.isfile(path):
            conf.append(confidence(Predictor.load(path).probabilities(table, halves), graded))
        print("reference: {}".format(match), flush=True)
    reference["clean"] = _mean_dicts(clean)
    reference["damaged_default"] = _mean_dicts(damaged)
    reference["features"] = feature_stats(np.vstack(tables), np.ones(sum(map(len, tables)), bool))
    reference["confidence"] = _mean_dicts(conf) if conf else None
    reference["recovery"] = recovery_check(matches)
    return reference


def recovery_check(matches):
    """The estimator on damaged DFL with known knobs; estimator built without that match."""
    rows = []
    wide = {"SHARE_ON_DEAD": (0.0, 1.0), "PLAYER_DROPS_PER_MIN": (0.0, 4.0),
            "PLAYER_DROP_MAX_S": (1.0, 10.0), "BALL_JITTER": (0.0, 1.5),
            "PLAYER_JITTER": (0.0, 1.0)}
    for m_index, match in enumerate(matches):
        estimator = build_estimator([m for m in matches if m != match])
        tracking, gt, halves, cfg = load_match(match)
        seed = RECOVERY_SEED + m_index
        spec = NoiseSpec.sample(np.random.default_rng(seed), wide)
        copy = damage(tracking, gt, halves, spec, seed=seed)
        _, lengths, x = gap_descriptors(copy, halves, cfg)
        found = estimate(measure(copy, halves), (lengths, x), estimator)
        rows.append({knob: [float(getattr(spec, knob)), value] for knob, value in found.items()})
        print("recovery: {}".format(match), flush=True)
    return rows


# --- report ------------------------------------------------------------------------------------

def flag_features(atd, reference, shift_limit=0.5, outside_limit=0.10, table=None):
    """[(name, shift in reference sd, share of ATD frames outside the reference p1-p99)]."""
    out = []
    for i, name in enumerate(features.FEATURES):
        if name not in reference or name not in atd:
            continue  # a feature added after the reference was built
        ref = reference[name]
        shift = (atd[name]["mean"] - ref["mean"]) / (ref["std"] + 1e-6)
        outside = 0.0
        if table is not None:
            column = table[:, i]
            outside = float(np.mean((column < ref["p"][0] - 1e-6) | (column > ref["p"][4] + 1e-6)))
        if abs(shift) > shift_limit or outside > outside_limit:
            out.append((name, float(shift), outside))
    return out


def print_report(measured, estimates, reference, flags, conf, smart):
    clean, default = reference["clean"], reference["damaged_default"]
    row = "  {:<34} {:>10} {:>10} {:>10}"
    fmt = lambda v: "{:.3f}".format(v) if isinstance(v, float) else str(v)  # noqa: E731

    print("\nA. ball" + " " * 29 + "{:>10} {:>10} {:>10}".format("ATD", "DFL clean", "DFL dmg"))
    for key in ("missing_share", "long_gap_share", "long_gaps_per_90", "short_drops_per_min"):
        print(row.format(key, fmt(measured["ball"][key]), fmt(clean["ball"][key]),
                         fmt(default["ball"][key])))
    print("  gap lengths (s)      gaps / 90 min    share of graded frames")
    for i, (lo, hi) in enumerate(zip(GAP_BINS_S[:-1], GAP_BINS_S[1:])):
        print("    {:>5}-{:<5} {:>12.1f} {:>8.1f} {:>14.3f} {:>8.3f}   (ATD, DFL dmg)".format(
            lo, hi, measured["ball"]["gaps_per_90"][i], default["ball"]["gaps_per_90"][i],
            measured["ball"]["gap_frame_share"][i], default["ball"]["gap_frame_share"][i]))
    print(row.format("long gap s p25/50/75/90", "", "", "") + "\n    ATD {}   DFL dmg {}".format(
        [round(v, 1) for v in measured["ball"]["long_gap_s_p25_50_75_90"]],
        [round(v, 1) for v in default["ball"]["long_gap_s_p25_50_75_90"]]))
    if smart is not None:
        print("  SET PIECES tags starting in a long gap: {:.0%} of {} tags; long gaps holding one: "
              "{:.0%}".format(*smart))

    print("\nB. players")
    for key in ("share_below_10", "share_above_11", "missing_fraction", "deficit_runs_per_min",
                "deficit_run_mean_s", "drops_per_player_min", "track_slots"):
        print(row.format(key, fmt(measured["players"][key]), fmt(clean["players"][key]),
                         fmt(default["players"][key])))
    print("  visible per team p5/50/95: ATD {}  DFL clean {}  DFL dmg {}".format(
        *[[round(v, 1) for v in d["players"]["visible_p5_50_95"]]
          for d in (measured, clean, default)]))

    print("\nC. positions")
    for key in ("ball_wobble_m", "player_wobble_m", "speed_ratio"):
        print(row.format(key, fmt(measured[key]), fmt(clean[key]), fmt(default[key])))
    if not 0.8 < measured["speed_ratio"] < 1.25:
        print("  !! reported speed is not m/s-consistent with the positions (ratio {:.2f})".format(
            measured["speed_ratio"]))

    print("\nD. features outside the training distribution "
          "(shift in training sd, share of frames outside training p1-p99)")
    for name, shift, outside in flags or [("none", 0.0, 0.0)]:
        print("  {:<22} {:>+7.2f} {:>8.1%}".format(name, shift, outside))

    if conf is not None:
        ref = reference["confidence"] or {}
        print("\nE. model confidence           ATD   DFL held-out")
        for key in ("mean_top", "share_hesitant"):
            print("  {:<26} {:>6.3f} {:>8.3f}".format(key, conf[key], ref.get(key, float("nan"))))
        print("  class share H/A/D          {}  {}".format(
            [round(v, 3) for v in conf["class_share_HAD"]],
            [round(v, 3) for v in ref.get("class_share_HAD", [])]))

    errors = recovery_errors(reference["recovery"])
    print("\nsuggested NoiseSpec  (estimate | current default | training range now | "
          "DFL recovery mean abs error)")
    for knob, value in estimates.items():
        print("  {:<22} {:>8.3f} {:>8.3f}   {:<14} {:>7.3f}".format(
            knob, value, getattr(NoiseSpec(), knob), str(list(RANGES.get(knob, ("fixed",)))),
            errors.get(knob, float("nan"))))
    medians = measured["players"]["visible_p5_50_95"][1]
    if medians < 10:
        print("  !! a team shows a median of {:.0f} players: the camera does not see the whole "
              "team, which no NoiseSpec knob models -- needs a new damage type".format(medians))


def recovery_errors(rows):
    return {knob: float(np.mean([abs(r[knob][1] - r[knob][0]) for r in rows])) for knob in rows[0]}


def smart_tagging_check(events, gaps, fps=FPS):
    """(share of SET PIECES starts inside a long gap +/- 1 s, count, share of gaps holding one)."""
    starts = [e["start_frame"] for e in events
              if (e["code"] or "").strip().lower() == "set pieces"]
    if not starts or not gaps:
        return None
    inside = [any(a - fps <= s < b + fps for a, b in gaps) for s in starts]
    holding = [any(a - fps <= s < b + fps for s in starts) for a, b in gaps]
    return float(np.mean(inside)), len(starts), float(np.mean(holding))


def run_on_atd(model_path, reference):
    """Report on BAR-ATH and the suggestions. Imports BAR-ATH loaders only here."""
    import sys
    from pipeline.data import load_events, load_tracking
    from pipeline.tcn import Predictor, deterministic
    sys.path.insert(0, ROOT)
    from games import GAME, half_video_blocks

    deterministic()
    cfg = Config()
    tracking = load_tracking(cfg=cfg)
    halves = half_video_blocks(GAME)
    graded = graded_mask(len(tracking), halves)

    measured = measure(tracking, halves)
    gaps, lengths, x = gap_descriptors(tracking, halves, cfg)
    estimates = estimate(measured, (lengths, x), reference["estimator"])

    rules = label_match(tracking, [], halves, cfg.replace(PAKAI_SMART_TAGGING=False)).labels
    table = features.compute(tracking, halves, cfg, rule_labels=rules)
    flags = flag_features(feature_stats(table, graded), reference["features"], table=table[graded])
    conf = None
    if os.path.isfile(model_path):
        conf = confidence(Predictor.load(model_path).probabilities(table, halves), graded)
    try:
        smart = smart_tagging_check(load_events(), gaps)
    except Exception:  # noqa: BLE001 -- Smart Tagging is optional evidence here
        smart = None
    print_report(measured, estimates, reference, flags, conf, smart)
    return estimates, flags


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build-reference", action="store_true",
                        help="DFL only: write models/atd_reference.json")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model whose confidence to report")
    parser.add_argument("--out", default=SUGGESTED, help="where the suggested ranges go")
    args = parser.parse_args()

    if args.build_reference:
        reference = build_reference(available_matches())
        with open(REFERENCE, "w") as handle:
            json.dump(reference, handle, indent=1)
        print("recovery mean abs error: {}".format(
            {k: round(v, 3) for k, v in recovery_errors(reference["recovery"]).items()}))
        print("wrote {}".format(os.path.relpath(REFERENCE, ROOT)))
        raise SystemExit(0)

    with open(REFERENCE) as handle:
        reference = json.load(handle)
    estimates, flags = run_on_atd(args.model, reference)
    with open(args.out, "w") as handle:
        json.dump({"estimates": {k: round(v, 4) for k, v in estimates.items()},
                   "ranges": suggest_ranges(estimates),
                   "flagged_features": [name for name, _, _ in flags]}, handle, indent=2)
    print("\nwrote {}  (train with: python -m pipeline.train_tcn --noise {})".format(
        os.path.relpath(args.out, ROOT), os.path.relpath(args.out, ROOT)))
