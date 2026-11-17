"""Features at 5 Hz for the learned models.

One function, `compute`, for damaged IDSSE and for BAR-ATH alike: it reads an ATD-like Tracking
(io_atd.Tracking: slots, NaN where nothing is tracked) and the halves, nothing else. Every fifth
frame of a half is a row (the middle frame of its block of five); a row's label covers its block.

The set follows Wira's published list (ball and observation status, per team nearest distances, counts
within 2 / 5 / 10 m, speed median / 75th percentile / share under 1 m/s, spread, centroid distance,
nearest distance at the previous and next ball observation, 1 / 4 / 12 s means) plus the outputs of our
own rules as inputs (Dzaky's idea): the owner rule's team call and P(dead). Positions are in the home
team's attacking frame (home attacks towards +x, read off the kickoff formation like
features.attack_signs), so the same feature means the same thing in both halves.

Where the ball is missing, distances use an estimated ball: linear between the observations either
side of a gap of up to INTERP_S seconds, else the last (or next) observed position; the features
`ball_seen`, `since_last` and `until_next` tell the model how much to trust it. NaN stays NaN
(sklearn's boosting reads it natively; the network gets a fill value and a mask).

Team swap (the augmentation, and the test-time average): exchange the home_ and away_ columns,
negate the SIGNED ones (positions in the attacking frame), exchange rule_H and rule_A, and
exchange the labels H and A.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from mtah.external.convert import ball_speed_ms

FPS = 25
STEP = 5                      # frames per row (5 Hz)
HZ = FPS // STEP
CAP_S = 30.0                  # since_last / until_next cap
INTERP_S = 5.0                # longest ball gap bridged by linear interpolation
WINDOWS_S = (1, 4, 12)
SLOW = 1.0                    # m/s
KICKOFF_FRAMES = 50           # 2 s after the kickoff to read the attacking direction from
NEAR_RADII = (2.0, 5.0, 10.0)

BALL = ("ball_seen", "since_last", "until_next", "ball_x_att", "ball_x_abs", "ball_y_abs", "boundary",
        "ball_speed", "ball_accel", "ball_x_last_att", "ball_x_next_att")
BALL_WINDOWED = ("ball_seen", "boundary", "ball_speed")
TEAM = ("n_seen", "d1", "d2", "d3", "c2", "c5", "c10", "speed_med", "speed_p75", "slow_frac",
        "spread_x", "spread_y", "cdist", "cxrel", "d1_prev", "d1_next")
TEAM_WINDOWED = ("n_seen", "d1", "c5", "speed_med", "speed_p75", "slow_frac")
RULES = ("rule_H", "rule_A", "rule_D", "rule_pdead")
SIGNED = frozenset({"ball_x_att", "ball_x_last_att", "ball_x_next_att", "home_cxrel", "away_cxrel"})


def feature_names():
    names = list(BALL)
    names += ["{}_{}".format(side, name) for side in ("home", "away") for name in TEAM]
    for w in WINDOWS_S:
        names += ["{}_w{}".format(name, w) for name in BALL_WINDOWED]
        names += ["{}_{}_w{}".format(side, name, w) for side in ("home", "away") for name in TEAM_WINDOWED]
    names += list(RULES)
    return names


NAMES = feature_names()
INDEX = {name: i for i, name in enumerate(NAMES)}


def swap_plan(names=None):
    """(permutation, sign) so that X[:, permutation] * sign is the team-swapped feature matrix.

    names defaults to NAMES; a longer list (NAMES + extra columns) gives the plan of an extended matrix.
    """
    names = NAMES if names is None else list(names)
    index = {name: i for i, name in enumerate(names)}

    def swapped(name):
        if name.startswith("home_"):
            return "away_" + name[len("home_"):]
        if name.startswith("away_"):
            return "home_" + name[len("away_"):]
        return {"rule_H": "rule_A", "rule_A": "rule_H"}.get(name, name)
    permutation = np.array([index[swapped(name)] for name in names])
    sign = np.array([-1.0 if swapped(name) in SIGNED or name in SIGNED else 1.0 for name in names],
                    dtype=np.float32)
    # home_cxrel <-> away_cxrel exchange AND negate; ball_x_* negate in place
    return permutation, sign


PERMUTATION, SIGN = swap_plan()
SWAP_LABELS = np.array([1, 0, 2], dtype=np.int8)       # H <-> A, D stays

# Every column that comes from the rule model: the owner rule's team call and P(dead). No windowed
# means of them exist. `keep_columns(drop_rules=True)` is what the learned models read instead.
RULE_COLUMNS = RULES


# Extended matrices (NAMES + optional extra columns, mtah/external/extra_features.py): one swap plan per
# registered width. Widths must be unique, so swap_teams can dispatch on x.shape[1].
_EXTRA_PLANS = {}


def register_extra(names):
    """Register NAMES + names as an extended layout; returns its width. Same width twice needs the same names."""
    names = tuple(names)
    full = list(NAMES) + list(names)
    width = len(full)
    assert len(set(full)) == width, "duplicate column names in the extended layout"
    if width in _EXTRA_PLANS:
        if _EXTRA_PLANS[width][0] != names:
            raise ValueError("width {} is already registered for other extra columns".format(width))
        return width
    assert width != len(NAMES), "an empty extra set is the base layout"
    permutation, sign = swap_plan(full)
    _EXTRA_PLANS[width] = (names, permutation, sign)
    return width


def keep_columns(drop_rules=False, width=None):
    """Column indices a model reads; all of them unless drop_rules (then NAMES minus RULE_COLUMNS, in order).

    width: the matrix width when it is wider than NAMES (extra columns come after the base ones and are kept).
    """
    if width is None or width == len(NAMES):
        if not drop_rules:
            return np.arange(len(NAMES))
        return np.array([i for i, name in enumerate(NAMES) if name not in RULE_COLUMNS])
    if not drop_rules:
        return np.arange(width)
    return np.array([i for i in range(width) if i >= len(NAMES) or NAMES[i] not in RULE_COLUMNS])


def swap_teams(features):
    """Team-swapped copy of a (rows, len(NAMES)) matrix, or of a registered extended matrix (by its width)."""
    width = features.shape[1]
    if width == len(NAMES):
        return features[:, PERMUTATION] * SIGN
    if width not in _EXTRA_PLANS:
        from mtah.external import extra_features          # noqa: F401  (registers the known layouts)
    if width not in _EXTRA_PLANS:
        raise ValueError("no swap plan for a feature matrix of width {}".format(width))
    _, permutation, sign = _EXTRA_PLANS[width]
    return features[:, permutation] * sign


@dataclass
class FeatureSet:
    x: np.ndarray             # (rows, len(NAMES)) float32
    frame: np.ndarray         # (rows,) the frame each row sits on (the middle of its block)
    block: np.ndarray         # (rows, 2) half-open frame span the row covers
    runs: list                # [(first_row, last_row_exclusive)] per half


def sample_frames(start, stop):
    """(frames, blocks): the middle frame of each block of STEP frames in [start, stop)."""
    firsts = np.arange(start, stop, STEP)
    lasts = np.minimum(firsts + STEP, stop)
    frames = np.minimum(firsts + STEP // 2, stop - 1)
    return frames, np.column_stack([firsts, lasts])


def _fill_ball(bx, by, seen):
    """(estimated x, estimated y, last, next): see the module docstring."""
    t = len(bx)
    index = np.arange(t)
    last = np.maximum.accumulate(np.where(seen, index, -1))
    nxt = np.minimum.accumulate(np.where(seen, index, t)[::-1])[::-1]
    has_last, has_next = last >= 0, nxt < t
    safe_last, safe_next = np.clip(last, 0, t - 1), np.clip(nxt, 0, t - 1)
    span = np.maximum(nxt - last, 1)
    weight = (index - last) / span
    bridge = has_last & has_next & (nxt - last <= INTERP_S * FPS)
    out = []
    for values in (bx, by):
        est = np.full(t, np.nan)
        est[has_next] = values[safe_next][has_next]
        est[has_last] = values[safe_last][has_last]
        est[bridge] = (values[safe_last] * (1 - weight) + values[safe_next] * weight)[bridge]
        est[seen] = values[seen]
        out.append(est)
    return out[0], out[1], last, nxt


def _window_mean(series, half_width):
    """Centred mean over 2 * half_width + 1 steps, NaN-aware (NaN where nothing is valid)."""
    valid = ~np.isnan(series)
    values = np.where(valid, series, 0.0)
    cum_v = np.concatenate([[0.0], np.cumsum(values)])
    cum_n = np.concatenate([[0.0], np.cumsum(valid)])
    n = len(series)
    lo = np.clip(np.arange(n) - half_width, 0, n)
    hi = np.clip(np.arange(n) + half_width + 1, 0, n)
    count = cum_n[hi] - cum_n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (cum_v[hi] - cum_v[lo]) / np.maximum(count, 1), np.nan)


def attack_sign(home_x, kickoff, length):
    """+1 if home attacks towards +x (its kickoff formation stands in the low-x half), else -1."""
    centre = np.nanmean(home_x[kickoff:kickoff + KICKOFF_FRAMES])
    return 1.0 if np.isnan(centre) or centre < 0 else -1.0


def _run_features(tracking, start, stop, kickoff, rule_codes, rule_pdead, frame_teams=None):
    length, width = tracking.pitch
    half_l, half_w = length / 2.0, width / 2.0
    players = tracking.players[start:stop]
    px = (players[..., 0] - 0.5) * length
    py = (players[..., 1] - 0.5) * width
    speed = players[..., 2]
    ball = tracking.ball[start:stop]
    bx_raw = (ball[:, 0] - 0.5) * length
    by_raw = (ball[:, 1] - 0.5) * width
    bs_raw = ball_speed_ms(bx_raw.astype(np.float64), by_raw.astype(np.float64))   # from positions, never the file's column
    seen = ~np.isnan(bx_raw)
    t = stop - start
    teams = np.asarray(tracking.team_of_slot)
    # The attacking direction is read off the fixed slot teams even when frame_teams is given: the
    # kickoff formation (22 players over 2 s) is not moved by a few swapped slots.
    sign = attack_sign(_nanmean_rows(px[:, teams == 0]), kickoff - start, length)

    bx, by, last, nxt = _fill_ball(bx_raw, by_raw, seen)
    frames, blocks = sample_frames(0, t)
    s = frames
    rows = len(s)
    out = {}

    out["ball_seen"] = seen[s].astype(np.float64)
    out["since_last"] = np.where(last[s] >= 0, np.minimum((s - last[s]) / FPS, CAP_S), CAP_S)
    out["until_next"] = np.where(nxt[s] < t, np.minimum((nxt[s] - s) / FPS, CAP_S), CAP_S)
    out["ball_x_att"] = sign * bx[s] / half_l
    out["ball_x_abs"] = np.abs(bx[s]) / half_l
    out["ball_y_abs"] = np.abs(by[s]) / half_w
    out["boundary"] = np.minimum(half_l - np.abs(bx[s]), half_w - np.abs(by[s]))
    out["ball_speed"] = bs_raw[s]
    before = np.clip(s - STEP, 0, t - 1)
    out["ball_accel"] = np.abs(bs_raw[s] - bs_raw[before])
    out["ball_x_last_att"] = np.where(last[s] >= 0, sign * bx_raw[np.clip(last[s], 0, t - 1)] / half_l, np.nan)
    out["ball_x_next_att"] = np.where(nxt[s] < t, sign * bx_raw[np.clip(nxt[s], 0, t - 1)] / half_l, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for side, team in (("home", 0), ("away", 1)):
            if frame_teams is None:
                mask = teams == team
                tx, ty, ts = px[:, mask], py[:, mask], speed[:, mask]
            else:
                # per frame: a slot counts for the side it belongs to on that frame (shirt colour)
                inside = frame_teams[start:stop] == team
                tx, ty, ts = (np.where(inside, v, np.nan) for v in (px, py, speed))
            dist = np.hypot(tx - bx[:, None], ty - by[:, None])
            d1_full = np.nanmin(np.where(np.isnan(dist), np.inf, dist), axis=1)
            d1_full[np.isinf(d1_full)] = np.nan
            ordered = np.sort(dist[s], axis=1)
            width_cols = ordered.shape[1]
            for k, name in enumerate(("d1", "d2", "d3")):
                out["{}_{}".format(side, name)] = ordered[:, k] if width_cols > k else np.full(rows, np.nan)
            for radius, name in zip(NEAR_RADII, ("c2", "c5", "c10")):
                out["{}_{}".format(side, name)] = (dist[s] < radius).sum(axis=1).astype(np.float64)
            tracked = ~np.isnan(tx[s])
            n_seen = tracked.sum(axis=1).astype(np.float64)
            out[side + "_n_seen"] = n_seen
            sp = ts[s]
            out[side + "_speed_med"] = np.nanmedian(sp, axis=1)
            out[side + "_speed_p75"] = np.nanpercentile(sp, 75, axis=1)
            known = ~np.isnan(sp)
            out[side + "_slow_frac"] = np.where(known.any(axis=1), ((sp < SLOW) & known).sum(axis=1)
                                                / np.maximum(known.sum(axis=1), 1), np.nan)
            out[side + "_spread_x"] = np.nanstd(tx[s], axis=1)
            out[side + "_spread_y"] = np.nanstd(ty[s], axis=1)
            cx, cy = np.nanmean(tx[s], axis=1), np.nanmean(ty[s], axis=1)
            out[side + "_cdist"] = np.hypot(cx - bx[s], cy - by[s])
            out[side + "_cxrel"] = sign * (cx - bx[s]) / half_l
            out[side + "_d1_prev"] = np.where(last[s] >= 0, d1_full[np.clip(last[s], 0, t - 1)], np.nan)
            out[side + "_d1_next"] = np.where(nxt[s] < t, d1_full[np.clip(nxt[s], 0, t - 1)], np.nan)

    for w in WINDOWS_S:
        half_width = max(1, int(round(w * HZ / 2)))
        for name in BALL_WINDOWED:
            out["{}_w{}".format(name, w)] = _window_mean(out[name], half_width)
        for side in ("home", "away"):
            for name in TEAM_WINDOWED:
                out["{}_{}_w{}".format(side, name, w)] = _window_mean(out[side + "_" + name], half_width)

    codes = np.asarray(rule_codes)[start:stop][s]
    out["rule_H"] = (codes == 0).astype(np.float64)
    out["rule_A"] = (codes == 1).astype(np.float64)
    out["rule_D"] = (codes == 2).astype(np.float64)
    out["rule_pdead"] = np.asarray(rule_pdead, dtype=np.float64)[start:stop][s]

    matrix = np.column_stack([out[name] for name in NAMES]).astype(np.float32)
    return matrix, start + frames, blocks + start


def _nanmean_rows(values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=1)


def compute(tracking, runs, kickoffs, rule_codes, rule_pdead, frame_teams=None):
    """FeatureSet for the given halves. rule_codes / rule_pdead: our rules' per-frame call and P(dead).

    frame_teams: optional (n_frames, n_slots) array of 0 / 1 teams per frame (shirt-colour teams);
    None uses the fixed team_of_slot, exactly as before.
    """
    if frame_teams is not None:
        frame_teams = np.asarray(frame_teams)
    matrices, frames, blocks, spans, at = [], [], [], [], 0
    for (start, stop), kickoff in zip(runs, kickoffs):
        matrix, frame, block = _run_features(tracking, start, stop, kickoff, rule_codes, rule_pdead, frame_teams)
        matrices.append(matrix)
        frames.append(frame)
        blocks.append(block)
        spans.append((at, at + len(matrix)))
        at += len(matrix)
    return FeatureSet(x=np.concatenate(matrices), frame=np.concatenate(frames),
                      block=np.concatenate(blocks), runs=spans)


def expand(values, feature_set, n_frames, fill=2):
    """Per-row values (rows,) back to one per frame (n_frames,): each row covers its block."""
    out = np.full(n_frames, fill, dtype=np.asarray(values).dtype)
    for row, (a, b) in enumerate(feature_set.block):
        out[a:b] = values[row]
    return out
