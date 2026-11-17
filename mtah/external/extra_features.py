"""Optional extra feature sets appended to the 92 base columns.

    smooth  ball speed / acceleration from a gap-safe smoothed ball track (B1)
    touch   ball direction-change / speed-jump events and the nearest players at them (B2)
    near5   4th and 5th nearest player per team (B3)
    deadcue dead-ball cues without team or direction, alone only

A set name may be a "+"-joined combination ("smooth+touch"): the concatenation in that order. Rows are
aligned exactly with features.compute (same sample_frames per half). No new column is signed; the
home_ / away_ columns exchange under the team swap (features.register_extra). Default pipelines never
call this module, so the base features and caches are untouched.
"""

import itertools
import warnings

import numpy as np

from mtah.external import features
from mtah.external.convert import ball_speed_ms

EXTRA_VERSION = 1

SETS = {
    "smooth": ["ball_speed_sm", "ball_accel_sm", "ball_speed_sm_w1", "ball_speed_sm_w4", "ball_speed_sm_w12"],
    "touch": ["ball_turn", "ball_jump", "ball_event", "home_touch_d", "away_touch_d",
              "home_touch_age", "away_touch_age"],
    "near5": ["home_d4", "home_d5", "away_d4", "away_d5"],
    # touch without any event threshold: same code as "touch"
    "touchc": ["ball_turn", "ball_jump"],
    # Plan: 7 columns; a 7-wide set would share width 99 with "touch" (swap plans are looked up
    # by width), so an 8th column, still_run_s, was added. No home_ / away_ name, nothing signed: the swap is the identity.
    "deadcue": ["ball_still", "ball_still_w1", "ball_still_w4", "ring_n", "close_n", "speed_drop", "still_at_line",
                "still_run_s"],
}
ALONE_ONLY = ("touchc", "deadcue")      # shares column names with "touch": usable on its own, never in a combination

SMOOTH_HALF = 2               # centred mean over +-2 frames (0.2 s at 25 Hz)
TOUCH_LAG = 3                 # velocity over 3 frames
TOUCH_DT = TOUCH_LAG / features.FPS
TURN_DEG, RATIO, ACC = 12.5, 1.5, 4.0
MIN_SPEED, RATIO_FLOOR = 1.0, 0.5
RATIO_LOG_FLOOR = 0.01       # a still ball gives ratio 0; log(0) would be -inf
NEAR_M = 2.0
STILL_MS = 0.5               # deadcue: smoothed ball speed under this is "still"
RING_M = (2.0, 9.0)          # deadcue: players 2-9 m from the ball
LINE_M = 2.0                 # deadcue: ball within this of a touch or goal line
DROP_S = 3.0                 # deadcue: speed_drop compares with this long ago


def components(set_name):
    parts = set_name.split("+")
    for part in parts:
        if part not in SETS:
            raise ValueError("unknown extra set {!r}; known: {}".format(part, ", ".join(SETS)))
        if part in ALONE_ONLY and len(parts) > 1:
            raise ValueError("extra set {!r} cannot be combined".format(part))
    return parts


def names(set_name):
    return [n for part in components(set_name) for n in SETS[part]]


# Register every combination in canonical order, so swap_teams works in any process that imports this
# module (features.swap_teams imports it on demand). Widths 5, 7, 4, 12, 9, 11, 16 (and 2 for touchc, 8 for deadcue) are all different.
_COMBINABLE = [name for name in SETS if name not in ALONE_ONLY]
for _r in range(1, len(_COMBINABLE) + 1):
    for _combo in itertools.combinations(_COMBINABLE, _r):
        features.register_extra([n for part in _combo for n in SETS[part]])
for _name in ALONE_ONLY:
    features.register_extra(SETS[_name])


# -- building blocks ----------------------------------------------------------------------------------

def smooth_track(x, y, seen, half=SMOOTH_HALF):
    """Centred mean over +-half frames inside each contiguous run of observed frames; NaN elsewhere."""
    t = len(x)
    index = np.arange(t)
    previous = np.concatenate([[False], seen[:-1]])
    following = np.concatenate([seen[1:], [False]])
    run_start = np.maximum.accumulate(np.where(seen & ~previous, index, -1))
    run_end = np.minimum.accumulate(np.where(seen & ~following, index, t)[::-1])[::-1]
    lo = np.maximum(index - half, run_start)
    hi = np.minimum(index + half, run_end)
    out = []
    for values in (x, y):
        filled = np.where(seen, values, 0.0)
        cum = np.concatenate([[0.0], np.cumsum(filled)])
        lo_c = np.clip(lo, 0, t)
        hi_c = np.clip(hi + 1, 0, t)
        count = np.maximum(hi_c - lo_c, 1)
        out.append(np.where(seen, (cum[hi_c] - cum[lo_c]) / count, np.nan))
    return out[0], out[1]


def _blocks(values, rows, fill):
    """(rows, STEP) view of a per-frame series, padded with `fill` after the last frame."""
    padded = np.full(rows * features.STEP, fill, dtype=np.float64)
    padded[:len(values)] = values
    return padded.reshape(rows, features.STEP)


def _team_arrays(tracking, start, stop, frame_teams):
    length, width = tracking.pitch
    players = tracking.players[start:stop]
    px = (players[..., 0] - 0.5) * length
    py = (players[..., 1] - 0.5) * width
    teams = np.asarray(tracking.team_of_slot)
    out = {}
    for side, team in (("home", 0), ("away", 1)):
        if frame_teams is None:
            mask = teams == team
            out[side] = (px[:, mask], py[:, mask])
        else:
            inside = frame_teams[start:stop] == team
            out[side] = tuple(np.where(inside, v, np.nan) for v in (px, py))
    return out


def _nearest(tx, ty, bx, by):
    """(t,) nearest player distance to the ball per frame; NaN if none."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dist = np.hypot(tx - bx[:, None], ty - by[:, None])
        d1 = np.nanmin(np.where(np.isnan(dist), np.inf, dist), axis=1) if dist.shape[1] else np.full(len(bx), np.nan)
    d1 = np.asarray(d1, dtype=np.float64)
    d1[np.isinf(d1)] = np.nan
    return d1


# -- the sets -----------------------------------------------------------------------------------------

def _smooth(ball_m, seen, frames, t):
    sx, sy = smooth_track(ball_m[0], ball_m[1], seen)
    speed = ball_speed_ms(sx.astype(np.float64), sy.astype(np.float64)).astype(np.float64)
    s = frames
    before = np.clip(s - features.STEP, 0, t - 1)
    row_speed = speed[s]
    out = {"ball_speed_sm": row_speed, "ball_accel_sm": np.abs(row_speed - speed[before]) / 0.2}
    for w in features.WINDOWS_S:
        half_width = max(1, int(round(w * features.HZ / 2)))
        out["ball_speed_sm_w{}".format(w)] = features._window_mean(row_speed, half_width)
    return out


def _touch(ball_m, seen, frames, t, players):
    rows = len(frames)
    sx, sy = smooth_track(ball_m[0], ball_m[1], seen)
    lag = TOUCH_LAG
    out = {}
    nan = np.full(t, np.nan)
    vx, vy = nan.copy(), nan.copy()
    if t > lag:
        cum = np.concatenate([[0], np.cumsum(seen)])
        full = (cum[lag + 1:] - cum[:t - lag]) == lag + 1            # frames f-lag..f all observed, for f >= lag
        f = np.arange(lag, t)[full]
        vx[f] = (sx[f] - sx[f - lag]) / TOUCH_DT
        vy[f] = (sy[f] - sy[f - lag]) / TOUCH_DT
    ix, iy = vx.copy(), vy.copy()                                    # v_in(f) = v(f)
    ox, oy = nan.copy(), nan.copy()                                  # v_out(f) = v(f + lag)
    if t > lag:
        ox[:t - lag], oy[:t - lag] = vx[lag:], vy[lag:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sin, sout = np.hypot(ix, iy), np.hypot(ox, oy)
        pair = ~np.isnan(sin) & ~np.isnan(sout)
        moving = pair & (sin >= MIN_SPEED) & (sout >= MIN_SPEED)
        cosine = (ix * ox + iy * oy) / np.where(moving, sin * sout, 1.0)
        turn = np.where(moving, np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))), np.nan)
        ratio = np.where(pair, np.maximum(sin, sout) / np.maximum(np.minimum(sin, sout), RATIO_FLOOR), np.nan)
        acc = np.where(pair, np.abs(sout - sin) / TOUCH_DT, np.nan)
        fires = (np.nan_to_num(turn, nan=-1) >= TURN_DEG) | (np.nan_to_num(ratio, nan=-1) >= RATIO) \
            | (np.nan_to_num(acc, nan=-1) >= ACC)
        strength = np.fmax.reduce([np.nan_to_num(turn / TURN_DEG, nan=-np.inf),
                                   np.nan_to_num(ratio / RATIO, nan=-np.inf),
                                   np.nan_to_num(acc / ACC, nan=-np.inf)])
    event = fires & pair

    turn_b = np.nan_to_num(_blocks(turn, rows, np.nan), nan=-np.inf).max(axis=1)
    out["ball_turn"] = np.where(np.isinf(turn_b), np.nan, turn_b / 180.0)
    ratio_b = np.nan_to_num(_blocks(ratio, rows, np.nan), nan=-np.inf).max(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["ball_jump"] = np.where(np.isinf(ratio_b), np.nan, np.log(np.where(np.isinf(ratio_b), 1.0, np.maximum(ratio_b, RATIO_LOG_FLOOR))))
    any_event = _blocks(event.astype(np.float64), rows, 0.0).max(axis=1) > 0
    any_pair = _blocks(pair.astype(np.float64), rows, 0.0).max(axis=1) > 0
    out["ball_event"] = np.where(any_event, 1.0, np.where(any_pair, 0.0, np.nan))

    # the strongest event frame of each block
    strong = np.where(event, strength, -np.inf)
    sb = _blocks(strong, rows, -np.inf)
    pick = sb.argmax(axis=1)
    event_frame = np.arange(rows) * features.STEP + pick
    event_frame = np.where(any_event, np.minimum(event_frame, t - 1), 0)

    bxs, bys = sx, sy
    for side in ("home", "away"):
        tx, ty = players[side]
        d1 = _nearest(tx, ty, bxs, bys)
        out[side + "_touch_d"] = np.where(any_event, d1[event_frame], np.nan)
        flag = event & (np.nan_to_num(d1, nan=np.inf) <= NEAR_M)
        last = np.maximum.accumulate(np.where(flag, np.arange(t), -1))
        age = np.where(last[frames] >= 0, np.minimum((frames - last[frames]) / features.FPS, features.CAP_S),
                       features.CAP_S)
        out[side + "_touch_age"] = age
    return out


def _near5(ball_m, seen, frames, t, players):
    bx, by, _, _ = features._fill_ball(ball_m[0], ball_m[1], seen)
    rows = len(frames)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for side in ("home", "away"):
            tx, ty = players[side]
            dist = np.hypot(tx - bx[:, None], ty - by[:, None])
            ordered = np.sort(dist[frames], axis=1)
            for k, name in ((3, "d4"), (4, "d5")):
                out["{}_{}".format(side, name)] = ordered[:, k] if ordered.shape[1] > k else np.full(rows, np.nan)
    return out


def _deadcue(ball_m, seen, frames, t, players, pitch, speeds):
    """The 8 deadcue columns; `speeds` (t, slots) player speeds of the run (team-free), `players` as _team_arrays."""
    length, width = pitch
    sx, sy = smooth_track(ball_m[0], ball_m[1], seen)
    speed = ball_speed_ms(sx.astype(np.float64), sy.astype(np.float64)).astype(np.float64)
    row_speed = speed[frames]
    still = np.where(~np.isnan(row_speed) & (row_speed < STILL_MS), 1.0, 0.0)       # NaN (unseen / edge) counts as not still
    out = {"ball_still": still}
    for w in (1, 4):
        out["ball_still_w{}".format(w)] = features._window_mean(still, max(1, int(round(w * features.HZ / 2))))
    bx, by, _, _ = features._fill_ball(ball_m[0], ball_m[1], seen)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        dist = np.concatenate([np.hypot(players[side][0][frames] - bx[frames][:, None],
                                        players[side][1][frames] - by[frames][:, None]) for side in ("home", "away")], axis=1)
        out["ring_n"] = ((dist >= RING_M[0]) & (dist < RING_M[1])).sum(axis=1).astype(np.float64)
        out["close_n"] = (dist < RING_M[0]).sum(axis=1).astype(np.float64)
        median = np.nanmedian(np.asarray(speeds, dtype=np.float64)[frames], axis=1)      # both teams pooled
        lag = int(round(DROP_S * features.HZ))
        drop = np.full(len(frames), np.nan)
        drop[lag:] = median[lag:] - median[:-lag]
        out["speed_drop"] = drop
        edge = np.minimum(length / 2.0 - np.abs(sx[frames]), width / 2.0 - np.abs(sy[frames]))
        out["still_at_line"] = np.where((still == 1.0) & (edge <= LINE_M), 1.0, 0.0)
    run = np.zeros(len(frames))                                                         # seconds the ball has stayed still
    for i in range(len(frames)):
        run[i] = min((run[i - 1] if i else 0.0) + 1.0 / features.HZ, features.CAP_S) if still[i] else 0.0
    out["still_run_s"] = run
    return out


def _run_extra(tracking, start, stop, parts, frame_teams):
    length, width = tracking.pitch
    ball = tracking.ball[start:stop]
    bx = ((ball[:, 0] - 0.5) * length).astype(np.float64)
    by = ((ball[:, 1] - 0.5) * width).astype(np.float64)
    seen = ~np.isnan(bx)
    t = stop - start
    frames, _ = features.sample_frames(0, t)
    players = _team_arrays(tracking, start, stop, frame_teams)
    columns = []
    for part in parts:
        if part == "smooth":
            out = _smooth((bx, by), seen, frames, t)
        elif part == "deadcue":
            out = _deadcue((bx, by), seen, frames, t, players, tracking.pitch, tracking.players[start:stop, :, 2])
        elif part in ("touch", "touchc"):
            out = _touch((bx, by), seen, frames, t, players)
        else:
            out = _near5((bx, by), seen, frames, t, players)
        columns += [out[name] for name in SETS[part]]
    return np.column_stack(columns).astype(np.float32)


def compute_extra(tracking, runs, kickoffs, set_name, frame_teams=None):
    """(rows, k) float32: the extra columns of `set_name`, rows aligned with features.compute(...)."""
    parts = components(set_name)
    if frame_teams is not None:
        frame_teams = np.asarray(frame_teams)
    return np.concatenate([_run_extra(tracking, start, stop, parts, frame_teams) for start, stop in runs])
