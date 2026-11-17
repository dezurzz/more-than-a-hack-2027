"""MS-TCN stage -- per-frame features at team level, so nothing depends on a slot's identity.

The ATD has no player ids and its slots are tracks that come and go, so every feature is a
statistic of a team (or of the ball), never of slot k. The same function runs on damaged DFL
matches (training, validation) and on the ATD (the final run only), and needs nothing but a
Tracking, the halves and a Config.

Smart Tagging is not an input: the external matches have none. The rule-based labels are, as
three one-hot channels -- they carry the ATD-specific knowledge (gap rule, ball over a line,
restart layouts) the model cannot learn from DFL.

Columns are named in FEATURES; team-specific ones come in (home, away) pairs, listed in
TEAM_PAIRS, so augmentation can swap the teams by swapping columns.

The touch group (appended last) describes control rather than nearness, because the error
breakdown showed the model losing possessions won in live play: a kink in the ball's path, a
duel, and per team how long it has kept the ball, how long since it last touched it and last
touched it under control (the rules' own Bagian 3 detector), and how fast it is closing on it.
Models trained before the group existed read only the first N_BASE columns (tcn.Predictor).
"""

import numpy as np

from pipeline.config import A, H, Config
from pipeline.possession import DUEL, control_frames, find_touches, is_controlled
from pipeline.labels import label_match
from pipeline.preprocess import preprocess_ball

K_NEAREST = 3
NEAR_RADII = (3.0, 8.0)       # metres: players of each team within these of the ball
SINCE_CAP = 10 * 25           # frames: "time since the ball was last seen" saturates here
FAR = 1.0                     # distance feature (pitch lengths) when the ball is not known
HOLD_CAP = 3 * 25             # frames: "kept the ball for" saturates at 3 s
APPROACH_STEP = 5             # frames over which the closing speed on the ball is measured

_TEAM_STATS = (["dist{}".format(k + 1) for k in range(K_NEAREST)]
               + ["near{:g}".format(r) for r in NEAR_RADII]
               + ["cx", "cy", "sx", "sy", "speed", "speed_max", "seen", "ahead_of_ball"])
_TOUCH_STATS = ["hold", "touch_since", "control_since", "approach"]
FEATURES = (["ball_x", "ball_y", "ball_speed", "ball_seen", "ball_since", "ball_out",
             "last_x", "last_y", "all_speed", "rule_H", "rule_A", "rule_D", "in_half"]
            + ["{}_{}".format(side, name) for name in _TEAM_STATS for side in ("h", "a")])
N_BASE = len(FEATURES)        # what tcn1 / tcn2 were trained on
FEATURES += (["ball_kink", "duel"]
             + ["{}_{}".format(side, name) for name in _TOUCH_STATS for side in ("h", "a")])
TEAM_PAIRS = [(FEATURES.index("h_" + name), FEATURES.index("a_" + name))
              for name in _TEAM_STATS + _TOUCH_STATS]
# Columns that change sign when the pitch is mirrored along its length / its width.
MIRROR_X = [FEATURES.index(name) for name in ("ball_x", "last_x", "h_cx", "a_cx")]
MIRROR_Y = [FEATURES.index(name) for name in ("ball_y", "last_y", "h_cy", "a_cy")]


def _frames_since_seen(seen):
    """Frames since seen was last True (0 where it is True), capped at SINCE_CAP."""
    index = np.arange(len(seen))
    last = np.maximum.accumulate(np.where(seen, index, -SINCE_CAP))
    return np.minimum(index - last, SINCE_CAP)


def _forward_fill(xy):
    """Last seen position at every frame; NaN before the first sighting."""
    seen = ~np.isnan(xy[:, 0])
    carry = np.maximum.accumulate(np.where(seen, np.arange(len(xy)), -1))
    out = np.full_like(xy, np.nan)
    ok = carry >= 0
    out[ok] = xy[carry[ok]]
    return out


def _nan_stat(function, values, axis, fill):
    """function over axis ignoring NaN, fill where every value is NaN, without the warnings."""
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = function(values, axis=axis)
    return np.where(np.isnan(out), fill, out)


def _run_length(mask, cap):
    """Frames mask has been True without a break, up to and including each frame, capped."""
    index = np.arange(len(mask))
    last_false = np.maximum.accumulate(np.where(mask, -1, index))
    return np.minimum(np.where(mask, index - last_false, 0), cap)


def _touch_features(tracking, ball, motion, nearest_m, cfg):
    """{name: column} for the touch group. nearest_m: {side: nearest player distance (m)}."""
    out = {}
    kink = np.hypot(*(motion.v_out - motion.v_in).T)
    out["ball_kink"] = np.nan_to_num(np.minimum(kink, 20.0) / 10.0)
    status, near_team, _ = control_frames(tracking.players, ball, tracking.team, cfg)
    out["duel"] = (status == DUEL).astype(np.float32)
    touches = find_touches(near_team, motion, cfg)
    n = len(ball)
    for side, code in (("h", H), ("a", A)):
        out[side + "_hold"] = _run_length(near_team == code, HOLD_CAP) / float(HOLD_CAP)
        touching = np.zeros(n, dtype=bool)
        controlling = np.zeros(n, dtype=bool)
        for i, touch in enumerate(touches):
            if touch.team == code:
                touching[touch.start:touch.stop] = True
                if is_controlled(touches, i, cfg):
                    controlling[touch.start:touch.stop] = True
        out[side + "_touch_since"] = _frames_since_seen(touching) / float(SINCE_CAP)
        out[side + "_control_since"] = _frames_since_seen(controlling) / float(SINCE_CAP)
        d = nearest_m[side]
        closing = np.zeros(n, dtype=np.float32)
        step = APPROACH_STEP
        with np.errstate(invalid="ignore"):  # inf - inf where the ball is unknown -> 0 below
            closing[step:] = (d[:-step] - d[step:]) * cfg.FPS / step
        out[side + "_approach"] = np.clip(np.nan_to_num(closing, nan=0.0, posinf=0.0,
                                                        neginf=0.0), -10.0, 10.0) / 10.0
    return out


def compute(tracking, halves, cfg=Config(), rule_labels=None):
    """(n, len(FEATURES)) float32. Positions are centred and scaled to [-1, 1] by the pitch.

    rule_labels is label_match(...).labels if already at hand; computed here otherwise.
    """
    n = len(tracking)
    length, width = cfg.LAPANGAN_P, cfg.LAPANGAN_L
    scale = np.array([length / 2, width / 2], dtype=np.float32)
    centre = scale

    ball, motion = preprocess_ball(tracking.ball, cfg)
    seen = ~np.isnan(ball[:, 0])
    last = _forward_fill(ball)
    known = np.where(seen[:, None], ball, last)          # best guess of where the ball is
    has_guess = ~np.isnan(known[:, 0])

    out = {}
    out["ball_x"], out["ball_y"] = (np.nan_to_num((ball - centre) / scale)).T
    out["ball_speed"] = np.nan_to_num(np.minimum(motion.speed, 40.0) / 10.0)
    out["ball_seen"] = seen.astype(np.float32)
    out["ball_since"] = _frames_since_seen(seen) / float(SINCE_CAP)
    margin = cfg.MARGIN_LUAR
    out["ball_out"] = (seen & ((ball[:, 0] < -margin) | (ball[:, 0] > length + margin)
                               | (ball[:, 1] < -margin) | (ball[:, 1] > width + margin))
                       ).astype(np.float32)
    out["last_x"], out["last_y"] = (np.nan_to_num((last - centre) / scale)).T
    out["all_speed"] = _nan_stat(np.nanmean, tracking.speeds, 1, 0.0) / 5.0

    if rule_labels is None:
        rule_labels = label_match(tracking, [], halves, cfg).labels
    for letter in "HAD":
        out["rule_" + letter] = (rule_labels == letter).astype(np.float32)
    in_half = np.zeros(n, dtype=np.float32)
    for _, first, last_frame in halves:
        in_half[first:last_frame + 1] = 1.0
    out["in_half"] = in_half

    players = tracking.players
    nearest_m = {}
    distance = np.hypot(players[:, :, 0] - known[:, None, 0],
                        players[:, :, 1] - known[:, None, 1]) / length
    distance[~has_guess] = np.nan
    for side, code in (("h", H), ("a", A)):
        mine = tracking.team == code
        xy = players[:, mine]
        d = np.where(np.isnan(distance[:, mine]), np.inf, distance[:, mine])
        nearest = np.sort(d, axis=1)
        nearest_m[side] = (nearest[:, 0] * length if nearest.shape[1]
                           else np.full(n, np.inf)).astype(np.float64)
        for k in range(K_NEAREST):
            column = nearest[:, k] if k < nearest.shape[1] else np.full(n, np.inf)
            out["{}_dist{}".format(side, k + 1)] = np.where(np.isfinite(column),
                                                            np.minimum(column, FAR), FAR)
        for radius in NEAR_RADII:
            out["{}_near{:g}".format(side, radius)] = (d * length <= radius).sum(1) / 5.0
        cx = _nan_stat(np.nanmean, xy[:, :, 0], 1, length / 2)
        cy = _nan_stat(np.nanmean, xy[:, :, 1], 1, width / 2)
        out[side + "_cx"] = (cx - length / 2) / (length / 2)
        out[side + "_cy"] = (cy - width / 2) / (width / 2)
        out[side + "_sx"] = _nan_stat(np.nanstd, xy[:, :, 0], 1, 0.0) / (length / 2)
        out[side + "_sy"] = _nan_stat(np.nanstd, xy[:, :, 1], 1, 0.0) / (width / 2)
        speed = tracking.speeds[:, mine]
        out[side + "_speed"] = _nan_stat(np.nanmean, speed, 1, 0.0) / 5.0
        out[side + "_speed_max"] = _nan_stat(np.nanmax, speed, 1, 0.0) / 10.0
        out[side + "_seen"] = (~np.isnan(xy[:, :, 0])).sum(1) / 11.0
        # Share of the team closer to the right-hand goal than the ball is: a sign-free hint of
        # which way it is facing relative to the ball, without knowing the attacking direction.
        ahead = (xy[:, :, 0] > known[:, None, 0]) & has_guess[:, None]
        counted = np.maximum(1, (~np.isnan(xy[:, :, 0])).sum(1))
        out[side + "_ahead_of_ball"] = ahead.sum(1) / counted

    out.update(_touch_features(tracking, ball, motion, nearest_m, cfg))
    table = np.stack([np.asarray(out[name], dtype=np.float32) for name in FEATURES], axis=1)
    return np.nan_to_num(table, nan=0.0, posinf=0.0, neginf=0.0)


def swap_teams(table):
    """Same frames with home and away exchanged (augmentation). Labels must swap H<->A too."""
    out = table.copy()
    for h, a in TEAM_PAIRS:
        out[:, [h, a]] = table[:, [a, h]]
    rule_h, rule_a = FEATURES.index("rule_H"), FEATURES.index("rule_A")
    out[:, [rule_h, rule_a]] = table[:, [rule_a, rule_h]]
    return out


def mirror(table, along_x=True, along_y=False):
    """Same frames on a mirrored pitch (augmentation); labels are unchanged.

    ahead_of_ball counts the side towards +x, so mirroring x turns it into its complement.
    """
    out = table.copy()
    if along_x:
        out[:, MIRROR_X] *= -1
        for side in ("h", "a"):
            column = FEATURES.index(side + "_ahead_of_ball")
            seen = table[:, FEATURES.index(side + "_seen")] > 0
            out[:, column] = np.where(seen, 1.0 - table[:, column], 0.0)
    if along_y:
        out[:, MIRROR_Y] *= -1
    return out
