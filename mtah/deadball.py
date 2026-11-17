"""P(ball dead) per frame from bounded rule indicators.

    logit P(dead) = DEAD_BIAS + sum_k DEAD_WEIGHTS[k] * r_k,   each r_k in [0, 1]

Rules, not a learned model: the public window has four stoppages, nothing to train on. A missing
feature counts as "this rule does not fire", so the result is never NaN; a frame with no usable
feature at all gets sigmoid(DEAD_BIAS), the prior.
"""

import numpy as np

from mtah import config, features


def _get(feats, name):
    """The feature, or all-NaN if this run did not build it (e.g. no Smart Tagging)."""
    return feats[name] if name in feats else np.full(len(feats["ball_visible"]), np.nan)


def slow_play(feats):
    """1 where the players are walking, 0 where they are running, linear in between."""
    low, high = config.SLOW_PLAY_RAMP
    ramp = np.clip((high - _get(feats, "all_speed_file")) / (high - low), 0.0, 1.0)
    return np.nan_to_num(ramp, nan=0.0)


def ball_exit(feats, horizon=None, min_gap=None, cover=None, fit=None, margin=None,
              goal_half_width=None, fps=25.0):
    """1 on the frames of a ball gap whose last tracked motion leads out of the pitch, else 0.

    The ball went out and then vanished from the tracking, so the "out" rule never saw it outside
    Uses only the ball features: the distance to the nearer goal line and
    touch line, and on which side of the pitch the ball is.

    goal_half_width: an exit over a goal line within this many metres of the pitch's centre line
    (lengthwise) does not count: a keeper catching the ball in front of his goal makes it vanish
    the same way. None = every line counts.
    """
    horizon = config.BALL_EXIT_HORIZON if horizon is None else horizon
    min_gap = config.BALL_EXIT_MIN_GAP if min_gap is None else min_gap
    cover = config.BALL_EXIT_COVER if cover is None else cover
    fit = config.BALL_EXIT_FIT if fit is None else fit
    margin = config.OUT_MARGIN if margin is None else margin
    goal_half_width = config.BALL_EXIT_GOAL_HALF_WIDTH if goal_half_width is None else goal_half_width

    visible = np.nan_to_num(_get(feats, "ball_visible"), nan=0.0) > 0
    x, y = _get(feats, "ball_x"), _get(feats, "ball_y")
    goal, touch = _get(feats, "ball_goal_line_dist"), _get(feats, "ball_touch_line_dist")
    # Pitch width from the ball features: y + distance to the far touch line, where y is past half.
    far = visible & ~np.isclose(y, touch)
    width = float(np.median(y[far] + touch[far])) if far.any() else np.nan
    out = np.zeros(len(visible))
    edges = np.diff(np.concatenate(([0], (~visible).astype(np.int8), [0])))
    for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if hi - lo < min_gap or lo < fit or not visible[lo - fit:lo].all():
            continue
        last = lo - 1
        steps = (fit - 1) / fps
        vx, vy = (x[last] - x[lo - fit]) / steps, (y[last] - y[lo - fit]) / steps
        # The nearer line is at 0 when the coordinate equals its distance, else at the far side.
        to_goal = goal[last] + (vx if np.isclose(x[last], goal[last]) else -vx) * horizon
        to_touch = touch[last] + (vy if np.isclose(y[last], touch[last]) else -vy) * horizon
        if goal_half_width and to_goal < -margin and to_goal <= to_touch:
            end_y = y[last] + vy * horizon
            if abs(end_y - width / 2.0) < goal_half_width:
                continue  # towards the goal mouth: maybe a keeper's catch
        if min(to_goal, to_touch) < -margin:
            out[lo:min(hi, lo + cover)] = 1.0
    return out


def indicators(feats):
    """{rule: (n_frames,) float array in [0, 1]}."""
    slow = slow_play(feats)

    ball_out = _get(feats, "ball_line_dist") < -config.OUT_MARGIN

    still = np.nan_to_num(_get(feats, "ball_speed"), nan=np.inf) < config.BALL_STILL_SPEED
    still_long = features.run_lengths(still, ignore=False) >= config.BALL_STILL_FRAMES

    lost = _get(feats, "ball_gap_len") > config.LOST_BALL_FRAMES

    setpiece = np.nan_to_num(_get(feats, "setpiece_window"), nan=0.0) > 0

    threshold = config.KICKOFF_OWN_HALF_FRAC
    kickoff = ((_get(feats, "home_own_half_frac") > threshold)
               & (_get(feats, "away_own_half_frac") > threshold))

    return {
        "out": ball_out.astype(np.float64),
        "slow_play": slow,
        "ball_still": still_long.astype(np.float64),
        "lost_and_slow": lost * slow,
        "setpiece": setpiece.astype(np.float64),
        "kickoff": kickoff.astype(np.float64),
        "exit": ball_exit(feats),
    }


def p_dead(feats):
    """(n_frames,) probability that the ball is dead. Never NaN."""
    logit = np.full(len(feats["ball_visible"]), config.DEAD_BIAS)
    for name, rule in indicators(feats).items():
        logit = logit + config.DEAD_WEIGHTS[name] * rule
    return 1.0 / (1.0 + np.exp(-logit))
