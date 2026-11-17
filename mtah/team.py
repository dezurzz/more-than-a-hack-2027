"""P(home | ball alive) per frame.

    z = W_DIST * clip(dist_diff) + W_CONTROL * (who has been on the ball, and how long)
        + W_NEAR * (home - away players within 2 m) + P4 terms
    P(home | alive) = sigmoid(z)

Positive z is HOME. NaN terms count as 0; the result is NaN only where there is nothing to go on
at all (no ball, and no tag evidence), which the decoder reads as "keep the current team".
"""

import numpy as np

from mtah import config, features
from games import GAME, GAMES  # noqa: E402


def _get(feats, name):
    return feats[name] if name in feats else np.full(len(feats["ball_visible"]), np.nan)


def carried_control(feats):
    """Signed control run: how many frames in a row one team has been ON the ball, +home / -away.

    Control only counts while the ball travels with the player (ball_rel_speed below
    CONTROL_MAX_REL_SPEED): a deflection or a long kick passes near a player without being
    controlled by him. Frames without control are 0.
    """
    control = _get(feats, "control_team")
    carried = ~np.isnan(control) & (_get(feats, "ball_rel_speed") < config.CONTROL_MAX_REL_SPEED)
    codes = np.where(carried, control, 0.0)
    run = features.run_lengths(codes, ignore=0.0)
    return np.nan_to_num(codes) * np.minimum(run, config.CONTROL_SATURATION)


def restart_term(feats):
    """Restarting team's signed bonus, decaying over config.RESTART_PROFILE after a window starts.

    The windows are 35 s long and start about 3 s before the restart. The window's
    own team tag decides (+1 home, -1 away); 0 for N/A (every throw-in), so no bonus there.
    Frames outside any window, or past the profile, get 0.
    """
    out = np.zeros(len(feats["ball_visible"]))
    window = np.nan_to_num(_get(feats, "setpiece_window"), nan=0.0) > 0
    team = np.nan_to_num(_get(feats, "setpiece_team"), nan=0.0)
    rising = window & ~np.concatenate(([False], window[:-1]))
    fps = GAMES[GAME]["fps"]
    for start in np.flatnonzero(rising):
        begin = start
        for end_seconds, multiplier in config.RESTART_PROFILE:
            stop = start + int(round(end_seconds * fps))
            out[begin:stop] = multiplier * np.where(window[begin:stop], team[begin:stop], 0.0)
            begin = stop
    return out


def tag_term(feats):
    """Weighted phase tags whose direction was fixed from the feature report (config)."""
    out = np.zeros(len(feats["ball_visible"]))
    for name in sorted(config.TAG_DIRECTIONS):  # sorted: float sums must not depend on dict order
        out = out + config.TAG_DIRECTIONS[name] * np.nan_to_num(_get(feats, name), nan=0.0)
    return out


def z_score(feats):
    """(n_frames,) log-odds of home, 0 where there is no evidence."""
    dist = np.clip(np.nan_to_num(_get(feats, "dist_diff"), nan=0.0),
                   -config.DIST_CLIP, config.DIST_CLIP)
    near = (np.nan_to_num(_get(feats, "n_home_2m"), nan=0.0)
            - np.nan_to_num(_get(feats, "n_away_2m"), nan=0.0))
    return (config.W_DIST * dist
            + config.W_CONTROL * carried_control(feats) / config.CONTROL_SATURATION
            + config.W_NEAR * near
            + config.W_RESTART * restart_term(feats)
            + config.W_TAG * tag_term(feats))


def p_home(feats):
    """(n_frames,) P(home | alive); NaN where there is no evidence at all."""
    z = z_score(feats)
    ball_evidence = (~np.isnan(_get(feats, "dist_diff")) | ~np.isnan(_get(feats, "control_team"))
                     | ~np.isnan(_get(feats, "n_home_2m")))
    return np.where(ball_evidence | (z != 0.0), 1.0 / (1.0 + np.exp(-z)), np.nan)


def p_home_from_nearest(feats):
    """E003 only: a raw team call from the nearest player (BASELINE_TEAM_P), NaN without a ball."""
    nearest = _get(feats, "nearest_team")
    return np.where(np.isnan(nearest), np.nan,
                    np.where(nearest == 0, config.BASELINE_TEAM_P, 1.0 - config.BASELINE_TEAM_P))
