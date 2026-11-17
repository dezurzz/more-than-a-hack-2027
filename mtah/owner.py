"""A mechanical "last owner" rule, and a label-free full-match proxy for team accuracy.

The ground truth is derived mechanically from corrected tracking: a frame takes "the team the
tracking names as owner" (CHALLENGE.md). Short possessions of 0.76 s in the public truth suggest a
simple ownership rule rather than a football judgement. last_owner() is the simplest such rule:
the ball belongs to the team of the last player who stayed nearest to it, within `radius`, for
`confirm` consecutive frames; it keeps that team until another team does the same.

tag_agreement() scores a full-match H/A sequence against the strongest Smart Tagging phases, an
independent source whose team agrees with our calls far above chance. Tags are noisy too,
but between two versions of our model, more agreement over 106 minutes is evidence of better team
calls, from ~95x more frames than the public window. It needs no ground truth.
"""

import numpy as np

# Params
# Tags with the largest whole-match lift, attacking side only (their mirrors are the same
# intervals). Used for the proxy only, never as model input by this module.
PROXY_TAGS = ("tag_build_up", "tag_progression", "tag_attacking_transition")


def last_owner(nearest_team, nearest_dist, radius, confirm, carried=None, touch_radius=None,
               touch_lookback=250, confirm_from=None, hold=None, initial=-1, forced=None,
               fallback=None):
    """(n,) int8 owner team per frame: 0 home, 1 away, -1 before anyone has owned the ball.

    initial: the owner at the first frame (-1 = nobody), e.g. the team taking a restart.
    fallback: (radius, after) or None. Once no player has been within `radius` (the main one) for
    `after` frames, a player within the fallback radius counts too, until someone is close again:
    a loosely tracked ball never comes within 1 m of anybody, and the owner froze.
    forced: optional (n,) int array; where it is 0 / 1 that team owns the frame outright (a keeper
    holding a ball the tracking has lost), and it stays the owner until another is confirmed.

    hold: (min_run, max_gap) or None. A ball that disappears from the tracking right after
    `min_run` frames within `radius` of a player keeps counting for that player's team for up to
    `max_gap` frames: a keeper holding the ball is not tracked.

    A change is dated back to the first frame of the confirming run, so `confirm` does not add lag.
    `confirm_from`, if given, is an (n, 2) array: the run needed to take the ball from team 0 / 1
    at that frame (the base `confirm` applies while nobody owns it yet).
    `carried`, if given, is a boolean mask of frames where the ball travels with the nearest player
    (low relative speed); a frame outside it cannot count towards ownership.

    touch_radius: if given, a change is dated back further, to the new team's first touch: the first
    frame, after the old owner's last touch, with a player of the new team within `touch_radius`.
    (A touch is any frame a player is that close; the old owner's last touch is the last frame one
    of its players was.) The ball in between was loose, and the truth hands a loose ball to the
    team that touched it and then controlled it. Looks back at most `touch_lookback` frames.
    """
    near = (nearest_dist < radius) & ~np.isnan(nearest_team)
    if carried is not None:
        near &= carried
    candidate = np.where(near, nearest_team, -1)
    candidate = candidate.astype(np.int8)
    loose = None
    if fallback is not None:
        loose = np.where((nearest_dist < fallback[0]) & ~np.isnan(nearest_team), nearest_team,
                         -1).astype(np.int8).tolist()
    touching = None
    if touch_radius is not None:
        touching = np.where((nearest_dist < touch_radius) & ~np.isnan(nearest_team),
                            nearest_team, -1).astype(np.int8)
    out = np.full(len(candidate), -1, dtype=np.int8)
    current, run_team, run = initial, -1, 0
    missing = np.isnan(nearest_dist).tolist()
    gap = 0
    since_close = 0
    for i, team in enumerate(candidate.tolist()):
        if loose is not None:
            since_close = 0 if team >= 0 else since_close + 1
            if team < 0 and since_close > fallback[1]:
                team = loose[i]
        if hold is not None:
            if team < 0 and missing[i] and run_team >= 0 and run >= hold[0] and gap < hold[1]:
                team = run_team     # the ball vanished in this player's hands: he still has it
            gap = gap + 1 if missing[i] else 0
        if team >= 0 and team == run_team:
            run += 1
        elif team >= 0:
            run_team, run = team, 1
        else:
            run_team, run = -1, 0
        if forced is not None and forced[i] >= 0:
            current = run_team = int(forced[i])
            run = confirm
            out[i] = current
            continue
        need = confirm if confirm_from is None or current < 0 else confirm_from[i, current]
        if run_team >= 0 and run >= need and run_team != current:
            start = i - run + 1
            if touching is not None and current >= 0:
                low = max(0, start - touch_lookback)
                window = touching[low:start]
                last_old = np.flatnonzero(window == current)
                first_new = np.flatnonzero(window == run_team)
                first_new = first_new[first_new > (last_old[-1] if len(last_old) else -1)]
                if len(first_new):
                    start = low + int(first_new[0])
            current = run_team
            out[start:i] = current
        out[i] = current
    return out


def owner_by_segment(nearest_team, nearest_dist, alive, radius, confirm, carried=None,
                     touch_radius=None, touch_lookback=250, confirm_from=None, hold=None,
                     taker=None, forced=None, fallback=None):
    """Like last_owner, but the owner is reset at every restart.

    taker: optional (n,) int array, the team taking the restart at each frame (0 / 1, -1 unknown).
    A run starting where it is known begins owned by that team (the taker touches the ball first,
    even when nobody stays near it long enough to be confirmed: a corner).

    `alive` is a boolean array, False while the ball is dead. Each maximal alive run is handled on
    its own: the owner carried over a dead ball is not kept (CHALLENGE.md: after every restart there
    is a clear possessor, so the previous owner says nothing about the new possession); frames before
    the run's first confirmed owner take that owner's team; a run with no confirmed owner keeps the
    last team of the run before it. -1 only where nothing was ever known.
    """
    out = np.full(len(alive), -1, dtype=np.int8)
    edges = np.diff(np.concatenate(([0], alive.astype(np.int8), [0])))
    previous = -1
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        run = last_owner(nearest_team[start:stop], nearest_dist[start:stop], radius, confirm,
                         None if carried is None else carried[start:stop], touch_radius,
                         touch_lookback,
                         None if confirm_from is None else confirm_from[start:stop], hold,
                         -1 if taker is None else int(taker[start]),
                         None if forced is None else forced[start:stop], fallback)
        known = run[run >= 0]
        run = np.where(run >= 0, run, known[0] if known.size else previous).astype(np.int8)
        out[start:stop] = run
        previous = run[-1] if run[-1] >= 0 else previous
    return out


def throw_in_takers(owners, alive, setpiece_type, taker, types=(1,)):
    """Add throw-in takers to `taker`: the team opposite to the last owner before the stoppage.

    Throw-ins carry no team in the tags. For each alive run that starts inside a throw-in window and
    has no taker yet, the taker is the other team than the owner at the end of the previous alive run
    (the team that touched the ball last put it out). Unknown stays -1. Returns a new array.
    """
    out = np.full(len(alive), -1, dtype=np.int8) if taker is None else np.asarray(taker).copy()
    edges = np.diff(np.concatenate(([0], alive.astype(np.int8), [0])))
    last = -1
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if out[start] < 0 and last >= 0 and setpiece_type[start] in types:
            out[start] = 1 - last
        known = owners[start:stop][owners[start:stop] >= 0]
        last = int(known[-1]) if known.size else last
    return out


def drop_loose_blips(owners, nearest_dist, alive, max_len, min_median_dist):
    """Hand a short possession back to the team on both sides of it, if nobody was near the ball.

    A blip is an owner run of at most `max_len` frames, inside one alive stretch, between two runs of
    the other team. It is dropped when the median distance from the ball to the nearest player
    (either team) over the run is above `min_median_dist`: the team was confirmed by a fleeting
    approach and the ball then ran loose, which is not control.
    Returns a new array; a run with no tracked ball keeps its label.
    """
    out = owners.copy()
    edges = np.diff(np.concatenate(([0], alive.astype(np.int8), [0])))
    for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        seg = out[lo:hi]
        cuts = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = np.concatenate(([0], cuts, [len(seg)]))
        for k in range(1, len(bounds) - 2):
            a, b = bounds[k], bounds[k + 1]
            before, after = seg[bounds[k - 1]], seg[bounds[k + 1]]
            if before < 0 or seg[a] < 0 or before != after or before == seg[a] or b - a > max_len:
                continue
            dist = nearest_dist[lo + a:lo + b]
            if np.isnan(dist).all():
                continue
            if np.nanmedian(dist) > min_median_dist:
                seg[a:b] = before
    return out


def keeper_catches(feats, radius, min_gap, cover, box_half_width, box_depth):
    """(n,) int8: 0 / 1 where that team's keeper has caught a ball the tracking then lost, else -1.

    A gap of at least `min_gap` frames in the ball tracking whose last tracked position is inside a
    penalty box with that box's own keeper within `radius` m: a keeper holding the ball is not
    tracked. Covers the first `cover` frames of the gap.
    """
    visible = np.nan_to_num(feats["ball_visible"], nan=0.0) > 0
    x, y = feats["ball_x_home_attack"], feats["ball_y"]      # x: 0 = home goal line
    goal, touch = feats["ball_goal_line_dist"], feats["ball_touch_line_dist"]
    far = visible & ~np.isclose(y, touch)
    width = float(np.median(y[far] + touch[far]))           # pitch width from the ball features
    out = np.full(len(visible), -1, dtype=np.int8)
    edges = np.diff(np.concatenate(([0], (~visible).astype(np.int8), [0])))
    for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if hi - lo < min_gap or lo == 0:
            continue
        last = lo - 1
        inside = 0 <= goal[last] < box_depth and abs(y[last] - width / 2.0) <= box_half_width
        if np.isnan(x[last]) or not inside:
            continue
        if x[last] <= goal[last] + 1e-6:                     # nearer the home goal line
            team, keeper = 0, feats["home_keeper_dist"][last]
        else:
            team, keeper = 1, feats["away_keeper_dist"][last]
        if keeper < radius:
            out[lo:min(hi, lo + cover)] = team
    return out


def confirm_from_counts(home_n, away_n, confirm, short_confirm, full=11):
    """(n, 2) run needed to take the ball from home / away: `short_confirm` while that team has
    fewer than `full` players tracked and the other side is complete, else `confirm`.

    An untracked ball carrier hands the ball to the opponent marking him, so a change away from a
    short team is the less trustworthy one.
    """
    home_n, away_n = np.nan_to_num(home_n, nan=0.0), np.nan_to_num(away_n, nan=0.0)
    need = np.full((len(home_n), 2), confirm, dtype=np.int32)
    need[(home_n < full) & (away_n >= full), 0] = short_confirm
    need[(away_n < full) & (home_n >= full), 1] = short_confirm
    return need


def owner_labels(owner):
    """Owner codes as H / A labels (H before the first owner)."""
    return np.where(owner == 1, "A", "H")


def tag_agreement(labels, feats, mask, tags=PROXY_TAGS):
    """(agreement, frames) of H/A labels with the proxy tags' teams, over `mask`."""
    labels = np.asarray(labels)
    sign = np.zeros(len(labels))
    for name in tags:
        sign = sign + np.nan_to_num(feats[name], nan=0.0)
    keep = mask & (sign != 0) & ((labels == "H") | (labels == "A"))
    if not keep.any():
        return None, 0
    return float(np.mean((sign[keep] > 0) == (labels[keep] == "H"))), int(keep.sum())
