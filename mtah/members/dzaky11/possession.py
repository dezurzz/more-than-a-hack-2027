"""Bagian 2-4 -- who is near the ball, who actually touched it, and when possession changes.

Everything here is decided per team, never per player. ATD slots are tracklets, not people: they
swap identity as the tracker loses and re-acquires players, so a rule such as "the next player to
touch it is not A" would fire on every swap. The submission only asks for H or A, and the only
question that matters for it is whether the ball has passed to the other team under control.
"""

import collections

import numpy as np

from mtah.members.dzaky11.config import H, NONE, Config
from mtah.members.dzaky11.runs import value_runs

# Bagian 2 status codes.
HILANG, DUEL, POSSESSION, TANPA_POSSESSION = 0, 1, 2, 3

CHUNK = 20000  # frames per block when computing distances, to keep memory flat

Touch = collections.namedtuple("Touch", "start stop team")


def control_frames(players, ball, team, cfg=Config()):
    """Bagian 2. (status, near_team, near_distance) per frame.

    near_team is the team of the closest player when status is POSSESSION and NONE otherwise: in
    a DUEL nobody gains anything, possession simply stays where it was. near_distance is the
    distance to the closest player of either team, NaN when the ball or every player is missing.
    """
    n = len(ball)
    status = np.full(n, HILANG, dtype=np.int8)
    near_team = np.full(n, NONE, dtype=np.int8)
    near_distance = np.full(n, np.nan, dtype=np.float32)
    is_home = team == H

    for start in range(0, n, CHUNK):
        stop = min(n, start + CHUNK)
        offset = players[start:stop] - ball[start:stop, None, :]
        distance = np.hypot(offset[:, :, 0], offset[:, :, 1])
        distance = np.where(np.isnan(distance), np.inf, distance)

        nearest = np.argmin(distance, axis=1)
        closest = distance[np.arange(stop - start), nearest]
        seen = np.isfinite(closest)

        in_duel = distance <= cfg.R_DZ
        duel = in_duel[:, is_home].any(axis=1) & in_duel[:, ~is_home].any(axis=1)
        owned = ~duel & (closest <= cfg.R_PZ)

        block = np.where(duel, DUEL, np.where(owned, POSSESSION, TANPA_POSSESSION))
        block[~seen] = HILANG
        status[start:stop] = block
        near_team[start:stop] = np.where(block == POSSESSION, team[nearest], NONE)
        near_distance[start:stop] = np.where(seen, closest, np.nan)

    return status, near_team, near_distance


def is_touch(start, stop, motion, cfg=Config()):
    """Bagian 3, step 1 [ASLI]: did the ball turn or change speed while it was near this team?

    A ball that rolls past a player in a straight line at constant speed was not touched. Where
    the direction or speed cannot be measured -- the run sits against a gap in the ball track --
    a ball that stayed near the team for T_KONTROL frames is taken as touched.
    """
    last = stop - 1
    speed_in, speed_out = motion.speed_in[start], motion.speed_out[last]
    if speed_in >= cfg.V_ARAH_MIN and speed_out >= cfg.V_ARAH_MIN:
        cosine = np.dot(motion.v_in[start], motion.v_out[last]) / (speed_in * speed_out)
        if cosine < cfg.EPS_THETA:
            return True

    change = np.abs(motion.speed_in[start:stop] - motion.speed_out[start:stop])
    known = ~np.isnan(change)
    if known.any() and (change[known] > cfg.EPS_V).any():
        return True

    measurable = not (np.isnan(speed_in) or np.isnan(speed_out)) or known.any()
    return not measurable and stop - start >= cfg.T_KONTROL


def find_touches(near_team, motion, cfg=Config()):
    """Bagian 3, step 1. [Touch] in frame order, one per run of frames near one team."""
    out = []
    for start, stop, value in value_runs(near_team):
        if value != NONE and is_touch(start, stop, motion, cfg):
            out.append(Touch(start, stop, int(value)))
    return out


def is_controlled(touches, index, cfg=Config()):
    """Bagian 3, step 2 [ADAPTASI]: is touches[index] controlled possession, not a deflection?

    CHALLENGE.md: a touch without control, a deflected pass and an aerial duel won without
    keeping the ball do not change possession. The proxies are the ones the organisers suggest --
    the team keeps the ball a moment, or touches it again before the other team does.
    """
    touch = touches[index]
    if touch.stop - touch.start >= cfg.T_KONTROL:
        return True
    following = touches[index:index + cfg.N_SENTUH_KONTROL]
    return (len(following) == cfg.N_SENTUH_KONTROL
            and all(other.team == touch.team for other in following))


def possession_in_segment(start, stop, touches, initial=NONE, switches=(), cfg=Config()):
    """Bagian 4 [ASLI, team level]. Team in possession on every frame of one live segment.

    possession starts with `initial` -- the team restarting play -- and changes only when the
    other team gains controlled possession. Loose and contested balls stay with the last team,
    as CHALLENGE.md defines it, so there is no separate loss rule: a loss is the other side of a
    gain.

    The gain is placed on the first frame of the controlling touch, not on the frame where control
    is confirmed, so the spell boundary is not late. A boundary 1 s late costs a perfect
    submission about a third of its spellF1 (EVALUATION.md).

    If `initial` is NONE the first team to touch the ball is taken to have had it from the start
    of the segment -- after a stoppage, the first team to touch it is the one restarting.

    switches are (frame, team) pieces of outside evidence (Bagian 7, Smart Tagging) that take
    effect when reached, exactly like a gain.

    Returns an int8 array of length stop - start, NONE where nothing is known at all.
    """
    inside = [touch for touch in touches if start <= touch.start < stop]
    events = [(touch.start, 0, i) for i, touch in enumerate(inside)]
    events += [(frame, 1, team) for frame, team in switches if start <= frame < stop]
    events.sort()

    current = initial
    gains = [(start, initial)] if initial != NONE else []
    for frame, kind, value in events:
        if kind == 1:
            if value != current:
                current = value
                gains.append((frame, value))
            continue
        touch = inside[value]
        if current == NONE:
            current = touch.team
            gains.append((start, current))
        elif touch.team != current and is_controlled(inside, value, cfg):
            current = touch.team
            gains.append((touch.start, current))

    out = np.full(stop - start, NONE, dtype=np.int8)
    for (frame, team), following in zip(gains, gains[1:] + [(stop, None)]):
        out[frame - start:following[0] - start] = team
    return out


def nearest_team_at(players, ball, team, frame, radius):
    """Team of the player within `radius` of the ball on one frame, or NONE."""
    if frame < 0 or frame >= len(ball) or np.isnan(ball[frame, 0]):
        return NONE
    distance = np.hypot(*(players[frame] - ball[frame]).T)
    if np.all(np.isnan(distance)):
        return NONE
    nearest = int(np.nanargmin(distance))
    return int(team[nearest]) if distance[nearest] <= radius else NONE
