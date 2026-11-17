"""IDSSE match -> the ATD's Tracking structure.

Same container as mtah/io_atd.py: coordinates normalised 0-1 (x along the length, y across, as the
ATD's DATA.md says), speeds in m/s (the ATD's speed column equals the finite difference in m/s), 46 tracker slots (24 home, 22 away, like the ATD) instead of person ids, NaN where nothing
is tracked, no ball height. IDSSE rows are contiguous per half; `present` is True everywhere and
the halves are `Match.runs`.

Slots: a person who is tracked takes a slot of their team; when they stop being tracked the slot is
free again, and when they come back they take another free slot of the team. With an rng the free
slot is a random one (identities churn like the ATD's); without, the lowest free (clean copy).
"""

import numpy as np

from mtah.io_atd import Tracking

N_SLOTS = (24, 22)          # home, away: the ATD's team_of_slot counts


def team_of_slot():
    return np.array([0] * N_SLOTS[0] + [1] * N_SLOTS[1], dtype=np.int8)


KMH_TO_MS = 1.0 / 3.6
SPEED_HALF_WINDOW = 2            # frames either side for the ball's speed


def ball_speed_ms(x_m, y_m, half=SPEED_HALF_WINDOW, fps=25):
    """(n,) m/s: the ball's displacement over 2 * half frames, NaN unless both ends are observed.

    The ball's speed column cannot be trusted in either source (IDSSE's is in other units than its
    positions imply; the ATD's correlates 0.06 with its own positions), so every consumer computes
    the speed from the positions, the same way on both.
    """
    n = len(x_m)
    out = np.full(n, np.nan, dtype=np.float32)
    if n > 2 * half:
        centre = np.arange(half, n - half)
        distance = np.hypot(x_m[centre + half] - x_m[centre - half], y_m[centre + half] - y_m[centre - half])
        out[centre] = distance / (2 * half) * fps
    return out


def normalise(match):
    """(players (n, P, 3), ball (n, 3)) float32: x, y in 0-1 and speed in m/s; the ball's z is dropped.

    IDSSE's speed column is km/h (against the finite difference of its positions the ratio is 3.60): players' speed = S / 3.6. The ball's speed comes from its positions (ball_speed_ms).
    """
    length, width = match.pitch
    players = match.players.copy()
    players[..., 0] = players[..., 0] / length + 0.5
    players[..., 1] = players[..., 1] / width + 0.5
    players[..., 2] = players[..., 2] * KMH_TO_MS
    ball = np.empty((match.n_frames, 3), dtype=np.float32)
    ball[:, 0] = match.ball[:, 0] / length + 0.5
    ball[:, 1] = match.ball[:, 1] / width + 0.5
    ball[:, 2] = ball_speed_ms(match.ball[:, 0].astype(np.float64), match.ball[:, 1].astype(np.float64))
    return players, ball


def _segments(column):
    edges = np.diff(np.concatenate(([0], column.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def assign_slots(presence, person_team, rng=None):
    """(n, P) int16 slot of each tracked person, -1 where not tracked (see the module docstring)."""
    n, n_persons = presence.shape
    slot_of = np.full((n, n_persons), -1, dtype=np.int16)
    free = [list(range(N_SLOTS[0])), list(range(N_SLOTS[0], N_SLOTS[0] + N_SLOTS[1]))]
    events = []                                  # (frame, 0 release / 1 take, person, start)
    for person in range(n_persons):
        for start, stop in _segments(presence[:, person]):
            events.append((int(start), 1, person, int(start), int(stop)))
            events.append((int(stop), 0, person, int(start), int(stop)))
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    held = {}
    for frame, kind, person, start, stop in events:
        team = int(person_team[person])
        if kind == 1:
            pool = free[team]
            if not pool:
                raise ValueError("more than {} tracked people of one team at once".format(N_SLOTS[team]))
            slot = pool.pop(int(rng.integers(len(pool))) if rng is not None else pool.index(min(pool)))
            held[(person, start)] = slot
            slot_of[start:stop, person] = slot
        else:
            free[team].append(held.pop((person, start)))
    return slot_of


def to_tracking(match, ball_missing=None, absent=None, rng=None):
    """Tracking of one match. ball_missing (n,) bool blanks the ball; absent (n, P) bool blanks people."""
    players, ball = normalise(match)
    if ball_missing is not None:
        ball[ball_missing] = np.nan
    presence = ~np.isnan(players[..., 0])
    if absent is not None:
        presence &= ~absent
    slot_of = assign_slots(presence, match.person_team, rng)
    slots = np.full((match.n_frames, sum(N_SLOTS), 3), np.nan, dtype=np.float32)
    for person in range(presence.shape[1]):
        for start, stop in _segments(presence[:, person]):
            slots[start:stop, slot_of[start, person]] = players[start:stop, person]
    return Tracking(players=slots, ball=ball, present=np.ones(match.n_frames, dtype=bool),
                    team_of_slot=team_of_slot(), team_names=(match.home_name, match.away_name),
                    pitch=tuple(match.pitch), n_skipped_lines=0)
