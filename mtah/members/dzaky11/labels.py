"""Bagian 7-8 -- put the pieces together into one H/A/D/X label per frame, then tidy the spells.

label_match() is the whole pipeline as one pure function of the tracking and the tags, so the same
code runs on the provided match and, in Tahap 3, on external games converted to the same arrays.
"""

import collections

import numpy as np

from mtah.members.dzaky11.config import A, H, NONE, Config, team_of_tag
from mtah.members.dzaky11.dead_ball import (SET_PIECES, ball_gaps, dead_intervals, dead_mask, dead_score,
                                hard_dead_mask, home_goal_by_half, out_of_play_mask,
                                resting_ball_mask, restarts)
from mtah.members.dzaky11.possession import (control_frames, find_touches, nearest_team_at,
                                 possession_in_segment)
from mtah.members.dzaky11.preprocess import preprocess_ball
from mtah.members.dzaky11.runs import long_runs_mask, value_runs

TEAM_LABEL = {H: "H", A: "A"}

Result = collections.namedtuple("Result", "labels intervals restarts touches gaps")


def phase_switches(events, ball_missing_long, cfg=Config()):
    """Bagian 7 [ADAPTASI]: Smart Tagging phases that may switch possession while the ball is lost.

    Only phases that start inside a long gap in the ball track count. Where the ball is seen, the
    touch rules have better evidence than a tag accurate to about a second.
    """
    out = []
    for event in events:
        if (event["code"] or "").strip().lower() == SET_PIECES:
            continue
        frame = event["start_frame"]
        team_code = team_of_tag(event["tags"].get("Team"))
        if team_code != NONE and 0 <= frame < len(ball_missing_long) and ball_missing_long[frame]:
            out.append((frame, team_code))
    return sorted(out)


def tagged_phase_team(events, start, stop):
    """Team whose Smart Tagging phases cover most of [start, stop), or NONE."""
    cover = {H: 0, A: 0}
    for event in events:
        if (event["code"] or "").strip().lower() == SET_PIECES:
            continue
        team_code = team_of_tag(event["tags"].get("Team"))
        overlap = min(stop, event["end_frame"]) - max(start, event["start_frame"])
        if team_code != NONE and overlap > 0:
            cover[team_code] += overlap
    if cover[H] == cover[A]:
        return NONE
    return H if cover[H] > cover[A] else A


def live_segments(first, last, intervals):
    """[(start, stop)] stretches of a half between its dead intervals, half-open."""
    out, cursor = [], first
    for start, stop in intervals:
        if start > cursor:
            out.append((cursor, start))
        cursor = max(cursor, stop)
    if cursor <= last:
        out.append((cursor, last + 1))
    return out


def tidy_spells(labels, min_length):
    """Bagian 8 [ADAPTASI]. Hand every H/A spell shorter than min_length to a neighbour.

    Shortest first. A short spell between two spells of the same team is a deflection and joins
    them; between two different teams it goes to the longer neighbour; with no H/A neighbour --
    squeezed between two stoppages -- it becomes D. D spells are left alone: Bagian 5 already
    dropped the ones too short to trust.
    """
    runs = [[value, stop - start] for start, stop, value in value_runs(labels)]
    while len(runs) > 1:
        short = [i for i, (value, length) in enumerate(runs)
                 if value in ("H", "A") and length < min_length]
        if not short:
            break
        i = min(short, key=lambda k: (runs[k][1], k))
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(runs) and runs[j][0] in ("H", "A")]
        if neighbours:
            runs[i][0] = runs[max(neighbours, key=lambda j: (runs[j][1], -j))][0]
        else:
            runs[i][0] = "D"
        merged = [runs[0]]
        for value, length in runs[1:]:
            if value == merged[-1][0]:
                merged[-1][1] += length
            else:
                merged.append([value, length])
        runs = merged
    return np.concatenate([np.full(length, value, dtype="<U1") for value, length in runs])


def label_match(tracking, events, halves, cfg=Config()):
    """The whole pipeline, Bagian 1-8. Returns a Result whose labels has one letter per frame.

    halves is [(half, first_frame, last_frame)] inclusive, as games.half_video_blocks returns it.
    events is the Smart Tagging list (may be empty).
    """
    if not cfg.PAKAI_SMART_TAGGING:
        events = []
    n = len(tracking)

    # Bagian 1-3
    ball, motion = preprocess_ball(tracking.ball, cfg)
    _, near_team, _ = control_frames(tracking.players, ball, tracking.team, cfg)
    touches = find_touches(near_team, motion, cfg)

    # Bagian 5-6
    home_goal = home_goal_by_half(tracking.players, tracking.team, halves, cfg)
    gaps = ball_gaps(tracking.ball, tracking.speeds, halves, cfg)
    hard = hard_dead_mask(gaps, n, out_of_play_mask(ball, cfg), resting_ball_mask(motion, cfg))
    score = dead_score(tracking, ball, motion, events, halves, home_goal, cfg)
    intervals = dead_intervals(dead_mask(score, hard, np.isnan(ball[:, 0]), cfg), halves, cfg)
    restart_list = restarts(tracking, ball, intervals, events, halves, home_goal, cfg)
    restart_team = {restart.stop: restart.team for restart in restart_list}

    switches = phase_switches(
        events, long_runs_mask(np.isnan(ball[:, 0]), cfg.GAP_HILANG_PANJANG), cfg)

    # Bagian 7
    labels = np.full(n, "X", dtype="<U1")
    for _, first, last in halves:
        half_intervals = [(a, b) for a, b in intervals if first <= a <= last]
        team_codes = np.full(last + 1 - first, NONE, dtype=np.int8)

        for start, stop in live_segments(first, last, half_intervals):
            if start == first:
                initial = nearest_team_at(tracking.players, ball, tracking.team, start, cfg.R_PZ)
            else:
                initial = restart_team.get(start, NONE)
            segment = possession_in_segment(start, stop, touches, initial, switches, cfg)
            if np.all(segment == NONE):
                segment[:] = tagged_phase_team(events, start, stop)
            team_codes[start - first:stop - first] = segment

        # Anything still unknown takes the last known team (forward fill), or the first one at
        # the head of a half. Never X inside a half: spellF1 would fill it the same way, and
        # macroF1 would count it wrong.
        known = np.flatnonzero(team_codes != NONE)
        if len(known):
            carry = np.maximum.accumulate(np.where(team_codes != NONE,
                                                   np.arange(len(team_codes)), -1))
            carry[carry < 0] = known[0]
            team_codes = team_codes[carry]

        half_labels = np.where(team_codes == H, "H", np.where(team_codes == A, "A", "D"))
        for start, stop in half_intervals:
            half_labels[start - first:stop - first] = "D"

        # Bagian 8
        labels[first:last + 1] = tidy_spells(half_labels, cfg.SPELL_MIN)

    return Result(labels, intervals, restart_list, touches, gaps)
