"""Bagian 5-6 -- when is the ball dead, and which team restarts play?

The paper reads ball-in-play from the data provider. We have no such flag, so dead ball is
inferred, by two rules whose results are combined:

- direct evidence, taken frame for frame: the ball seen out of play, the ball seen at rest for
  two seconds (a placed goal kick or free kick), and the gap rule -- on this
  ATD the ball goes missing in long blocks that line up with stoppages (pipeline/diagnose.py,
  section D), so a long gap is a stoppage unless the players are clearly still playing;
- the score rule: evidence points per frame (ball out of play, ball resting, SET PIECES tag,
  restart layout), averaged and thresholded -- mainly for the tag and layout, which are too
  loose in time to be taken frame for frame.

Any gap in the ball track with dead frames on both sides is then dead as well.

Smart Tagging helps but cannot carry it alone: it has 59 SET PIECES instances, a match has well
over a hundred stoppages, and every throw-in is tagged Team = N/A.

Which team restarts is not part of the answer -- the ground truth labels every dead frame D, with
no owner -- but it decides the label of the spell that follows, so it is worth getting right.
"""

import collections
import warnings

import numpy as np

from mtah.members.dzaky11.config import A, H, NONE, Config, team_of_tag
from mtah.members.dzaky11.possession import nearest_team_at
from mtah.members.dzaky11.runs import centred_mean, long_runs_mask, true_runs

SET_PIECES = "set pieces"

KICK_OFF, PENALTY, CORNER, GOAL_KICK, THROW_IN, FREE_KICK = (
    "kick_off", "penalty", "corner", "goal_kick", "throw_in", "free_kick")

# Pitch markings (Laws of the Game), in metres.
PENALTY_SPOT = 11.0
GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH = 5.5, 9.16
PENALTY_AREA_DEPTH, PENALTY_AREA_HALF_WIDTH = 16.5, 20.16
MIN_PLAYERS_KICKOFF = 10  # a kickoff layout needs most of the players to be seen

CHUNK = 20000

Restart = collections.namedtuple("Restart", "start stop kind team source")
Gap = collections.namedtuple("Gap", "start stop median_speed live")


def _nanmean_rows(values):
    """Row mean ignoring NaN, NaN for an all-NaN row, without numpy's empty-slice warning."""
    count = np.sum(~np.isnan(values), axis=1)
    total = np.nansum(values, axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _nanmedian(values, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def home_goal_by_half(players, team, halves, cfg=Config()):
    """{half: x of the goal Barcelona defends}. Ends swap at halftime.

    Read off the kickoff, the one moment every player is guaranteed to be in their own half.
    """
    out = {}
    for half, first, _ in halves:
        window = players[first:first + cfg.FPS, team == H, 0]
        mean_x = np.nanmean(window) if np.any(~np.isnan(window)) else np.nan
        out[half] = 0.0 if not mean_x > cfg.LAPANGAN_P / 2 else cfg.LAPANGAN_P
    return out


def own_goal_x(team_code, home_goal, length):
    return home_goal if team_code == H else length - home_goal


def _goal_frame_array(n, halves, home_goal):
    """Per frame, the x of the goal home defends; NaN outside the halves."""
    out = np.full(n, np.nan)
    for half, first, last in halves:
        out[first:last + 1] = home_goal[half]
    return out


def restart_layout_mask(players, team, halves, home_goal, cfg=Config()):
    """Bagian 5 [ADAPTASI]: per frame, does the layout look like a throw-in, corner or kickoff?

    Only the layouts that cannot occur in open play are used per frame. A goal kick or a penalty
    needs the ball or a goalkeeper, and a goalkeeper stands in the goal area all match, so those
    two are only checked once a dead stretch is known (classify_restart).
    """
    length, width = cfg.LAPANGAN_P, cfg.LAPANGAN_L
    corners = np.array([(0, 0), (0, width), (length, 0), (length, width)], dtype=np.float32)
    goal_of_frame = _goal_frame_array(len(players), halves, home_goal)
    out = np.zeros(len(players), dtype=bool)

    for start in range(0, len(players), CHUNK):
        stop = min(len(players), start + CHUNK)
        x, y = players[start:stop, :, 0], players[start:stop, :, 1]
        seen = ~np.isnan(x)

        throw_in = np.any(seen & ((y < -cfg.MARGIN_LUAR) | (y > width + cfg.MARGIN_LUAR))
                          & (x >= 0) & (x <= length), axis=1)

        corner = np.zeros(stop - start, dtype=bool)
        for cx, cy in corners:
            corner |= np.any(np.hypot(x - cx, y - cy) <= cfg.R_POJOK, axis=1)

        home_goal = goal_of_frame[start:stop, None]
        defends_left = np.where(team[None, :] == H, home_goal == 0, home_goal != 0)
        own_half = np.where(defends_left, x <= length / 2 + cfg.TOL_SEPARUH,
                            x >= length / 2 - cfg.TOL_SEPARUH)
        at_spot = np.any(np.hypot(x - length / 2, y - width / 2) <= cfg.R_TITIK, axis=1)
        kick_off = (np.all(own_half | ~seen, axis=1) & at_spot
                    & (seen.sum(axis=1) >= MIN_PLAYERS_KICKOFF)
                    & ~np.isnan(goal_of_frame[start:stop]))

        out[start:stop] = throw_in | corner | kick_off
    return out


def set_piece_mask(n, events, cfg=Config()):
    """Frames within JENDELA_SET_PIECE of the start of a SET PIECES tag."""
    out = np.zeros(n, dtype=bool)
    for event in events:
        if (event["code"] or "").strip().lower() != SET_PIECES:
            continue
        low = max(0, event["start_frame"] - cfg.JENDELA_SET_PIECE)
        out[low:max(low, event["start_frame"] + cfg.JENDELA_SET_PIECE + 1)] = True
    return out


def mean_player_speed(speeds):
    """Mean reported speed of the players seen on each frame, NaN where nobody is seen."""
    return _nanmean_rows(speeds)


def out_of_play_mask(ball, cfg=Config()):
    """Frames where the ball is seen more than MARGIN_LUAR beyond a line."""
    length, width, margin = cfg.LAPANGAN_P, cfg.LAPANGAN_L, cfg.MARGIN_LUAR
    with np.errstate(invalid="ignore"):
        return ((ball[:, 0] < -margin) | (ball[:, 0] > length + margin)
                | (ball[:, 1] < -margin) | (ball[:, 1] > width + margin))


def ball_gaps(raw_ball, speeds, halves, cfg=Config()):
    """Bagian 5, gap rule [ADAPTASI]. [Gap] for every stretch the ball is unseen for at least
    DURASI_HILANG_MATI frames, cut at the half boundaries.

    Each carries the median of the per-frame mean player speed inside it, and `live` when that
    is above V_GAP_HIDUP -- players running hard while the tracker has lost the ball are taken to
    be playing on, not waiting for a restart.
    """
    speed = mean_player_speed(speeds)
    missing = np.isnan(raw_ball[:, 0])
    out = []
    for _, first, last in halves:
        for start, stop in true_runs(missing[first:last + 1]):
            start, stop = first + start, first + stop
            if stop - start < cfg.DURASI_HILANG_MATI:
                continue
            inside = speed[start:stop]
            inside = inside[~np.isnan(inside)]
            median = float(np.median(inside)) if len(inside) else float("nan")
            out.append(Gap(start, stop, median, bool(median > cfg.V_GAP_HIDUP)))
    return out


def resting_ball_mask(motion, cfg=Config()):
    """Frames in a run of at least DURASI_DIAM_MIN where the ball is seen at rest.

    No condition on who is near it: a ball placed for a goal kick or a free kick always has the
    taker beside it -- an earlier version required nobody within R_PZ and so missed exactly
    those (diagnose, section G: goal kicks with the ball seen, half undetected). In open play a
    ball stays still for two seconds rarely; a goalkeeper standing still with it in his hands is
    the known exception.
    """
    with np.errstate(invalid="ignore"):
        return long_runs_mask(motion.speed < cfg.V_BOLA_DIAM, cfg.DURASI_DIAM_MIN)


def hard_dead_mask(gaps, n, *masks):
    """Frames that are dead on direct evidence: every gap not vetoed as live, plus every frame
    of each mask given (ball seen out of play, ball at rest).

    Taken frame for frame rather than averaged like the score, because where a stoppage starts
    and ends decides the IoU of the possessions on either side of it -- a placed ball is dead
    exactly until it moves.
    """
    mask = np.zeros(n, dtype=bool)
    for other in masks:
        mask |= other
    for gap in gaps:
        if not gap.live:
            mask[gap.start:gap.stop] = True
    return mask


def bridge_missing(dead, missing):
    """Copy of dead with every run of missing ball that has dead frames on both sides filled in.

    The ball seen out of play, then lost, then seen out of play again was dead throughout,
    however short the gap.
    """
    out = dead.copy()
    for start, stop in true_runs(missing):
        if start > 0 and stop < len(dead) and dead[start - 1] and dead[stop]:
            out[start:stop] = True
    return out


def dead_evidence(tracking, ball, motion, events, halves, home_goal, cfg=Config()):
    """Bagian 5, score rule. {name: (weight, boolean mask)}, one entry per piece of evidence.

    This covers stoppages where the ball stays visible; long gaps are handled by the gap rule.
    Kept as separate masks so pipeline/diagnose.py can report how often each one fires.
    """
    with np.errstate(invalid="ignore"):
        slow = mean_player_speed(tracking.speeds) < cfg.V_PEMAIN_DIAM

    return collections.OrderedDict([
        ("bola_luar", (cfg.BOBOT_LUAR, out_of_play_mask(ball, cfg))),
        ("pemain_diam", (cfg.BOBOT_PEMAIN_DIAM, slow)),
        ("bola_diam", (cfg.BOBOT_BOLA_DIAM, resting_ball_mask(motion, cfg))),
        ("set_piece_tag", (cfg.BOBOT_SET_PIECE, set_piece_mask(len(ball), events, cfg))),
        ("pola_restart", (cfg.BOBOT_POLA, restart_layout_mask(tracking.players, tracking.team,
                                                              halves, home_goal, cfg))),
    ])


def dead_score(tracking, ball, motion, events, halves, home_goal, cfg=Config()):
    """Bagian 5. Evidence points per frame that the ball is out of play."""
    evidence = dead_evidence(tracking, ball, motion, events, halves, home_goal, cfg)
    return sum(weight * mask for weight, mask in evidence.values())


def dead_mask(score, hard_dead, missing, cfg=Config()):
    """Bagian 5. Dead frames: direct evidence, or an averaged score reaching AMBANG_MATI, with
    the ball gaps between dead frames bridged."""
    soft = centred_mean(score, cfg.JENDELA_SKOR) >= cfg.AMBANG_MATI
    return bridge_missing(hard_dead | soft, missing)


def dead_intervals(dead, halves, cfg=Config()):
    """Bagian 5. [(start, stop)] dead stretches, half-open, never spanning halftime."""
    out = []
    for _, first, last in halves:
        for start, stop in true_runs(dead[first:last + 1]):
            if stop - start >= cfg.DURASI_MATI_MIN:
                out.append((first + start, first + stop))
    return out


def classify_restart(players, ball, team, start, stop, home_goal, cfg=Config()):
    """Bagian 6 [ASLI]. (kind, team or NONE) from the layout at the end of a dead stretch.

    Priority KICK_OFF > PENALTY > CORNER > GOAL_KICK > THROW_IN; nothing matching is a FREE_KICK,
    whose team is left to the caller.
    """
    length, width = cfg.LAPANGAN_P, cfg.LAPANGAN_L
    low = max(start, stop - cfg.JENDELA_RESTART)
    layout = _nanmedian(players[low:stop], axis=0)          # (slots, 2)
    ball_at = _nanmedian(ball[low:stop], axis=0)            # (2,)
    seen = ~np.isnan(layout[:, 0])
    if not seen.any():
        return FREE_KICK, NONE
    x, y = layout[:, 0], layout[:, 1]
    slot_team = team

    def nearest_to(px, py, radius):
        distance = np.where(seen, np.hypot(x - px, y - py), np.inf)
        slot = int(np.argmin(distance))
        return (int(slot_team[slot]) if distance[slot] <= radius else NONE), slot

    # Kickoff: every player in their own half and one on the centre spot.
    own_left = np.where(slot_team == H, home_goal == 0, home_goal != 0)
    own_half = np.where(own_left, x <= length / 2 + cfg.TOL_SEPARUH,
                        x >= length / 2 - cfg.TOL_SEPARUH)
    kicker, _ = nearest_to(length / 2, width / 2, cfg.R_TITIK)
    if kicker != NONE and np.all(own_half[seen]) and seen.sum() >= MIN_PLAYERS_KICKOFF:
        return KICK_OFF, kicker

    goals = (0.0, length)
    # Penalty: a player on the spot and one (the keeper) on the goal line.
    for goal in goals:
        spot_x = abs(goal - PENALTY_SPOT)
        taker, _ = nearest_to(spot_x, width / 2, cfg.R_TITIK)
        keeper, _ = nearest_to(goal, width / 2, cfg.R_TITIK)
        if taker != NONE and keeper != NONE and taker != keeper:
            return PENALTY, taker

    # Corner: a player at a corner flag.
    for cx in goals:
        for cy in (0.0, width):
            taker, _ = nearest_to(cx, cy, cfg.R_POJOK)
            if taker != NONE:
                return CORNER, taker

    # Goal kick: the ball in a goal area, or (ball unseen) a player in a goal area with no
    # attacker in that penalty area. Taken by the side defending that goal.
    for goal in goals:
        defender = H if own_goal_x(H, home_goal, length) == goal else A

        def inside(px, py, depth, half_width):
            return (np.abs(px - goal) <= depth) & (np.abs(py - width / 2) <= half_width)

        if not np.isnan(ball_at[0]):
            if inside(ball_at[0], ball_at[1], GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH):
                return GOAL_KICK, defender
            continue
        in_goal_area = seen & inside(x, y, GOAL_AREA_DEPTH, GOAL_AREA_HALF_WIDTH)
        attackers_in_box = (seen & (slot_team != defender)
                            & inside(x, y, PENALTY_AREA_DEPTH, PENALTY_AREA_HALF_WIDTH))
        if in_goal_area.any() and not attackers_in_box.any():
            return GOAL_KICK, defender

    # Throw-in: a player beyond a touchline. Smart Tagging never names the team for these.
    beyond = np.where(seen, np.maximum(-y, y - width), -np.inf)
    slot = int(np.argmax(beyond))
    if beyond[slot] > cfg.MARGIN_LUAR:
        return THROW_IN, int(slot_team[slot])

    return FREE_KICK, NONE


def tagged_team(events, start, stop, cfg=Config()):
    """Team of the SET PIECES tag around a restart, or NONE (including every 'N/A' throw-in)."""
    best, best_gap = NONE, None
    for event in events:
        if (event["code"] or "").strip().lower() != SET_PIECES:
            continue
        if event["end_frame"] < start or event["start_frame"] > stop + cfg.FPS:
            continue
        team_code = team_of_tag(event["tags"].get("Team"))
        gap = abs(event["start_frame"] - stop)
        if team_code != NONE and (best_gap is None or gap < best_gap):
            best, best_gap = team_code, gap
    return best


def restarts(tracking, ball, intervals, events, halves, home_goal, cfg=Config()):
    """Bagian 6. One Restart per dead stretch: what kind, which team, and how that was decided.

    Team, in the pseudocode's order: the player nearest the ball on the first live frame; failing
    that (ball unseen) the layout trigger; then the Smart Tagging team; else NONE, which leaves
    possession_in_segment to take the first team to touch the ball.
    """
    out = []
    for start, stop in intervals:
        half = next(h for h, first, last in halves if first <= start <= last)
        kind, layout_team = classify_restart(tracking.players, ball, tracking.team, start, stop,
                                             home_goal[half], cfg)
        team_code, source = nearest_team_at(tracking.players, ball, tracking.team, stop,
                                            cfg.R_PZ), "nearest"
        if team_code == NONE:
            team_code, source = layout_team, "layout"
        if team_code == NONE and cfg.PAKAI_SMART_TAGGING:
            team_code, source = tagged_team(events, start, stop, cfg), "smart_tagging"
        if team_code == NONE:
            source = "first_touch"
        out.append(Restart(start, stop, kind, team_code, source))
    return out
