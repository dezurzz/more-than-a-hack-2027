"""The possession model end to end: features -> P(dead) (+ team evidence) -> decoding -> structure.

Team sources:
    team="owner"     default: alive / dead from the Viterbi decoder, team from
                     the mechanical last-owner rule (owner_labels)
    team="model"     the P2-P3 Viterbi team model (P(home | alive) as team evidence)
    team="nearest"   raw team call from the nearest player
dead="none" switches the dead-ball model off (P(dead) NaN, read as alive).
"""

import numpy as np

from mtah import config, deadball, decode, owner, postprocess, team

DEAD_SOURCES = ("model", "none")
TEAM_SOURCES = ("owner", "model", "nearest")


def probabilities(feats, dead="model", team_source="model"):
    """(p_dead, p_home) per frame; p_home is None for the owner rule, which needs none."""
    if dead not in DEAD_SOURCES or team_source not in TEAM_SOURCES:
        raise ValueError("dead must be one of {}, team one of {}".format(DEAD_SOURCES,
                                                                         TEAM_SOURCES))
    n = len(feats["ball_visible"])
    p_dead = deadball.p_dead(feats) if dead == "model" else np.full(n, np.nan)
    if team_source == "owner":
        return p_dead, None
    p_home = team.p_home(feats) if team_source == "model" else team.p_home_from_nearest(feats)
    return p_dead, p_home


def owner_labels(feats, p_dead, radius=None, confirm=None, to_dead=None, from_dead=None,
                 structure=True, runs=None):
    """Team from the mechanical last-owner rule, dead ball from the Viterbi alive / dead decoding.

    The decoder only decides alive vs D: it gets P(home) = 1, so "alive" scores log(1 - P(dead))
    with no team term. (P(home) = 0.5 would charge every alive frame log 0.5 against D and inflate
    the dead share: 43.4% instead of ~40% in the first run.) The alive frames then take the
    owner rule's team (mtah/owner.py).

    runs: half-open spans of the halves; None = BAR-ATH's graded halves (external matches pass theirs).
    """
    radius = config.OWNER_RADIUS if radius is None else radius
    confirm = config.OWNER_CONFIRM if confirm is None else confirm
    to_dead = config.TO_DEAD_COST if to_dead is None else to_dead
    from_dead = config.FROM_DEAD_COST if from_dead is None else from_dead
    alive_dead = labels(p_dead, np.ones(len(p_dead)), switch=min(to_dead, from_dead),
                        to_dead=to_dead, from_dead=from_dead, structure=False, runs=runs)
    carried = None
    if config.OWNER_MAX_REL_SPEED is not None:
        carried = feats["ball_rel_speed"] < config.OWNER_MAX_REL_SPEED
    touch = (config.OWNER_TOUCH_RADIUS, config.OWNER_TOUCH_LOOKBACK)
    confirm_from = None
    if config.OWNER_SHORT_CONFIRM is not None:
        confirm_from = owner.confirm_from_counts(feats["home_count"], feats["away_count"], confirm,
                                                 config.OWNER_SHORT_CONFIRM)
    hold = None if config.OWNER_HOLD_MIN is None else (config.OWNER_HOLD_MIN, config.OWNER_HOLD_MAX)
    taker = None
    if config.OWNER_TAKER_FROM_TAGS and "setpiece_team" in feats:
        tagged = np.nan_to_num(feats["setpiece_team"], nan=0.0)
        taker = np.where(tagged > 0, 0, np.where(tagged < 0, 1, -1))   # +1 home -> owner code 0
    fallback = None
    if config.OWNER_FALLBACK_RADIUS is not None:
        fallback = (config.OWNER_FALLBACK_RADIUS, config.OWNER_FALLBACK_AFTER)
    forced = None
    if config.KEEPER_CATCH_RADIUS is not None:
        forced = owner.keeper_catches(feats, config.KEEPER_CATCH_RADIUS, config.KEEPER_CATCH_MIN_GAP,
                                      config.KEEPER_CATCH_COVER, config.PENALTY_BOX[1] / 2.0,
                                      config.PENALTY_BOX[0])
    if config.OWNER_RESET_AT_RESTART and config.OWNER_TAKER_THROW_IN and "setpiece_type" in feats:
        first = owner.owner_by_segment(feats["nearest_team"], feats["nearest_dist"],
                                       alive_dead == "H", radius, confirm, carried, *touch,
                                       confirm_from, hold, taker, forced, fallback)
        taker = owner.throw_in_takers(first, alive_dead == "H", feats["setpiece_type"], taker)
    if config.OWNER_RESET_AT_RESTART:
        owners = owner.owner_by_segment(feats["nearest_team"], feats["nearest_dist"],
                                        alive_dead == "H", radius, confirm, carried, *touch,
                                        confirm_from, hold, taker, forced, fallback)
    else:
        owners = owner.last_owner(feats["nearest_team"], feats["nearest_dist"], radius, confirm,
                                  carried, *touch, confirm_from, hold)
    if config.BLIP_MAX_LEN is not None:
        owners = owner.drop_loose_blips(owners, feats["nearest_dist"], alive_dead == "H",
                                        config.BLIP_MAX_LEN, config.BLIP_MIN_MEDIAN_DIST)
    team = owner.owner_labels(owners)
    out = np.where(alive_dead == "D", "D", np.where(alive_dead == "X", "X", team))
    if config.RESTART_KICK_SPEED is not None:
        out = hold_dead_until_kick(out, feats["ball_speed"], config.RESTART_KICK_SPEED,
                                   config.RESTART_MAX_HOLD)
    if config.END_KICK_SPEED is not None:
        out = end_dead_at_kick(out, feats["ball_speed"], config.END_KICK_SPEED,
                               config.END_KICK_LOOKBACK)
    return postprocess.apply_match_structure(out) if structure else out


def hold_dead_until_kick(out, ball_speed, kick_speed, max_hold):
    """Extend each dead-ball stretch until the restart kick: the first frame the ball moves faster
    than `kick_speed` m/s, at most `max_hold` frames past the decoded end.

    A stoppage ends with a kick or a throw, but before it the ball is carried, rolled to the taker
    and placed while players walk, which the dead-ball rules read as live play. Frames
    with no ball speed do not count as a kick. Only alive frames are relabelled, never X.
    """
    out = out.copy()
    dead = out == "D"
    edges = np.flatnonzero(dead[:-1] & ~dead[1:]) + 1          # first alive frame after a D run
    fast = np.nan_to_num(ball_speed, nan=0.0) > kick_speed
    for start in edges:
        stop = min(len(out), start + max_hold)
        kick = np.flatnonzero(fast[start:stop])
        end = start + (int(kick[0]) if len(kick) else 0)          # no kick in reach: keep as decoded
        span = slice(start, end)
        alive = np.isin(out[span], ("H", "A"))
        out[span] = np.where(alive, "D", out[span])
    return out


def end_dead_at_kick(out, ball_speed, kick_speed, lookback):
    """End each dead-ball stretch at the restart kick when the decoder ended it later.

    Within the last `lookback` frames of a D run, the first frame the ball moves faster than
    `kick_speed` m/s is the restart; the frames from there to the decoded end take the team of the
    first alive frame after the run (who restarted). The decoder ends D ~2 s after the restart,
    once the players have sped up.
    """
    out = out.copy()
    dead = out == "D"
    ends = np.flatnonzero(dead[:-1] & ~dead[1:]) + 1
    fast = np.nan_to_num(ball_speed, nan=0.0) > kick_speed
    for end in ends:
        if out[end] not in ("H", "A"):
            continue
        start = end
        while start > 0 and dead[start - 1] and end - start < lookback:
            start -= 1
        kick = np.flatnonzero(fast[start:end])
        if len(kick):
            out[start + int(kick[0]):end] = out[end]
    return out


def labels(p_dead, p_home, switch=config.SWITCH_COST, to_dead=config.TO_DEAD_COST,
           from_dead=config.FROM_DEAD_COST, structure=True, runs=None):
    """H/A/D per graded frame via Viterbi; with structure=True also X ungraded and D pre-kickoff.

    runs: half-open spans to decode; None = BAR-ATH's graded halves.
    """
    costs = decode.transition_costs(switch, to_dead, from_dead)
    out = decode.decode(decode.scores_from_probabilities(p_dead, p_home), costs,
                        postprocess.graded_runs() if runs is None else runs)
    return postprocess.apply_match_structure(out) if structure else out
