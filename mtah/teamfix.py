"""Per-frame team of every tracked slot, repairing goalkeepers that come back on a new slot.

The ATD gives each slot one fixed team. When a keeper is lost and re-acquired he can come back as a
different, newly created slot attached to the wrong team (a keeper drawn as an opponent right after
a catch): a phantom opponent next to the ball, read as the other team having it.

Two steps, no ground truth:
1. find_keeper_slots: each team has one established keeper slot, the slot that is the tracked player
   nearest its own goal line (within 20 m) for most of its life.
2. keeper_teams: in a frame where that established keeper is NOT tracked, the tracked player nearest
   the goal line (within `max_dist`) is taken to be its replacement and gets the defending team.
   While the established keeper is tracked nobody else is touched.

A rule that relabelled the nearest player to a goal line in every frame was tried first and made
the team composition worse: it hit outfield players.
"""

import numpy as np


def find_keeper_slots(x_m, home_goal_x, away_goal_x, near=20.0, min_share=0.5):
    """(home_keeper_slot, away_keeper_slot): the slot nearest each goal line, within `near` m, on at
    least `min_share` of the frames it is tracked; None for a side with no such slot."""
    present = ~np.isnan(x_m)
    usable = np.isfinite(home_goal_x)
    rows = np.arange(len(x_m))
    tracked = np.maximum((present & usable[:, None]).sum(axis=0), 1)
    found = []
    for goal_x in (home_goal_x, away_goal_x):
        distance = np.where(present, np.abs(x_m - goal_x[:, None]), np.inf)
        nearest = distance.argmin(axis=1)
        within = usable & (distance[rows, nearest] < near)
        share = np.bincount(nearest[within], minlength=x_m.shape[1]) / tracked
        best = int(share.argmax())
        found.append(best if share[best] >= min_share else None)
    return tuple(found)


def swap_keeper_slots(x_m, teams, keeper_slots, home_goal_x, away_goal_x, half_length):
    """Give an established keeper slot the other team while it stands in the other team's half.

    The two keeper slots swap identities: the keeper the camera sees is written into the other
    keeper's slot (for instance the home keeper drawn as the away team, or the reverse). A real keeper is never in the opponents' half in open play; the away keeper
    slot is in the home penalty box on 11% of its frames, and almost never anywhere in between
. teams is changed in place and returned.
    """
    rows = np.arange(len(x_m))
    for keeper, other_goal, other_team in ((keeper_slots[0], away_goal_x, 1),
                                           (keeper_slots[1], home_goal_x, 0)):
        if keeper is None:
            continue
        wrong = np.abs(x_m[:, keeper] - other_goal) < half_length  # NaN compares False
        teams[rows[wrong], keeper] = other_team
    return teams


def keeper_teams(x_m, team_of_slot, home_goal_x, away_goal_x, max_dist, keeper_slots):
    """(n_frames, n_slots) int8 team per slot per frame.

    x_m: (n, slots) pitch x in metres, NaN where untracked. home_goal_x / away_goal_x: (n,) x of the
    goal line each team defends in that frame, NaN where unknown (then nothing changes).
    keeper_slots: (home, away) established keeper slots from find_keeper_slots; a side with None
    has no established keeper, so its replacement rule applies in every frame.
    """
    n = len(x_m)
    teams = np.tile(np.asarray(team_of_slot, dtype=np.int8), (n, 1))
    present = ~np.isnan(x_m)
    rows = np.arange(n)
    for goal_x, team, keeper in ((home_goal_x, 0, keeper_slots[0]),
                                 (away_goal_x, 1, keeper_slots[1])):
        distance = np.where(present, np.abs(x_m - goal_x[:, None]), np.inf)
        slot = distance.argmin(axis=1)
        replace = np.isfinite(goal_x) & (distance[rows, slot] < max_dist)
        if keeper is not None:
            replace &= ~present[:, keeper]            # the established keeper is tracked: leave it
        teams[rows[replace], slot[replace]] = team
    return teams
