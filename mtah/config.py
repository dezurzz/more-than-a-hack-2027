"""Every parameter of the L2Kotak pipeline, in one place.

Changing a number here changes the submission, so each one says why it has its value. Importing
this module also puts the repository root and evaluation/ on sys.path, so the starter modules
(games, starter.baseline, possession_metrics) import the same way from anywhere.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (os.path.join(ROOT, "evaluation"), ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Params
TEAM_NAME = "L2Kotak"  # must match the registration form exactly

# Derived per-frame data. Both folders are gitignored and blocked by tools/hooks/pre-commit.
CACHE_DIR = os.path.join(ROOT, "cache")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, TEAM_NAME + ".csv")

PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

# The second half is graded from 110808 but kicks off at 113742 (FAQ.md): those frames are dead
# by definition. Half-open, like every interval in this package.
SECOND_HALF_PRE_KICKOFF = (110808, 113742)

# Majority-vote half-width of the current method, the reference baseline. 50 is the published
# floor (full-match spellF1 0.2433). Replaced by the P2/P3 model, not tuned.
BASELINE_WINDOW = 50

# -- features (P1) --------------------------------------------------------------------------------
# Kickoff frame of each half (DATA.md, FAQ.md). Used to read each team's attacking direction off
# its kickoff formation, when every player is in his own half.
KICKOFFS = (11185, 113742)
KICKOFF_DIRECTION_FRAMES = 50  # 2 s after kickoff to average the home centroid over

# Ball gaps up to this many frames (0.5 s) are filled linearly for the features: an occlusion that
# short rarely hides a pass. Longer gaps stay NaN, because the ball may have changed hands.
BALL_MAX_INTERP_GAP = 12

# Heading change of the ball over +/- this many frames, ignored when it moved less than
# HEADING_MIN_MOVE metres either side (a nearly still ball has no meaningful direction).
HEADING_LAG = 3
HEADING_MIN_MOVE = 0.3

# A player within this distance of the ball is treated as being on it (control features).
CONTROL_RADIUS = 1.5
# Counting radii for "how many players of each team are near the ball".
NEAR_RADII = (2.0, 5.0)

# Penalty box: depth from the goal line and full width, in metres (Laws of the Game).
PENALTY_BOX = (16.5, 40.32)

# Smart Tagging team values -> +1 home, -1 away. Matched lower case. 'Atletic' is the file's own
# spelling (DATA.md); the correct one is listed too in case the export is ever fixed.
TAG_TEAMS = {"barcelona": 1, "atletic club": -1, "athletic club": -1, "n/a": 0}
# Set-piece Type -> code for the setpiece_type feature. 9 = a type not listed here.
SET_PIECE_TYPES = {"throw-in": 1, "corner": 2, "corner kick": 2, "goal kick": 3, "free kick": 4,
                   "kick off (after goal)": 5, "kick off (start)": 5}
SET_PIECE_OTHER = 9
# Phase codes that get their own tag_<code> feature. Fixed here so feature names do not depend on
# what a file happens to contain; any other code lands in tag_other.
TAG_PHASE_CODES = (
    "build up", "progression", "ball in final third", "ball in the box", "players in the box",
    "creating chances", "long ball", "attacking transition", "defensive transition",
    "defending in attacking third", "defending in middle third", "defending in defensive third",
    "goals",
)

# -- dead-ball model -------------------------------------------
# logit P(dead) = DEAD_BIAS + sum of DEAD_WEIGHTS[rule] * indicator[rule]; weights are priors.
# The ball is alive most of the time. -2.2: with the set-piece rule off, the share of
# Smart Tagging restarts with D around them saturates there (90%, 43 of 48; -2.6 gives 79-85%, -1.8
# only one more), and the dead share is 37.8%, the middle of the plausible 30-45%.
DEAD_BIAS = -2.2
DEAD_WEIGHTS = {
    "out": 6.0,            # ball beyond a line: dead by definition (alone: P(dead) 0.95)
    "slow_play": 2.0,      # everyone walks at a stoppage
    "ball_still": 1.5,     # ball placed for a restart
    "lost_and_slow": 1.5,  # ball off camera during a stoppage
    # OFF: the windows are 35 s of mostly live play; the rule made false D in open play
    # (public D precision 0.80 with it, 1.00 without) and the model finds 90% of the tagged
    # restarts without it. Prior when on: 1.5.
    "setpiece": 0.0,
    "kickoff": 4.0,        # both teams in their own half: impossible in open play
    # The ball vanished heading out of the pitch: its last motion, carried on
    # for BALL_EXIT_HORIZON s, ends past a line. Restart recall 45 -> 46 of 48 at a similar dead
    # share (47 before the goal-mouth exclusion, one of them a keeper's catch); the lowest weight
    # that reaches it.
    "exit": 2.0,
}
OUT_MARGIN = 0.5              # metres beyond a line before the ball counts as out
# Set from the observed speeds: all_speed_file has a median of 1.19 in D and 2.05-2.47 in play (unit unconfirmed). The ramp is 1 at or below the first value and 0 at or above the
# second; both are plain numbers between those medians, not optimised.
SLOW_PLAY_RAMP = (1.0, 2.0)
BALL_STILL_SPEED = 1.0         # m/s, our own finite-difference ball_speed
BALL_STILL_FRAMES = 25         # 1 s of the ball not moving
LOST_BALL_FRAMES = 50          # 2 s without the ball
KICKOFF_OWN_HALF_FRAC = 0.95   # fraction of each team in its own half
# Rule "exit": a ball gap of at least BALL_EXIT_MIN_GAP frames whose last BALL_EXIT_FIT tracked
# positions, carried on for BALL_EXIT_HORIZON s, end more than OUT_MARGIN past a line; it fires on
# the first BALL_EXIT_COVER frames of the gap (a throw-in takes 5-10 s: at 125 frames the gain
# was gone). Recall is flat for horizons 1.0-2.0 s and gaps of 12-50 frames: the shortest horizon.
BALL_EXIT_HORIZON = 1.0
BALL_EXIT_MIN_GAP = 25
BALL_EXIT_COVER = 250
BALL_EXIT_FIT = 5
# A dead-ball stretch lasts until the restart kick: the first frame the ball is faster than
# RESTART_KICK_SPEED m/s, at most RESTART_MAX_HOLD frames past the decoded end. None = off.
RESTART_KICK_SPEED = None
RESTART_MAX_HOLD = 500
# The other way: end a dead-ball stretch at the restart kick, the first frame within its last
# END_KICK_LOOKBACK frames with the ball faster than END_KICK_SPEED m/s. None = off.
END_KICK_SPEED = None
END_KICK_LOOKBACK = 100
# An exit over a goal line within the penalty box's width does not count: a keeper may catch the
# ball there, and a caught ball vanishes the same way.
# 0 = every line counts.
BALL_EXIT_GOAL_HALF_WIDTH = PENALTY_BOX[1] / 2.0

# -- team model ---------------------------------------------------
W_DIST = 0.4                   # per metre of dist_diff
DIST_CLIP = 5.0                # metres; beyond this the nearer team is clear anyway
W_CONTROL = 2.5                # for a full CONTROL_SATURATION frames on the ball
CONTROL_SATURATION = 25        # frames (1 s) after which control stops growing
CONTROL_MAX_REL_SPEED = 3.0    # m/s; faster than this the ball is not travelling with the player
W_NEAR = 0.3                   # per player of local-numbers advantage within 2 m
# Smart Tagging terms, from the agreement of each tag with the
# model's own tag-free calls over the whole match (no labels involved).
W_RESTART = 1.0
# Restarting team's bonus over the seconds after a set-piece window starts: (end second,
# multiplier). The multipliers are the measured lifts by time bucket (22.1 / 12.5 / 9.2) divided by
# the first. After 15 s the lift is ~0 and then negative (-8.7 after 25 s: the opponent has the
# ball by then), so there is no bonus.
RESTART_PROFILE = ((5.0, 1.0), (10.0, 0.55), (15.0, 0.4))
# OFF: prior when on is 0.3.
W_TAG = 0.0
# tag feature name -> weight (+: its Team is the team in possession). Weight = the tag's lift
# over chance divided by the largest lift (build_up, +31.9 points), rounded. Only the attacking
# side of each mirrored pair is listed: tag_defending_in_* have the same intervals with the
# opposite team, so adding both would count one piece of evidence twice. Tags with lift under
# 10 points (creating_chances, players_in_the_box) are left out.
TAG_DIRECTIONS = {
    "tag_build_up": 1.0,
    "tag_progression": 0.72,
    "tag_attacking_transition": 0.64,
    "tag_ball_in_final_third": 0.55,
    "tag_goals": 0.47,
    "tag_long_ball": 0.44,
    "tag_ball_in_the_box": 0.43,
}
# Baseline team source only: raw team evidence taken from the nearest player, as P(home) for home / away.
BASELINE_TEAM_P = 0.9

# -- decoding (P3) ---------------------------------------------------------------------------------
# Switching penalties in log-score units, from sweeps over the penalty. The
# possession count falls smoothly with the penalty, so there is no plateau: 40 is the middle of
# the range where possessions, mean duration and dead stretches are all plausible (penalty 20-60,
# D costs equal to it), not the penalty with the best public spellF1 (the D costs must sum to
# more than the switch).
SWITCH_COST = 40.0     # H <-> A (Viterbi team model only; the default owner rule does not use it)
# alive -> D and D -> alive: 30. Between 10 and 30 they catch the same restarts; 30
# puts the full-match number of stoppages at 109, the middle of the plausible 80-140.
TO_DEAD_COST = 30.0
FROM_DEAD_COST = 30.0

# -- last-owner team rule (mtah/owner.py) -----------------------------------------------------
# The ball belongs to the team of the last player who stayed nearest to it, within OWNER_RADIUS
# metres, for OWNER_CONFIRM consecutive frames. The default team source: at the
# same full-match possession count it agrees ~3-5 points more with the independent Smart Tagging
# phases than the Viterbi team model. The values used are 1 m and 10 frames (0.4 s).
OWNER_RADIUS = 1.0
OWNER_CONFIRM = 10
# Reset the owner at every restart (owner.owner_by_segment): the owner before a dead ball says
# nothing about the possession after it, and carrying it over created a spurious short possession
# at the start of every restart until the new owner was confirmed. With the reset the
# confirmation can be 10 frames (0.4 s) and still give a plausible possession count (251).
OWNER_RESET_AT_RESTART = True
# A change of owner is dated back to the new team's first touch (a player of it within this many
# metres of the ball) after the old owner's last touch, looking back at most OWNER_TOUCH_LOOKBACK
# frames (10 s). The truth flips at the first touch of the team that then controls the ball, not
# when control is established. None = off.
OWNER_TOUCH_RADIUS = None
OWNER_TOUCH_LOOKBACK = 250
# m/s; when set, a frame counts towards ownership only if the ball moves with the nearest player
# (relative speed below this). None = distance alone.
OWNER_MAX_REL_SPEED = None
# Frames needed to take the ball from a team that has fewer than 11 players tracked while the other
# side is complete. A missing ball carrier hands the ball to the opponent marking him; in such frames the team call agrees far less with the Smart Tagging phases.
# None = same as OWNER_CONFIRM.
OWNER_SHORT_CONFIRM = None
# A ball that vanishes from the tracking right after OWNER_HOLD_MIN frames within OWNER_RADIUS of a
# player stays with that player's team for up to OWNER_HOLD_MAX frames (a keeper holding the ball
# is not tracked). None = off.
# A restart tagged with a team (corner, free kick, goal kick; throw-ins are N/A) starts owned by
# that team until another owner is confirmed: the taker touches the ball first.
OWNER_TAKER_FROM_TAGS = True
# Throw-ins carry no team in the tags: the taker is the opposite of the last owner before the stoppage
# (CV_PLAN N1). Off by default until gated.
OWNER_TAKER_THROW_IN = False
# A ball lost from the tracking inside a penalty box with that box's keeper within
# KEEPER_CATCH_RADIUS m: the keeper caught it, his team owns it (owner.keeper_catches).
# Gap of at least KEEPER_CATCH_MIN_GAP frames, covering its first KEEPER_CATCH_COVER. None = off.
# When nobody has been within OWNER_RADIUS of the ball for OWNER_FALLBACK_AFTER frames, a player
# within OWNER_FALLBACK_RADIUS can be confirmed as owner too (owner.last_owner). None = off.
OWNER_FALLBACK_RADIUS = None
OWNER_FALLBACK_AFTER = 75
KEEPER_CATCH_RADIUS = None
KEEPER_CATCH_MIN_GAP = 12
KEEPER_CATCH_COVER = 250
OWNER_HOLD_MIN = None
OWNER_HOLD_MAX = 250
# A possession of at most BLIP_MAX_LEN frames between two possessions of the same other team is
# handed back to that team when the nearest player stayed farther than BLIP_MIN_MEDIAN_DIST metres
# from the ball (median over the possession): no one controlled it. None = off.
BLIP_MAX_LEN = None
BLIP_MIN_MEDIAN_DIST = 1.5

# -- experiment overrides ----------------------------------------------------------------------
# `--set NAME=VALUE` (or NAME.key=VALUE for a dict entry) on run.py and the sweeps changes one of
# these for that run only, so an ablation needs no edit to this file. Only model parameters are
# listed: the features are cached on this file's bytes and must not change through a flag.
OVERRIDABLE = (
    "DEAD_BIAS", "DEAD_WEIGHTS", "OUT_MARGIN", "SLOW_PLAY_RAMP", "BALL_STILL_SPEED",
    "BALL_STILL_FRAMES", "LOST_BALL_FRAMES", "BALL_EXIT_HORIZON", "BALL_EXIT_MIN_GAP",
    "BALL_EXIT_COVER", "BALL_EXIT_GOAL_HALF_WIDTH", "RESTART_KICK_SPEED", "RESTART_MAX_HOLD",
    "END_KICK_SPEED", "END_KICK_LOOKBACK", "KICKOFF_OWN_HALF_FRAC", "W_DIST", "DIST_CLIP",
    "W_CONTROL", "CONTROL_SATURATION", "CONTROL_MAX_REL_SPEED", "W_NEAR", "W_RESTART",
    "RESTART_PROFILE", "W_TAG", "TAG_DIRECTIONS", "BASELINE_TEAM_P", "SWITCH_COST",
    "TO_DEAD_COST", "FROM_DEAD_COST", "OWNER_RADIUS", "OWNER_CONFIRM",
    "OWNER_MAX_REL_SPEED", "OWNER_SHORT_CONFIRM", "OWNER_HOLD_MIN", "OWNER_HOLD_MAX",
    "OWNER_TAKER_FROM_TAGS", "OWNER_TAKER_THROW_IN", "TEAM_SOURCE", "BALL_SOURCE", "KEEPER_CATCH_RADIUS", "KEEPER_CATCH_MIN_GAP", "KEEPER_CATCH_COVER",
    "OWNER_FALLBACK_RADIUS", "OWNER_FALLBACK_AFTER",
    "BLIP_MAX_LEN", "BLIP_MIN_MEDIAN_DIST", "OWNER_RESET_AT_RESTART", "OWNER_TOUCH_RADIUS",
    "OWNER_TOUCH_LOOKBACK", "LEARNED_TEAM_SOURCE", "LEARNED_DEAD_UNION",
)


def override(spec):
    """Apply one 'NAME=VALUE' or 'NAME.key=VALUE' (VALUE a Python literal) to this module."""
    import ast
    name, _, text = spec.partition("=")
    top, _, key = name.partition(".")
    if top not in OVERRIDABLE or not text:
        raise ValueError("--set {!r}: NAME must be one of {}".format(spec, ", ".join(OVERRIDABLE)))
    value = ast.literal_eval(text)
    module = sys.modules[__name__]
    if key:
        getattr(module, top)[key] = value
    else:
        setattr(module, top, value)


def apply_cli_overrides(argv):
    """Apply every `--set X` in argv; returns the specs applied. Call before building defaults."""
    applied = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == "--set"]
    for spec in applied:
        override(spec)
    return applied


# -- goalkeeper repair (mtah/teamfix.py) ------------------------------------------------------
# The tracked player nearest a goal line, within this many metres of it, takes the defending team.
# None = off (the ATD's own team for every slot). OFF: it showed that this per-frame rule makes
# the team composition worse (11 v 11 from 42.0% to 37.8% of frames), because the nearest player to
# a goal line is often not the keeper. A feature parameter: not in OVERRIDABLE, since the cached
# features depend on it.
KEEPER_MAX_DIST = None
# Drop "ghost" stretches of a player slot (perfectly linear motion for this many frames, a track the
# tracker holds rather than sees: features.drop_ghosts) before any feature uses the players.
# A feature parameter like KEEPER_MAX_DIST. None = off.
# Where each slot's team comes from: "atd" (the ATD label, keepers repaired) or "colour" (outfield
# slots decoded from the shirt colour, cache/cv/shirt_*.npz made by scripts/cv_run.py; CV_PLAN V1).
# Part of the feature cache key, so it may be overridden (--set TEAM_SOURCE=colour).
TEAM_SOURCE = "colour"
# Where the ball comes from: "atd" or "fused" (ATD plus our own detector filling its gaps,
# cache/cv/ball_det_*.npz from scripts/cv_run.py --stage ball_det; CV_PLAN V2).
BALL_SOURCE = "atd"
GHOST_MIN_FRAMES = None
# An established keeper slot standing in the opponents' half is the OTHER keeper written into the
# wrong slot: it gets the other team (teamfix.swap_keeper_slots). A feature parameter.
KEEPER_SWAP = True

# Draws for the scorer's block bootstrap. 1000 is what EVALUATION.md uses for a final board.
BOOTSTRAP_N = 1000

# Full-match plausibility ranges. Rough football estimates,
# not truth: a prediction outside them is suspect, one inside them is not thereby right.
PLAUSIBLE = {
    "possessions": (250, 450),
    "mean_duration_s": (8.0, 20.0),
    "dead_share": (0.30, 0.45),
    "dead_stretches": (80, 140),
    "home_share": (0.55, 0.70),
}

# -- learned model (development variant, needs mtah/weights) --------------------
# Five temporal networks (seeds 0-4, mapping T 2.5 s, trained on damaged IDSSE) in the portable
# form of learn.save_portable; infer.load_ensemble refuses a file whose sha256 differs.
LEARNED_WEIGHTS = {
    "tcn_t25_s0.pt": "1a94a77fd3da5ddfb774d9fa9fe6e7509755b18c94b26201cb2b7d2fb6b4a6fb",
    "tcn_t25_s1.pt": "9c92f10110006914ac1d257e7cd5f0559f98cc360d809565a63fec6438d7a2fa",
    "tcn_t25_s2.pt": "eb63573940e7a69d251784d0ce45a2745c690445914078213588857537db6956",
    "tcn_t25_s3.pt": "303f7e1572b7e46a2e512b5b17cab0951143df9a26dc9e22295f9489b83651d0",
    "tcn_t25_s4.pt": "fed061d46baba8bd488b1e7b6ec03479f4c364052a6c5afe8a7082976573bd21",
}
LEARNED_WEIGHTS_DIR = os.path.join(ROOT, "mtah", "weights")
LEARNED_DECODER = (2.0, 4.0, 4.0)     # switch, to_dead, from_dead
LEARNED_PROBABILITY_DECIMALS = 2      # probabilities are rounded before the Viterbi (CV_PLAN 4.4)
# Teams in the network's own features: "atd" (fixed slot teams, as trained) or "colour" (per-frame
# shirt-colour teams of the rules). Read by infer.labels only; part of no cache key.
LEARNED_TEAM_SOURCE = "colour"

# Dead ball of the default: D wherever the learned labels or the dead-ball rules (`--method model`) say D,
# teams from the learned labels elsewhere. Applied by run.py before the match structure.
LEARNED_DEAD_UNION = True

# -- three-model ensemble (development variant) --------------------
# Equal-weight mean of the per-frame H / A / D probabilities of ours (the five networks above, shirt-colour
# teams, no dead-ball union), Wira's initial TCN and Dzaky's hybrid MS-TCN, decoded once with our decoder.
# Their inference code is vendored in mtah/members (ATTRIBUTION.md in each); mtah.ensemble combines.
ENSEMBLE_MEMBERS = ("aji", "wira", "dzaky")
ENSEMBLE_DECODER = (2.0, 4.0, 4.0)    # switch, to_dead, from_dead: as LEARNED_DECODER
ENSEMBLE_PROBABILITY_DECIMALS = 2     # the mean is rounded before the Viterbi, as for the networks
# The teammates' checkpoints, checked against their sha256 before they are loaded.
ENSEMBLE_WEIGHTS = {
    "wira/tcn1800.joblib": "b20d1e6b070fd2dabc4487cb2037e2f14f8e3aa12b9e9776dab95a5221787c05",
    "dzaky/tcn2.pt": "38f956a3a8736d313df03637212f00a127faaa1a42c3e3194327a9e9afa58294",
    "dzaky/tcn6.pt": "9d0a17fffe831e0d0b38bf0e2cfd9e421f1fdf73ce6110cd5f3fb4bd97301ca3",
}
ENSEMBLE_WEIGHTS_DIR = os.path.join(ROOT, "mtah", "members")

# -- final model (`python run.py`) ------------------------------
# Equal mean (per frame, NaN-aware) of model A (five networks) and model B (MS-TCN), both
# probabilities rounded as in the selection run, decoded with Viterbi 2 / 4 / 4 per half. Both checkpoints are
# sha256-checked before loading (they are torch pickles). Member B's second half starts at the kickoff.
FINAL_DECODER = (2.0, 4.0, 4.0)
FINAL_AJI_WEIGHTS_DIR = os.path.join(ROOT, "candidates", "weights_robust3")
FINAL_AJI_WEIGHTS = {
    "robust3_t25_s0.pt": "417d0a5e8e0a2582ffe47560a254da97af622a2f033d130c871e3d435ede50b6",
    "robust3_t25_s1.pt": "6fab869dc00edf39121b7aeb16d710a11ed95e16dc5e99678c28ec3490d23d49",
    "robust3_t25_s2.pt": "eb1c7df39250d1086f5aface5fda59cfe67cbadd9314d0125cffe1a1fa74751b",
    "robust3_t25_s3.pt": "a515df88eb1cfaaf0a4e7e736a5d0e0d8f35bc176eea2924a09004e745676110",
    "robust3_t25_s4.pt": "fa2890710029ae96f30f18c80c2a97eff71268905c0ad57eddd0bcc285c43ec7",
}
# Member B (model B) is the MS-TCN trained with five seeds on DFL + SkillCorner +
# PFF World Cup games 32-63 (tcn11P), averaged. The previous version used the single checkpoint tcn11 as member B and
# stays reproducible with `python run.py --method final_c`.
FINAL_DZAKY_WEIGHTS = {
    "tcn11P_s0.pt": "34e5d766404c17527193b582629a69b0eed03e265b0f9eba16a759f7e78fe5a6",
    "tcn11P_s1.pt": "242c9a7ac870fd79c9a30c7539b7ed799f1d16b116919e04de0f019d4995f0e2",
    "tcn11P_s2.pt": "d117d8032e505947bc57b9055bc8d347b1a55cd3f32a1b24365eff3cb96feb4b",
    "tcn11P_s3.pt": "10e17627c3f426a832395336f924c018aadebb345c1c56b7ad0a412f1b5a21da",
    "tcn11P_s4.pt": "ba63ede8dacc76534648b2b47185636d5c75b4007e518fb2f809ad8d14cb12c2",
}
FINAL_C_DZAKY_WEIGHTS = {
    "tcn11.pt": "0a01f28c4f458aedc7fcd35d1933c0ac3f44671d9a4a8032be17c19ab33d923b",
}

