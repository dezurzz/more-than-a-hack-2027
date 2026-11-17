"""Per-frame features from the tracking, Smart Tagging and homography (plan P1).

build() returns {name: (n_frames,) float64 array}, NaN where a feature has no value (usually: no
ball). Distances are in metres. *_speed_file features are the speed column of the ATD, in whatever
unit the file uses; ball_speed and ball_rel_speed are our own finite differences, in m/s.

The feature names are the contract with the model code and scripts/feature_report.py. Renaming or
redefining one is a breaking change: bump FEATURE_VERSION so stale caches are not reused.

Signs: dist_diff > 0 and every *_team feature > 0 mean HOME. Team codes inside arrays follow
starter.baseline: 0 home, 1 away.
"""

import hashlib
import os
import re
import warnings

import numpy as np

from mtah import config, io_atd, postprocess, teamfix
from games import GAME, GAMES, require  # noqa: E402
from starter import load_homography, load_smart_tagging  # noqa: E402
from starter.baseline import AWAY, HOME  # noqa: E402

# Params
FEATURE_VERSION = 3  # 2: per-frame teams with the goalkeepers repaired; 3: keeper distances
CHUNK = 20000  # frames per block in the frames x slots computations


# -- small array helpers ---------------------------------------------------------------------------

def run_lengths(values, ignore=None):
    """Length of the constant run each element belongs to; 0 where the value equals `ignore`."""
    values = np.asarray(values)
    n = len(values)
    if not n:
        return np.zeros(0)
    starts = np.flatnonzero(np.concatenate(([True], values[1:] != values[:-1])))
    lengths = np.diff(np.concatenate((starts, [n])))
    out = np.repeat(lengths, lengths).astype(np.float64)
    if ignore is not None:
        out[values == ignore] = 0.0
    return out


def interpolate_short_gaps(points, max_gap):
    """Copy of points (n, k) with NaN runs of at most max_gap rows filled linearly.

    Only gaps with data on both sides are filled; gaps touching either end stay NaN.
    """
    out = points.copy()
    missing = np.isnan(points[:, 0]).astype(np.int8)
    edges = np.diff(np.concatenate(([0], missing, [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        length = stop - start
        if length > max_gap or start == 0 or stop == len(points):
            continue
        before, after = points[start - 1], points[stop]
        steps = np.arange(1, length + 1, dtype=np.float64)[:, None] / (length + 1)
        out[start:stop] = before + (after - before) * steps
    return out


def central_difference(values, fps):
    """d(values)/dt by central difference along axis 0; NaN at both ends and next to gaps."""
    out = np.full(values.shape, np.nan)
    out[1:-1] = (values[2:] - values[:-2]) * (fps / 2.0)
    return out


def heading_change(points, lag, min_move):
    """Angle in degrees between the ball's path over the last `lag` frames and the next `lag`.

    Near 0 for a ball running straight, near 180 for one played back the way it came: a touch.
    NaN where the ball moved less than min_move metres on either side.
    """
    before = np.full(points.shape, np.nan)
    after = np.full(points.shape, np.nan)
    before[lag:] = points[lag:] - points[:-lag]
    after[:-lag] = points[lag:] - points[:-lag]
    norm_before = np.hypot(before[:, 0], before[:, 1])
    norm_after = np.hypot(after[:, 0], after[:, 1])
    with np.errstate(invalid="ignore", divide="ignore"):  # still ball: 0/0, masked below
        cosine = (before * after).sum(axis=1) / (norm_before * norm_after)
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    angle[(norm_before < min_move) | (norm_after < min_move)] = np.nan
    return angle


def attack_signs(home_centroid_x, length, runs, kickoffs, frames=config.KICKOFF_DIRECTION_FRAMES):
    """+1 on each run where home attacks towards x = length, -1 where towards x = 0, NaN elsewhere.

    Read off the kickoff formation: at kickoff every player stands in his own half, so the side of
    the home centroid is the side home defends.
    """
    out = np.full(len(home_centroid_x), np.nan)
    for (start, stop), kickoff in zip(runs, kickoffs):
        centre = np.nanmean(home_centroid_x[kickoff:kickoff + frames])
        if not np.isnan(centre):
            out[start:stop] = 1.0 if centre < length / 2.0 else -1.0
    return out


def _slug(code):
    return re.sub(r"[^a-z0-9]+", "_", code.strip().lower()).strip("_")


# -- feature groups --------------------------------------------------------------------------------

def _ball_features(tracking, out):
    """Ball position, motion and geometry. Returns (filled positions, velocity), both (n, 2)."""
    fps = GAMES[GAME]["fps"]
    length, width = tracking.pitch
    raw = tracking.ball_m()
    visible = ~np.isnan(raw[:, 0])
    filled = interpolate_short_gaps(raw, config.BALL_MAX_INTERP_GAP)
    x, y = filled[:, 0], filled[:, 1]

    out["ball_visible"] = visible.astype(np.float64)
    out["ball_interp"] = (~visible & ~np.isnan(x)).astype(np.float64)
    out["ball_gap_len"] = run_lengths(~visible, ignore=False)
    out["ball_x"], out["ball_y"] = x, y
    out["ball_speed_file"] = np.where(visible, tracking.ball[:, 2], np.nan)

    velocity = central_difference(filled, fps)
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    out["ball_speed"] = speed
    out["ball_accel"] = central_difference(speed, fps)
    out["ball_heading_change"] = heading_change(filled, config.HEADING_LAG,
                                                config.HEADING_MIN_MOVE)

    # Signed distances to the lines: negative once the ball is outside the pitch.
    goal_line = np.minimum(x, length - x)
    touch_line = np.minimum(y, width - y)
    out["ball_goal_line_dist"] = goal_line
    out["ball_touch_line_dist"] = touch_line
    out["ball_line_dist"] = np.minimum(goal_line, touch_line)
    box_depth, box_width = config.PENALTY_BOX
    in_box = (goal_line >= 0) & (goal_line < box_depth) & (np.abs(y - width / 2.0) < box_width / 2.0)
    out["ball_in_box"] = np.where(np.isnan(x), np.nan, in_box.astype(np.float64))
    out["ball_centre_dist"] = np.hypot(x - length / 2.0, y - width / 2.0)
    return filled, velocity


def _ball_player_features(tracking, players, ball, out, teams):
    """Who is near the ball, how near, and how long the same team has been on it.

    teams: (n_frames, n_slots) team of each slot per frame (teamfix.keeper_teams)."""
    fps = GAMES[GAME]["fps"]
    filled, ball_velocity = ball
    n = tracking.n_frames
    player_velocity = central_difference(players, fps).astype(np.float32)

    names = ["dist_home", "dist_away", "nearest_dist", "nearest_team", "nearest_speed_file",
             "ball_rel_speed"]
    for radius in config.NEAR_RADII:
        names += ["n_home_{:g}m".format(radius), "n_away_{:g}m".format(radius)]
    result = {name: np.full(n, np.nan) for name in names}

    for start in range(0, n, CHUNK):
        stop = min(n, start + CHUNK)
        rows = np.arange(stop - start)
        ball_xy = filled[start:stop]
        has_ball = ~np.isnan(ball_xy[:, 0])
        distance = np.hypot(players[start:stop, :, 0] - ball_xy[:, None, 0],
                            players[start:stop, :, 1] - ball_xy[:, None, 1])
        distance = np.where(np.isnan(distance), np.inf, distance)

        slot = distance.argmin(axis=1)
        nearest = distance[rows, slot]
        found = has_ball & np.isfinite(nearest)
        block = slice(start, stop)
        home = teams[block] == HOME

        for name, side in (("dist_home", home), ("dist_away", ~home)):
            closest = np.where(side, distance, np.inf).min(axis=1)
            result[name][block] = np.where(has_ball & np.isfinite(closest), closest, np.nan)
        result["nearest_dist"][block] = np.where(found, nearest, np.nan)
        result["nearest_team"][block] = np.where(found, teams[block][rows, slot], np.nan)
        result["nearest_speed_file"][block] = np.where(
            found, tracking.players[start:stop][rows, slot, 2], np.nan)
        relative = ball_velocity[start:stop] - player_velocity[start:stop][rows, slot]
        result["ball_rel_speed"][block] = np.where(found, np.hypot(relative[:, 0], relative[:, 1]),
                                                   np.nan)
        for radius in config.NEAR_RADII:
            near = distance < radius
            result["n_home_{:g}m".format(radius)][block] = np.where(
                has_ball, (near & home).sum(axis=1), np.nan)
            result["n_away_{:g}m".format(radius)][block] = np.where(
                has_ball, (near & ~home).sum(axis=1), np.nan)

    out.update(result)
    out["dist_diff"] = out["dist_away"] - out["dist_home"]  # > 0: home is closer

    # Control: a team has a player within CONTROL_RADIUS and nobody is closer. control_run is how
    # long that team's spell on the ball lasts in total (non-causal: the whole run, both sides).
    controlling = np.where(out["nearest_dist"] < config.CONTROL_RADIUS, out["nearest_team"], -1.0)
    controlling = np.where(np.isnan(controlling), -1.0, controlling)
    out["control_team"] = np.where(controlling >= 0, np.where(controlling == HOME, 1.0, -1.0),
                                   np.nan)
    out["control_run"] = run_lengths(controlling, ignore=-1.0)


def _team_shape_features(tracking, players, out, runs, kickoffs):
    """Centroid, extent, speed and own-half share of each team; attacking direction per half."""
    length, _ = tracking.pitch
    speed = tracking.players[:, :, 2]
    for team, name in ((HOME, "home"), (AWAY, "away")):
        mask = tracking.team_of_slot == team
        x, y = players[:, mask, 0], players[:, mask, 1]
        out[name + "_count"] = (~np.isnan(x)).sum(axis=1).astype(np.float64)
        out[name + "_centroid_x"] = np.nanmean(x, axis=1)
        out[name + "_centroid_y"] = np.nanmean(y, axis=1)
        out[name + "_length"] = np.nanmax(x, axis=1) - np.nanmin(x, axis=1)
        out[name + "_width"] = np.nanmax(y, axis=1) - np.nanmin(y, axis=1)
        out[name + "_speed_file"] = np.nanmean(speed[:, mask], axis=1)
    out["all_speed_file"] = np.nanmean(speed, axis=1)

    sign = attack_signs(out["home_centroid_x"], length, runs, kickoffs)
    out["home_attack_sign"] = sign
    # Ball x measured along home's attacking direction: 0 = home goal line, length = away's.
    out["ball_x_home_attack"] = np.where(sign > 0, out["ball_x"],
                                         np.where(sign < 0, length - out["ball_x"], np.nan))
    for team, name, own_sign in ((HOME, "home", 1.0), (AWAY, "away", -1.0)):
        x = players[:, tracking.team_of_slot == team, 0]
        defends_low_x = (sign * own_sign > 0)[:, None]
        own_half = np.where(defends_low_x, x < length / 2.0, x > length / 2.0) & ~np.isnan(x)
        count = out[name + "_count"]
        out[name + "_own_half_frac"] = np.where((count > 0) & ~np.isnan(sign),
                                                own_half.sum(axis=1) / np.maximum(count, 1), np.nan)


def tag_features(events, n_frames):
    """Smart Tagging as per-frame arrays.

    setpiece_window: 1 inside a SET PIECES window, 0 outside.
    setpiece_type: config.SET_PIECE_TYPES code inside a SET PIECES window, NaN outside.
    setpiece_team: +1 home, -1 away, 0 N/A inside a window, NaN outside.
    tag_<code>:    sum of team signs of the phase tags covering the frame, 0 where none.
    Phase tags keep their own feature each because their team means different things: on an
    attacking phase it is probably the team with the ball, on a DEFENDING one probably not. The
    feature report shows which way each one points; nothing here assumes it.
    """
    out = {"setpiece_window": np.zeros(n_frames), "setpiece_type": np.full(n_frames, np.nan),
           "setpiece_team": np.full(n_frames, np.nan)}
    for code in config.TAG_PHASE_CODES:
        out["tag_" + _slug(code)] = np.zeros(n_frames)
    out["tag_other"] = np.zeros(n_frames)

    for event in events:
        start, stop = max(0, event["start_frame"]), min(n_frames, event["end_frame"])
        if start >= stop:
            continue
        code = (event["code"] or "").strip().lower()
        tags = {key.strip().lower(): (value or "").strip().lower()
                for key, value in event["tags"].items() if key}
        team = config.TAG_TEAMS.get(tags.get("team", "n/a"), 0)
        if code == "set pieces":
            out["setpiece_window"][start:stop] = 1.0
            out["setpiece_type"][start:stop] = config.SET_PIECE_TYPES.get(
                tags.get("type", ""), config.SET_PIECE_OTHER)
            out["setpiece_team"][start:stop] = team
        else:
            key = "tag_" + _slug(code)
            out[key if key in out else "tag_other"][start:stop] += team
    return out


def drop_ghosts(players, min_frames):
    """Copy of players (n, slots, 2) with "ghost" stretches set to NaN.

    A real player never moves in a perfectly straight line at constant speed: tracking noise alone
    makes his second difference non-zero. A slot whose second difference is zero (< 1e-5 m/frame^2)
    for at least `min_frames` frames is a track the tracker is extrapolating or holding, not a
    player it sees: mostly a keeper who was lost, 89% of them within 20 m of a goal line.
    """
    out = players.copy()
    acc = np.hypot(np.diff(players[:, :, 0], 2, axis=0), np.diff(players[:, :, 1], 2, axis=0))
    flat = np.zeros(players.shape[:2], dtype=bool)
    flat[1:-1] = acc < 1e-5
    for slot in range(players.shape[1]):
        ghost = flat[:, slot] & (run_lengths(flat[:, slot], ignore=False) >= min_frames)
        out[ghost, slot] = np.nan
    return out


def _keeper_features(tracking, players, ball, out, teams):
    """home_keeper_dist / away_keeper_dist: metres from the ball to that team's keeper.

    The keepers are the two established keeper slots (teamfix.find_keeper_slots); each counts for
    the team it has in that frame, so a swapped slot counts for the right team. NaN without
    a ball or with that keeper untracked.
    """
    filled, _ = ball
    length = tracking.pitch[0]
    sign = out["home_attack_sign"]
    home_goal = np.where(np.isnan(sign), np.nan, np.where(sign > 0, 0.0, length))
    keepers = [k for k in teamfix.find_keeper_slots(players[:, :, 0], home_goal, length - home_goal)
               if k is not None]
    for team, name in ((HOME, "home"), (AWAY, "away")):
        best = np.full(tracking.n_frames, np.nan)
        for slot in keepers:
            d = np.hypot(players[:, slot, 0] - filled[:, 0], players[:, slot, 1] - filled[:, 1])
            d = np.where(teams[:, slot] == team, d, np.nan)
            best = np.fmin(best, d)
        out[name + "_keeper_dist"] = best


def _teams_per_frame(tracking, players, out):
    """Team of each slot per frame: the slot's own team, with keepers repaired (teamfix)."""
    teams = _repaired_teams(tracking, players, out)
    if config.TEAM_SOURCE == "colour":
        teams = _colour_teams(tracking, teams)
    return teams


def _colour_teams(tracking, base):
    """Outfield slots take the team of their shirt colour (mtah.shirt_team); needs the V1 cache."""
    from games import video_path
    from mtah import cv_stages, shirt_team, video
    path = video.cache_path(video_path(), "shirt", cv_stages.VERSIONS["shirt"])
    cached = video.load_cache(path)
    if cached is None:
        raise FileNotFoundError("TEAM_SOURCE = 'colour' needs {}: run scripts/cv_run.py --stage "
                                "shirt".format(os.path.relpath(path, config.ROOT)))
    return shirt_team.frame_teams(cached["lab"], tracking.team_of_slot, base)


def _repaired_teams(tracking, players, out):
    if config.KEEPER_MAX_DIST is None and not config.KEEPER_SWAP:
        return np.tile(tracking.team_of_slot, (tracking.n_frames, 1))
    length = tracking.pitch[0]
    home_goal = np.where(out["home_attack_sign"] > 0, 0.0, length)  # home defends x = 0 if +x
    home_goal = np.where(np.isnan(out["home_attack_sign"]), np.nan, home_goal)
    x = players[:, :, 0]
    keepers = teamfix.find_keeper_slots(x, home_goal, length - home_goal)
    if config.KEEPER_MAX_DIST is None:
        teams = np.tile(np.asarray(tracking.team_of_slot, dtype=np.int8), (tracking.n_frames, 1))
    else:
        teams = teamfix.keeper_teams(x, tracking.team_of_slot, home_goal, length - home_goal,
                                     config.KEEPER_MAX_DIST, keepers)
    if config.KEEPER_SWAP:
        teams = teamfix.swap_keeper_slots(x, teams, keepers, home_goal, length - home_goal,
                                          length / 2.0)
    return teams


# -- entry points ----------------------------------------------------------------------------------

def build(tracking, events=None, reprojection_error=None, runs=None, kickoffs=config.KICKOFFS,
          owner_ball=None):
    """Every feature for every frame. events and reprojection_error are optional extras.

    owner_ball: optional (n, 3) ball array like Tracking.ball (ATD plus our detector, V2) used only
    for the features the owner rule reads (OWNER_BALL_KEYS). Everything else, the dead-ball
    indicators above all, keeps the ATD ball: those rules read the ATD's gaps as evidence.
    """
    runs = postprocess.graded_runs() if runs is None else runs
    out = {}
    players = tracking.players_m()
    if config.GHOST_MIN_FRAMES is not None:
        players = drop_ghosts(players, config.GHOST_MIN_FRAMES)
    with warnings.catch_warnings():
        # All-NaN rows (no ball, nobody tracked at halftime) are expected and come out NaN.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ball = _ball_features(tracking, out)
        _team_shape_features(tracking, players, out, runs, kickoffs)
        teams = _teams_per_frame(tracking, players, out)
        _ball_player_features(tracking, players, ball, out, teams)
        _keeper_features(tracking, players, ball, out, teams)
        if owner_ball is not None:
            import dataclasses
            fused = dataclasses.replace(tracking, ball=owner_ball)
            scratch = {}
            _ball_player_features(fused, players, _ball_features(fused, {}), scratch, teams)
            for key in OWNER_BALL_KEYS:
                out[key] = scratch[key]
            out["owner_ball_visible"] = (~np.isnan(owner_ball[:, 0])).astype(np.float64)
    if events is not None:
        out.update(tag_features(events, tracking.n_frames))
    if reprojection_error is not None:
        out["homography_error"] = np.asarray(reprojection_error, dtype=np.float64)
    return out


def frame_teams(tracking, runs=None, kickoffs=config.KICKOFFS):
    """(n_frames, n_slots) team of every slot per frame as build() sees it under config.TEAM_SOURCE."""
    runs = postprocess.graded_runs() if runs is None else runs
    out = {}
    players = tracking.players_m()
    if config.GHOST_MIN_FRAMES is not None:
        players = drop_ghosts(players, config.GHOST_MIN_FRAMES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        _ball_features(tracking, out)
        _team_shape_features(tracking, players, out, runs, kickoffs)
        return _teams_per_frame(tracking, players, out)


def load(game=GAME, use_cache=True):
    """build() on the match data, from cache/ when the inputs have not changed."""
    raw, tagging, homography = require(["atd_raw", "smart_tagging", "homography"], game)
    cache_path = _cache_path(raw, tagging, homography) if use_cache else None
    if cache_path and os.path.isfile(cache_path):
        with np.load(cache_path) as cached:
            return {name: cached[name] for name in cached.files}

    tracking = io_atd.load(game, use_cache=use_cache)
    owner_ball = _fused_ball(tracking) if config.BALL_SOURCE == "fused" else None
    events = load_smart_tagging.load(game)
    _, reprojection_error, _ = load_homography.load(game)
    features = build(tracking, events, reprojection_error, owner_ball=owner_ball)

    if cache_path:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        partial = cache_path + ".partial"
        with open(partial, "wb") as handle:
            np.savez(handle, **features)
        os.replace(partial, cache_path)
    return features


# The features the owner rule reads; the only ones the fused ball (V2) replaces.
OWNER_BALL_KEYS = ("nearest_dist", "nearest_team", "nearest_speed_file", "ball_rel_speed")


def _fused_ball(tracking):
    """The ball from ATD + detector (mtah.ball_fusion), shaped like Tracking.ball; needs the V2 cache."""
    from games import video_path
    from mtah import ball_fusion, cv_stages, video
    path = _ball_det_path()
    cached = video.load_cache(path)
    if cached is None:
        raise FileNotFoundError("BALL_SOURCE = 'fused' needs {}: run scripts/cv_run.py --stage "
                                "ball_det".format(os.path.relpath(path, config.ROOT)))
    matrices, _, valid = load_homography.load()
    ball, _ = ball_fusion.fuse(cached["det"], tracking.ball, matrices, valid, tracking.pitch,
                               players_m=tracking.players_m()[..., :2])
    return ball


def _ball_det_path():
    from games import video_path
    from mtah import cv_stages, video
    return video.cache_path(video_path(), "ball_det", cv_stages.VERSIONS["ball_det"],
                            cv_stages.cache_params("ball_det"))


def _cache_path(*sources):
    """Cache file keyed on FEATURE_VERSION, the config and each source file's size and mtime."""
    key = hashlib.sha1()
    key.update(str(FEATURE_VERSION).encode())
    key.update(str((config.TEAM_SOURCE, config.BALL_SOURCE)).encode())   # may be set at run time, not only in the file
    if config.TEAM_SOURCE == "colour":
        from mtah import shirt_team
        key.update(repr((shirt_team.SWITCH_COST, shirt_team.ATD_PULL)).encode())
    if config.BALL_SOURCE == "fused":
        from mtah import ball_fusion
        key.update(os.path.basename(_ball_det_path()).encode())   # keyed on the net's weights
        with open(ball_fusion.__file__, "rb") as handle:
            key.update(handle.read())
    with open(config.__file__, "rb") as handle:
        key.update(handle.read())  # a changed parameter must not reuse old features
    for path in sources:
        stat = os.stat(path)
        key.update("{}:{}:{}".format(os.path.basename(path), stat.st_size,
                                     stat.st_mtime_ns).encode())
    return os.path.join(config.CACHE_DIR, "features_v{}_{}.npz".format(
        FEATURE_VERSION, key.hexdigest()[:12]))
