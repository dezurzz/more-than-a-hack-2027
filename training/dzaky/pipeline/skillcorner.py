"""SkillCorner Open Data (A-League 2024/25, MIT) converted to the same arrays as the ATD, plus H/A/D.

Broadcast tracking: built from the TV picture, so players off screen and the ball are often
missing, much like the ATD and unlike DFL's stadium cameras. Download a match's four files from

    https://github.com/SkillCorner/opendata  (data/matches/<id>/, tracking via Git LFS)

into external/skillcorner/<id>/. Unlike the competition match this data may be read freely.

The ball is NaN on frames where SkillCorner marks it not detected, which leaves it missing about
as often as on the ATD (46% against 41%). Players keep SkillCorner's extrapolated positions off
screen: detected-only would leave a median of 9 of 22 on the pitch, while the ATD loses players
far less often (PLAYERS_DETECTED_ONLY switches that). 10 Hz is resampled to 25 Hz by linear interpolation between neighbouring frames;
a gap on either side stays a gap. Speeds are differences of the 25 Hz positions, in m/s.

Labels come from SkillCorner's phases of play, which cover the time a team has the ball under
control -- the challenge's notion of possession -- and nothing else. Fixed before any scoring:

    inside a phase                                   the team in possession, H or A
    gap >= RESTART_GAP_S before a restart phase      D  (throw-in, corner, free kick, goal kick,
                                                         or a phase opening without a possession)
    gap >= DEAD_GAP_S                                D  (whatever comes next)
    any shorter gap                                  the previous team: a loose ball stays with
                                                     the last team, as in the challenge
    before the first and after the last phase        D

Layout matches pipeline/dfl.py: halves back to back with HALFTIME_GAP frames of X between them,
origin at a pitch corner, one slot per player. Each match is cached under external/cache/.
"""

import csv
import json
import os

import numpy as np

from pipeline.config import ROOT, A, H, Config
from pipeline.data import Tracking
from pipeline.dfl import CACHE_DIR, FPS, HALFTIME_GAP

SKILLCORNER_DIR = os.path.join(ROOT, "external", "skillcorner")
SOURCE_FPS = 10
RESTART_GAP_S = 1.0
DEAD_GAP_S = 5.0
RESTARTS = ("throw_in", "free_kick", "corner", "goal_kick")
PLAYERS_DETECTED_ONLY = False


def match_dir(match, root=SKILLCORNER_DIR):
    return os.path.join(root, str(match))


def read_frames(path):
    """Tracking rows, skipping a truncated last line (an interrupted download)."""
    rows = []
    with open(path) as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except ValueError:
                break
    return rows


def phase_labels(phases, events, start, stop):
    """One of H/A/D per 10 Hz frame in [start, stop] of a period, from its phases of play."""
    labels = np.full(stop - start + 1, "D", dtype="<U1")
    opening = {}
    for event in events:
        if event["event_type"] != "player_possession" or not event["phase_index"]:
            continue
        phase, frame = event["phase_index"], int(event["frame_start"])
        if phase not in opening or frame < opening[phase][0]:
            opening[phase] = (frame, event["start_type"] or "")
    previous = None
    for phase in phases:
        a, b = int(phase["frame_start"]) - start, int(phase["frame_end"]) - start
        letter = phase["team"]
        if previous is not None:
            gap_s = (a - previous[1]) / SOURCE_FPS
            kind = opening.get(phase["index"], (0, ""))[1]
            restart = kind.startswith(RESTARTS) or phase["index"] not in opening
            dead = gap_s >= DEAD_GAP_S or (restart and gap_s >= RESTART_GAP_S)
            if not dead:
                labels[max(previous[1], 0):max(a, 0)] = previous[2]
        labels[max(a, 0):b + 1] = letter
        previous = (a, b + 1, letter)
    return labels


def resample(values, n_out):
    """Linear 10 -> 25 Hz along axis 0; NaN wherever either neighbour is NaN."""
    position = np.arange(n_out) * SOURCE_FPS / FPS
    low = np.minimum(position.astype(int), len(values) - 1)
    high = np.minimum(low + 1, len(values) - 1)
    weight = (position - low).reshape((-1,) + (1,) * (values.ndim - 1))
    return (1 - weight) * values[low] + weight * values[high]


def build_match(match, root=SKILLCORNER_DIR):
    """Tracking, labels and halves for one match, read from its downloaded files."""
    folder = match_dir(match, root)
    info = json.load(open(os.path.join(folder, "{}_match.json".format(match))))
    home = info["home_team"]["id"]
    pitch = (float(info["pitch_length"]), float(info["pitch_width"]))
    people = sorted(info["players"], key=lambda p: (p["team_id"] != home, p["id"]))
    slot = {p["id"]: i for i, p in enumerate(people)}
    team = np.array([H if p["team_id"] == home else A for p in people], dtype=np.int8)
    letter = {home: "H"}

    rows = read_frames(os.path.join(folder, "{}_tracking_extrapolated.jsonl".format(match)))
    by_frame = {row["frame"]: row for row in rows}
    with open(os.path.join(folder, "{}_phases_of_play.csv".format(match))) as handle:
        phases = list(csv.DictReader(handle))
    with open(os.path.join(folder, "{}_dynamic_events.csv".format(match))) as handle:
        events = list(csv.DictReader(handle))
    for phase in phases:
        phase["team"] = letter.get(int(phase["team_in_possession_id"]), "A")

    offset = np.array([pitch[0] / 2, pitch[1] / 2], dtype=np.float32)
    blocks, halves, cursor = [], [], 0
    for period in info["match_periods"]:
        start, stop = int(period["start_frame"]), int(period["end_frame"])
        n_in = stop - start + 1
        players = np.full((n_in, len(people), 2), np.nan, dtype=np.float32)
        ball = np.full((n_in, 2), np.nan, dtype=np.float32)
        for i in range(n_in):
            row = by_frame.get(start + i)
            if row is None:
                continue
            seen = row["ball_data"]
            if seen and seen.get("is_detected") and seen.get("x") is not None:
                ball[i] = (seen["x"], seen["y"])
            for player in row["player_data"]:
                seen_player = player.get("is_detected") or not PLAYERS_DETECTED_ONLY
                if seen_player and player["player_id"] in slot and player.get("x") is not None:
                    players[i, slot[player["player_id"]]] = (player["x"], player["y"])
        source_labels = phase_labels(
            [p for p in phases if int(p["period"]) == period["period"]],
            [e for e in events if str(e["period"]) == str(period["period"])], start, stop)

        n = int(n_in * FPS / SOURCE_FPS)
        players, ball = resample(players, n) + offset, resample(ball, n) + offset
        speeds = np.linalg.norm(np.gradient(players, axis=0), axis=2) * FPS
        labels = source_labels[np.minimum(np.rint(np.arange(n) * SOURCE_FPS / FPS).astype(int),
                                          n_in - 1)]
        if blocks:
            gap = HALFTIME_GAP
            blocks.append((np.full((gap, len(people), 2), np.nan, dtype=np.float32),
                           np.full((gap, len(people)), np.nan, dtype=np.float32),
                           np.full((gap, 2), np.nan, dtype=np.float32),
                           np.full(gap, "X", dtype="<U1")))
            cursor += gap
        blocks.append((players.astype(np.float32), speeds.astype(np.float32),
                       ball.astype(np.float32), labels))
        halves.append((period["period"], cursor, cursor + n - 1))
        cursor += n

    players, speeds, ball, labels = (np.concatenate(parts) for parts in zip(*blocks))
    return Tracking(players, speeds, ball, team), labels, halves, pitch


def load_match(match, root=SKILLCORNER_DIR, cache=True):
    """(Tracking, labels, halves, cfg) for one SkillCorner match; cfg carries its pitch size."""
    cache_path = os.path.join(CACHE_DIR, "skillcorner_{}.npz".format(match))
    if cache and os.path.isfile(cache_path):
        stored = np.load(cache_path)
        tracking = Tracking(stored["players"], stored["speeds"], stored["ball"], stored["team"])
        halves = [tuple(int(v) for v in row) for row in stored["halves"]]
        pitch = tuple(float(v) for v in stored["pitch"])
        labels = stored["labels"]
    else:
        tracking, labels, halves, pitch = build_match(match, root)
        if cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.savez_compressed(cache_path, players=tracking.players, speeds=tracking.speeds,
                                ball=tracking.ball, team=tracking.team, labels=labels,
                                halves=np.array(halves), pitch=np.array(pitch))
    cfg = Config(FPS=FPS, LAPANGAN_P=pitch[0], LAPANGAN_L=pitch[1], PAKAI_SMART_TAGGING=False)
    return tracking, labels, halves, cfg


def complete(match, root=SKILLCORNER_DIR):
    """True when all four files of a match are present and the tracking file ends cleanly."""
    folder = match_dir(match, root)
    names = ["{}_{}".format(match, suffix) for suffix in
             ("match.json", "phases_of_play.csv", "dynamic_events.csv", "tracking_extrapolated.jsonl")]
    if not all(os.path.isfile(os.path.join(folder, name)) for name in names):
        return False
    with open(os.path.join(folder, names[-1]), "rb") as handle:
        handle.seek(max(0, os.path.getsize(handle.name) - 65536))
        last = handle.read().strip().rsplit(b"\n", 1)[-1]
    try:
        return json.loads(last)["frame"] is not None
    except (ValueError, KeyError, TypeError):
        return False


def available_matches(root=SKILLCORNER_DIR):
    """Match ids in matches.json whose files are fully downloaded."""
    listing = os.path.join(root, "matches.json")
    if not os.path.isfile(listing):
        return []
    return [m["id"] for m in json.load(open(listing)) if complete(m["id"], root)]
