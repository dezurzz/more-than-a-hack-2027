"""One SkillCorner match (10 fps broadcast tracking) -> the same Match container as IDSSE, at 25 fps.

Reuses everything downstream (convert, damage-free tracking, features, scoring, models) unchanged, so a
SkillCorner match is one more "external match" whose damage is real.

Rows. For each half (`match_periods`: start_frame .. end_frame, 10 fps frames, inclusive) one row per
25 fps frame: row k sits at the 10 fps position u = start + 0.4 k (kept as the integers lo = start +
(2k) // 5 and rem = (2k) % 5, so u = lo + rem / 5). Positions are linear between the 10 fps frames lo
and lo + 1 when the object has a value at both (exact when rem = 0), else NaN. `half` from the period.

Variants. `sc`: players as listed (the off-camera ones are SkillCorner's extrapolation, the closest
thing to the ATD's tracker), the ball only where its `is_detected` is true. `sc_detected`: players only
where their own `is_detected` is true, ball as in `sc`.

Labels. Ball status: alive (1) where the row's 10 fps frame lo lies in some phase of play
(`phases_of_play.csv`, "defined only when the ball is in play"), else dead (0). Possession, two sources
(`source`): "phase" = the team in possession of the phase; "group" = `possession.group` of the nearest
10 fps frame, a null inside a phase filled with the last non-null of the same phase, else the next, else
the phase's team. 1 home, 2 away, like IDSSE. `Match.labels(mapping)` then works unchanged (mapping None =
as delivered).

Players' speed (the model's input) comes from their own positions, stored times 3.6 because
convert.normalise takes IDSSE's km/h.
"""

import csv
import json
import os

import numpy as np

from mtah import config
from mtah.external import convert, idsse, skillcorner_files as files

FORMAT_VERSION = 1
SOURCE_FPS = 10
VARIANTS = ("sc", "sc_detected")
SOURCES = ("phase", "group")
HOME, AWAY = idsse.HOME, idsse.AWAY


def read_match_json(match_id, data_dir=None):
    with open(files.verify(match_id, "match", data_dir), encoding="utf-8") as handle:
        return json.load(handle)


def read_phases(match_id, home_id, data_dir=None):
    """[(frame_start, frame_end, team)] team 1 home / 2 away, sorted."""
    out = []
    with open(files.verify(match_id, "phases", data_dir), encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            team = 1 if int(row["team_in_possession_id"]) == home_id else 2
            out.append((int(row["frame_start"]), int(row["frame_end"]), team))
    return sorted(out)


def parse_tracking(path, player_ids):
    """10 fps arrays indexed by frame number: ball (F, 3) x y z with NaN where the ball is not detected,
    ball_present (F,), players (F, P, 2) with NaN where absent, seen (F, P) the player's own is_detected,
    group (F,) 0 none / 1 home / 2 away, period (F,) 0 outside a half."""
    column = {pid: i for i, pid in enumerate(player_ids)}
    frames = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                frames.append(json.loads(line))
    n = max(r["frame"] for r in frames) + 1
    ball = np.full((n, 3), np.nan)
    players = np.full((n, len(player_ids), 2), np.nan)
    seen = np.zeros((n, len(player_ids)), dtype=bool)
    group = np.zeros(n, dtype=np.int8)
    period = np.zeros(n, dtype=np.int8)
    unknown = set()
    for r in frames:
        f = r["frame"]
        period[f] = r.get("period") or 0
        b = r.get("ball_data") or {}
        if b.get("is_detected") and b.get("x") is not None and b.get("y") is not None:
            ball[f] = (b["x"], b["y"], b["z"] if b.get("z") is not None else np.nan)
        g = (r.get("possession") or {}).get("group")
        group[f] = 1 if g == "home team" else (2 if g == "away team" else 0)
        for p in r.get("player_data") or []:
            i = column.get(p["player_id"])
            if i is None:
                unknown.add(p["player_id"])
                continue
            if p.get("x") is not None and p.get("y") is not None:
                players[f, i] = (p["x"], p["y"])
                seen[f, i] = bool(p.get("is_detected"))
    return ball, players, seen, group, period, unknown


def phase_arrays(phases, n, group):
    """alive (n,) bool, phase_team (n,) int8, group filled inside phases (n,) int8."""
    alive = np.zeros(n, dtype=bool)
    team = np.zeros(n, dtype=np.int8)
    filled = group.copy()
    for start, stop, t in phases:
        stop = min(stop, n - 1)
        if start > stop:
            continue
        alive[start:stop + 1] = True
        team[start:stop + 1] = t
        span = group[start:stop + 1]
        have = np.flatnonzero(span > 0)
        if not len(have):
            filled[start:stop + 1] = t
            continue
        idx = np.arange(len(span))
        last = np.maximum.accumulate(np.where(span > 0, idx, -1))
        nxt = np.minimum.accumulate(np.where(span > 0, idx, len(span))[::-1])[::-1]
        pick = np.where(last >= 0, last, nxt)
        filled[start:stop + 1] = span[pick]
    return alive, team, filled


def half_rows(start, end):
    """(lo, rem) of the 25 fps rows of one half: lo = 10 fps frame at or before the row, rem in 0..4 fifths."""
    k = np.arange(int((end - start) * 5 // 2) + 1)
    return start + (2 * k) // 5, (2 * k) % 5


def resample(values, lo, rem):
    """Linear in time between 10 fps frames lo and lo + 1 (values (F, ...)); NaN unless both ends exist."""
    top = np.minimum(lo + 1, len(values) - 1)
    w = (rem / 5.0).reshape((-1,) + (1,) * (values.ndim - 1))
    out = (1 - w) * values[lo] + w * values[top]
    exact = (rem == 0).reshape(w.shape)
    return np.where(exact, values[lo], out)


def assemble(meta, ball10, players10, seen10, group10, phases, variant):
    home_id = meta["home_team"]["id"]
    players_meta = {p["id"]: p for p in meta["players"]}
    ids = sorted(players_meta)
    n10 = len(ball10)
    alive10, team10, group_filled = phase_arrays(phases, n10, group10)
    if variant == "sc_detected":
        players10 = np.where(seen10[..., None], players10, np.nan)
    los, rems, halves = [], [], []
    for p in sorted(meta["match_periods"], key=lambda p: p["period"]):
        lo, rem = half_rows(p["start_frame"], p["end_frame"])
        los.append(lo)
        rems.append(rem)
        halves.append(np.full(len(lo), p["period"], dtype=np.int8))
    lo, rem, half = np.concatenate(los), np.concatenate(rems), np.concatenate(halves)
    lo = np.minimum(lo, n10 - 1)
    ball = resample(ball10, lo, rem).astype(np.float32)
    players_xy = resample(players10, lo, rem)                          # (n, P, 2)
    nearest = np.minimum(lo + (rem >= 3), n10 - 1)
    status = alive10[lo].astype(np.int8)
    poss_phase = np.where(alive10[lo], team10[lo], 0).astype(np.int8)
    poss_group = np.where(alive10[nearest], group_filled[nearest], 0).astype(np.int8)
    players = np.full(players_xy.shape[:2] + (3,), np.nan, dtype=np.float32)
    players[..., :2] = players_xy
    for a, b in idsse_runs(half):
        for i in range(players.shape[1]):
            speed = convert.ball_speed_ms(players_xy[a:b, i, 0].astype(np.float64),
                                          players_xy[a:b, i, 1].astype(np.float64))
            players[a:b, i, 2] = speed * 3.6
    person_team = np.array([HOME if players_meta[i]["team_id"] == home_id else AWAY for i in ids], dtype=np.int8)
    frame_number = np.rint((lo + rem / 5.0) * 2.5).astype(np.int32)
    return dict(frame_number=frame_number, half=half, ball=ball,
                ball_speed=np.full(len(lo), np.nan, dtype=np.float32), status=status,
                possession_phase=poss_phase, possession_group=poss_group,
                person_ids=np.array([str(i) for i in ids]), person_team=person_team, players=players)


def idsse_runs(half):
    out = []
    for h in (1, 2):
        rows = np.flatnonzero(half == h)
        if len(rows):
            out.append((int(rows[0]), int(rows[-1]) + 1))
    return out


def to_match(meta, arrays, source):
    return idsse.Match(
        match_id=str(meta["id"]), competition="A-League 2024/25",
        home_name=meta["home_team"]["short_name"], away_name=meta["away_team"]["short_name"],
        pitch=(float(meta["pitch_length"]), float(meta["pitch_width"])),
        frame_number=arrays["frame_number"], half=arrays["half"], ball=arrays["ball"],
        ball_speed=arrays["ball_speed"], status=arrays["status"],
        possession=arrays["possession_" + source], person_ids=tuple(str(p) for p in arrays["person_ids"]),
        person_team=arrays["person_team"], players=arrays["players"])


def load_files(match_json, phases, tracking_path, variant="sc", source="phase"):
    with open(match_json, encoding="utf-8") as handle:
        meta = json.load(handle)
    ids = sorted(p["id"] for p in meta["players"])
    ball, players, seen, group, _, _ = parse_tracking(tracking_path, ids)
    return to_match(meta, assemble(meta, ball, players, seen, group, phases, variant), source)


def load(match_id, variant="sc", source="phase", data_dir=None, use_cache=True):
    """One match from cache/external/ when possible; files checked against the manifest."""
    if variant not in VARIANTS or source not in SOURCES:
        raise ValueError("variant {} source {}".format(variant, source))
    tracking_path = files.verify(match_id, "tracking", data_dir)
    meta = read_match_json(match_id, data_dir)
    cache_path = os.path.join(config.CACHE_DIR, "external", "sc_v{}_{}_{}_{}.npz".format(
        FORMAT_VERSION, match_id, variant, os.path.getsize(tracking_path)))
    keys = ("frame_number", "half", "ball", "ball_speed", "status", "possession_phase", "possession_group",
            "person_ids", "person_team", "players")
    if use_cache and os.path.isfile(cache_path):
        with np.load(cache_path, allow_pickle=False) as cached:
            return to_match(meta, {k: cached[k] for k in keys}, source)
    phases = read_phases(match_id, meta["home_team"]["id"], data_dir)
    ids = sorted(p["id"] for p in meta["players"])
    ball, players, seen, group, _, _ = parse_tracking(tracking_path, ids)      # one parse, both variants
    wanted = None
    for name in VARIANTS:
        arrays = assemble(meta, ball, players, seen, group, phases, name)
        if name == variant:
            wanted = arrays
        if use_cache:
            path = os.path.join(config.CACHE_DIR, "external", "sc_v{}_{}_{}_{}.npz".format(
                FORMAT_VERSION, match_id, name, os.path.getsize(tracking_path)))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            partial = path + ".partial"
            with open(partial, "wb") as handle:
                np.savez(handle, **arrays)
            os.replace(partial, path)
    return to_match(meta, wanted, source)
