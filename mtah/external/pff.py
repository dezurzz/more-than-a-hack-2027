"""One PFF FC World Cup 2022 match (broadcast tracking, 29.97 fps) -> the same Match container as IDSSE, at 25 fps.

PFF_PLAN.md fixes the rules; this module implements them. The data is obtained from PFF FC under its terms
and is not redistributed: the files stay outside the repository, and only aggregates are printed by anything
built on this module.

Fields. Only the RAW `homePlayers`, `awayPlayers` and `balls` are read; any `*Smoothed` field is never
touched (ballsSmoothed places the ball at the carrier, i.e. it is derived from the possession events).
A file whose first frame lacks a raw field is refused. Coordinates: metres, origin at the centre.

Ball. A ball sample with visibility ESTIMATED (or no ball in the list) is NaN. Players keep their
ESTIMATED samples. Players are identified across the match by (side, jersey number): one person index each.

Rows. Per period (1 and 2; extra time is dropped) one row per 25 Hz tick: row k sits at PFF frame
position u = f0 + k * fps / 25, f0 = the period's first tracking frame. Positions are linear between the
two PFF frames around u, NaN unless both exist (exact when u is a frame). The label of a row is the label
of the nearest PFF frame. Players' speed is the finite difference of their resampled positions
(convert.ball_speed_ms), stored times 3.6 because convert.normalise takes km/h (as skillcorner does).

Labels. From the game events (Event Data, one record per gameEventId; the start frame is the tracking's
own `game_event.start_frame` when the tracking carries the event, else the tracking frame at the event's
start time), per period, sorted by start frame, see label_events().
"""

import bz2
import glob
import json
import os

import numpy as np

from mtah.external import convert, idsse

FORMAT_VERSION = 1
_WORK = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "work_data")
PFF_DIR = os.environ.get("MTAH_PFF_DIR", os.path.join(_WORK, "pff_data", "FIFA World Cup 2022"))
CACHE_DIR = os.environ.get("MTAH_PFF_CACHE", os.path.join(_WORK, "overnight_cache", "pff"))
HOME, AWAY = idsse.HOME, idsse.AWAY
FPS = 25
RAW_FIELDS = ("homePlayers", "awayPlayers", "balls")
KICKOFFS = ("FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF")
DEAD_EVENTS = ("OUT", "SUB", "OFF", "ON", "VID")
PERIODS = (1, 2)
VIS_NONE, VIS_VISIBLE, VIS_ESTIMATED = 0, 1, 2
MAX_PERSONS = 48


# ---------------------------------------------------------------- paths
def _dir(*parts):
    return os.path.join(PFF_DIR, *parts)


def game_ids():
    """Game ids (strings, numeric order) that have a tracking file."""
    names = glob.glob(_dir("Tracking Data", "*.jsonl.bz2"))
    return [str(i) for i in sorted(int(os.path.basename(n).split(".")[0]) for n in names)]


def cache_path(game_id, kind="match", ext="npz"):
    return os.path.join(CACHE_DIR, "{}_v{}_{}.{}".format(kind, FORMAT_VERSION, game_id, ext))


def _atomic(path, writer, mode="wb"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    partial = path + ".partial"
    with open(partial, mode) as handle:
        writer(handle)
    os.replace(partial, path)


# ---------------------------------------------------------------- tracking
def ball_sample(record):
    """(x, y, z, code): the raw ball of one frame. code VIS_VISIBLE keeps its position; ESTIMATED or
    absent give NaN. Reads only the raw `balls` list."""
    balls = record["balls"] or []
    pick = None
    for b in balls:
        if b.get("visibility") == "VISIBLE":
            pick = b
            break
    if pick is None:
        if balls:
            return np.nan, np.nan, np.nan, VIS_ESTIMATED if balls[0].get("visibility") == "ESTIMATED" else VIS_NONE
        return np.nan, np.nan, np.nan, VIS_NONE
    x, y, z = pick.get("x"), pick.get("y"), pick.get("z")
    if x is None or y is None:
        return np.nan, np.nan, np.nan, VIS_NONE
    return float(x), float(y), (float(z) if z is not None else np.nan), VIS_VISIBLE


class _Grow:
    """Append-only arrays that double their capacity."""

    def __init__(self, shapes, dtypes, fills, capacity=1 << 16):
        self.shapes, self.dtypes, self.fills = shapes, dtypes, fills
        self.n = 0
        self.arrays = [np.full((capacity,) + s, f, dtype=d) for s, d, f in zip(shapes, dtypes, fills)]

    def next_row(self):
        if self.n == len(self.arrays[0]):
            self.arrays = [np.concatenate([a, np.full(a.shape, f, dtype=a.dtype)])
                           for a, f in zip(self.arrays, self.fills)]
        self.n += 1
        return self.n - 1

    def done(self):
        return [a[:self.n] for a in self.arrays]


def parse_records(records):
    """Stream of frame dicts -> dict of arrays (see below). Duplicated frameNums keep the first record.

    frame (F,) int64, period (F,) int8, time_ms (F,) float64, ball (F, 3) float32 NaN unless VISIBLE,
    ball_code (F,) int8, players (F, P, 2) float32, player_code (F, P) int8 (0 absent / 1 VISIBLE / 2 ESTIMATED),
    persons [(side, jersey)] per column, events {game_event_id: (start_frame, home_ball)},
    fields: sorted raw coordinate field names seen (key names only).
    """
    grow = _Grow([(), (), (), (3,), (), (MAX_PERSONS, 2), (MAX_PERSONS,)],
                 [np.int64, np.int8, np.float64, np.float32, np.int8, np.float32, np.int8],
                 [-1, 0, np.nan, np.nan, 0, np.nan, 0])
    columns, events, fields, seen = {}, {}, set(), set()
    for k, record in enumerate(records):
        if k == 0:
            missing = [f for f in RAW_FIELDS if f not in record]
            if missing:
                raise ValueError("raw tracking field(s) missing: {}".format(", ".join(missing)))
        fields.update(key for key in record.keys() if "Player" in key or key.startswith("ball"))
        frame = int(record["frameNum"])
        if frame in seen:
            continue
        seen.add(frame)
        i = grow.next_row()
        frame_a, period_a, time_a, ball_a, code_a, pos_a, pcode_a = grow.arrays
        frame_a[i] = frame
        period_a[i] = record["period"] or 0
        time_a[i] = record["videoTimeMs"] if record["videoTimeMs"] is not None else np.nan
        x, y, z, code = ball_sample(record)
        ball_a[i] = (x, y, z)
        code_a[i] = code
        for side, key in ((HOME, "homePlayers"), (AWAY, "awayPlayers")):
            here = set()
            for p in record[key] or []:
                tag = (side, str(p.get("jerseyNum")))
                if tag in here:
                    continue
                here.add(tag)
                c = columns.get(tag)
                if c is None:
                    if len(columns) >= MAX_PERSONS:
                        continue
                    c = columns[tag] = len(columns)
                if p.get("x") is not None and p.get("y") is not None:
                    pos_a[i, c] = (p["x"], p["y"])
                    pcode_a[i, c] = VIS_ESTIMATED if p.get("visibility") == "ESTIMATED" else VIS_VISIBLE
        g = record.get("game_event")
        if g and record.get("game_event_id") is not None:
            eid = int(record["game_event_id"])
            if eid not in events:
                events[eid] = (int(g["start_frame"]), g.get("home_ball"))
    frame, period, time_ms, ball, ball_code, pos, pcode = grow.done()
    order = sorted(columns, key=lambda t: (t[0], int(t[1]) if t[1].isdigit() else 10 ** 6, t[1]))
    take = [columns[t] for t in order]
    return dict(frame=frame, period=period, time_ms=time_ms, ball=ball, ball_code=ball_code,
                players=pos[:, take], player_code=pcode[:, take], persons=order, events=events,
                fields=sorted(fields))


def parse_tracking(path):
    def lines():
        with bz2.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    return parse_records(lines())


# ---------------------------------------------------------------- events and labels
def read_events(game_id):
    """One dict per gameEventId from the Event Data file: id, type, period, time_ms, home (team of the player)."""
    with open(_dir("Event Data", "{}.json".format(game_id)), encoding="utf-8") as handle:
        rows = json.load(handle)
    out, seen = [], set()
    for row in rows:
        eid = int(row["gameEventId"])
        if eid in seen:
            continue
        seen.add(eid)
        g = row["gameEvents"]
        out.append(dict(id=eid, type=g["gameEventType"], period=int(g["period"]),
                        time_ms=float(row["startTime"]) * 1000.0, home=g.get("homeTeam")))
    return out


def label_events(events, f0, n):
    """(status (n,) int8, possession (n,) int8, first, end) of one period's n frames from frame f0.

    events: dicts with type, frame (absolute tracking frame) and team (True home / False away / None),
    processed in start-frame order (stable). PFF_PLAN section 2:
      kickoff, OTB       alive, team (OTB only after a kickoff)
      OUT, SUB, OFF, ON, VID   dead until the next OTB or kickoff
      G                  woodwork / corner flag, ball stays in play: no effect (PFF_PLAN amendment A1)
      END                the period's labels stop (X after it)
    Before the kickoff and after END: status -1 (X). first = frame index of the first kickoff, end = the END's
    index (n when there is none).
    """
    status = np.full(n, -1, dtype=np.int8)
    poss = np.zeros(n, dtype=np.int8)
    state = (-1, 0)
    started = False
    first, end, last = None, n, 0
    for e in sorted(events, key=lambda e: e["frame"]):
        at = int(min(max(e["frame"] - f0, 0), n))
        status[last:at], poss[last:at] = state
        last = at
        kind, team = e["type"], e.get("team")
        side = 1 if team else 2
        if kind in KICKOFFS:
            state, started = (1, side if team is not None else 0), True
            first = at if first is None else first
        elif kind == "OTB" and started and team is not None:
            state = (1, side)
        elif kind in DEAD_EVENTS and started:
            state = (0, 0)
        elif kind == "END" and started:
            end = at
            state = (-1, 0)
            break
    status[last:], poss[last:] = state
    return status, poss, first, end


def event_frames(events, parsed, period, f0, f1):
    """Events of one period with their tracking frame: the tracking's own start_frame when it carries the
    event, else the frame at the event's time. Returns (list for label_events, n_events, n_embedded, n_inside)."""
    sel = parsed["period"] == period
    frames, times = parsed["frame"][sel], parsed["time_ms"][sel]
    order = np.argsort(times)
    out, embedded, inside = [], 0, 0
    mine = [e for e in events if e["period"] == period]
    for e in mine:
        team = None
        if e["id"] in parsed["events"]:
            frame, home_ball = parsed["events"][e["id"]]
            team = home_ball if home_ball is not None else e["home"]
            embedded += 1
            inside += f0 <= frame <= f1
        else:
            j = int(np.searchsorted(times[order], e["time_ms"]))
            j = min(max(j, 0), len(order) - 1)
            frame = int(frames[order][j])
            team = e["home"]
            inside += bool(times.min() - 40 <= e["time_ms"] <= times.max() + 40)
        out.append(dict(type=e["type"], frame=frame, team=team))
    return out, len(mine), embedded, inside


# ---------------------------------------------------------------- resampling
def resample(values, u):
    """values (n, ...) float with NaN at positions 0..n-1; u (K,) float positions. Linear between the two
    frames around u, NaN unless both exist (exact when u is a frame)."""
    n = len(values)
    lo = np.floor(u + 1e-9).astype(np.int64)
    w = np.clip(u - lo, 0.0, 1.0)
    lo = np.clip(lo, 0, n - 1)
    top = np.minimum(lo + 1, n - 1)
    shape = (-1,) + (1,) * (values.ndim - 1)
    wb = w.reshape(shape)
    v0, v1 = values[lo].astype(np.float64), values[top].astype(np.float64)
    out = np.where(wb < 1e-6, v0, (1 - wb) * v0 + wb * v1)
    beyond = (np.floor(u + 1e-9) + (w >= 1e-6)) > n - 1
    out[beyond] = np.nan
    return out


def period_grid(n_frames, fps):
    """u (K,) PFF-frame offsets of the 25 Hz rows of a period of n_frames dense frames."""
    k = int(np.floor((n_frames - 1) * FPS / fps + 1e-9)) + 1
    return np.arange(k) * (fps / FPS)


# ---------------------------------------------------------------- Match
def read_meta(game_id):
    with open(_dir("Metadata", "{}.json".format(game_id)), encoding="utf-8") as handle:
        meta = json.load(handle)
    return meta[0] if isinstance(meta, list) else meta


def pitch_of(meta):
    pitches = meta["stadium"]["pitches"]
    current = [p for p in pitches if not p.get("endDate")] or pitches
    return float(current[0]["length"]), float(current[0]["width"])


def build(meta, parsed, events):
    """(Match, stats dict of plain numbers) from the metadata, the parsed tracking and the Event Data events."""
    fps = float(meta["fps"])
    persons = parsed["persons"]
    P = len(persons)
    balls, plays, labels_s, labels_p, halves, numbers = [], [], [], [], [], []
    stats = dict(period_frames=0, ball_visible=0, ball_estimated=0, ball_none=0, player_samples=0,
                 player_estimated=0, events=0, events_embedded=0, events_inside=0, kickoffs=0,
                 kickoffs_with_ball=0, kickoffs_ok=0, spans=[], fields=parsed["fields"], types={})
    offset = 0
    for period in PERIODS:
        sel = np.flatnonzero(parsed["period"] == period)
        if not len(sel):
            continue
        order = sel[np.argsort(parsed["frame"][sel])]
        frames = parsed["frame"][order]
        f0, f1 = int(frames[0]), int(frames[-1])
        n = f1 - f0 + 1
        at = frames - f0
        dense_ball = np.full((n, 3), np.nan)
        dense_ball[at] = parsed["ball"][order]
        dense_pos = np.full((n, P, 2), np.nan)
        dense_pos[at] = parsed["players"][order]
        code = np.zeros(n, dtype=np.int8)
        code[at] = parsed["ball_code"][order]
        stats["period_frames"] += len(order)
        stats["ball_visible"] += int((code == VIS_VISIBLE).sum())
        stats["ball_estimated"] += int((code == VIS_ESTIMATED).sum())
        stats["ball_none"] += int((code[at] == VIS_NONE).sum())
        pc = parsed["player_code"][order]
        stats["player_samples"] += int((pc > 0).sum())
        stats["player_estimated"] += int((pc == VIS_ESTIMATED).sum())

        ev, n_ev, n_emb, n_in = event_frames(events, parsed, period, f0, f1)
        stats["events"] += n_ev
        stats["events_embedded"] += n_emb
        stats["events_inside"] += n_in
        for e in ev:
            stats["types"][e["type"]] = stats["types"].get(e["type"], 0) + 1
            if e["type"] in KICKOFFS:
                stats["kickoffs"] += 1
                lo, hi = max(e["frame"] - f0 - 5, 0), min(e["frame"] - f0 + 6, n)
                seen = dense_ball[lo:hi]
                seen = seen[~np.isnan(seen[:, 0])]
                if len(seen):
                    stats["kickoffs_with_ball"] += 1
                    stats["kickoffs_ok"] += int(np.median(np.hypot(seen[:, 0], seen[:, 1])) <= 9.15)
        status_d, poss_d, first, end = label_events(ev, f0, n)

        u = period_grid(n, fps)
        k = len(u)
        nearest = np.clip(np.rint(u).astype(np.int64), 0, n - 1)
        pos25 = resample(dense_pos, u)
        players = np.full((k, P, 3), np.nan, dtype=np.float32)
        players[..., :2] = pos25
        for i in range(P):
            players[:, i, 2] = convert.ball_speed_ms(pos25[:, i, 0], pos25[:, i, 1]) * 3.6
        balls.append(resample(dense_ball, u).astype(np.float32))
        plays.append(players)
        labels_s.append(status_d[nearest])
        labels_p.append(poss_d[nearest])
        halves.append(np.full(k, period, dtype=np.int8))
        numbers.append(frames[0] + np.rint(u).astype(np.int64))
        a = int(np.searchsorted(u, first)) if first is not None else k
        b = int(np.searchsorted(u, end)) if end < n else k
        stats["spans"].append([offset + min(a, k), offset + min(max(b, a), k)])
        offset += k
    if not balls:
        raise ValueError("no frames in periods 1 or 2")
    ball = np.concatenate(balls)
    match = idsse.Match(
        match_id=str(meta["id"]), competition="FIFA World Cup 2022",
        home_name=meta["homeTeam"]["shortName"], away_name=meta["awayTeam"]["shortName"],
        pitch=pitch_of(meta), frame_number=np.concatenate(numbers).astype(np.int32),
        half=np.concatenate(halves), ball=ball, ball_speed=np.full(len(ball), np.nan, dtype=np.float32),
        status=np.concatenate(labels_s), possession=np.concatenate(labels_p),
        person_ids=tuple("{}{}".format("hA"[s], j) for s, j in persons),
        person_team=np.array([s for s, _ in persons], dtype=np.int8), players=np.concatenate(plays))
    stats["fps"] = fps
    return match, stats


_KEYS = ("frame_number", "half", "ball", "ball_speed", "status", "possession", "person_ids", "person_team",
         "players")


def _from_cache(meta, cached):
    return idsse.Match(
        match_id=str(meta["id"]), competition="FIFA World Cup 2022",
        home_name=meta["homeTeam"]["shortName"], away_name=meta["awayTeam"]["shortName"],
        pitch=pitch_of(meta), frame_number=cached["frame_number"], half=cached["half"], ball=cached["ball"],
        ball_speed=cached["ball_speed"], status=cached["status"], possession=cached["possession"],
        person_ids=tuple(str(p) for p in cached["person_ids"]), person_team=cached["person_team"],
        players=cached["players"])


def load(game_id, use_cache=True):
    """One match (Match-compatible), from the cache on D: when possible. Also writes the stats json."""
    game_id = str(game_id)
    meta = read_meta(game_id)
    path, spath = cache_path(game_id), cache_path(game_id, "stats", "json")
    if use_cache and os.path.isfile(path) and os.path.isfile(spath):
        with np.load(path, allow_pickle=False) as cached:
            return _from_cache(meta, {k: cached[k] for k in _KEYS})
    parsed = parse_tracking(_dir("Tracking Data", "{}.jsonl.bz2".format(game_id)))
    match, stats = build(meta, parsed, read_events(game_id))
    del parsed
    if use_cache:
        _atomic(path, lambda h: np.savez(h, frame_number=match.frame_number, half=match.half, ball=match.ball,
                                         ball_speed=match.ball_speed, status=match.status,
                                         possession=match.possession, person_ids=np.array(match.person_ids),
                                         person_team=match.person_team, players=match.players))
        _atomic(spath, lambda h: json.dump(stats, h), "w")
    return match


def stats(game_id):
    """The aggregates written next to the cached match (computes them when missing)."""
    path = cache_path(str(game_id), "stats", "json")
    if not os.path.isfile(path):
        load(game_id)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------- damage decision (PFF_PLAN section 4)
ATD_MISSING = 0.407
DECISION_GAMES = 16


def damage_decision():
    """{"m", "applies", "level"} from the pooled ball-missing share (VISIBLE only) of the first 16 games;
    computed once and stored in the cache dir. m >= 0.357: no added damage; else level = 1-(1-0.407)/(1-m)."""
    path = os.path.join(CACHE_DIR, "damage_decision.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    total = visible = 0
    for game_id in game_ids()[:DECISION_GAMES]:
        s = stats(game_id)
        total += s["period_frames"]
        visible += s["ball_visible"]
    m = 1.0 - visible / total
    applies = m < ATD_MISSING - 0.05
    out = {"m": m, "applies": bool(applies), "level": (1 - (1 - ATD_MISSING) / (1 - m)) if applies else None,
           "games": DECISION_GAMES}
    _atomic(path, lambda h: json.dump(out, h), "w")
    return out
