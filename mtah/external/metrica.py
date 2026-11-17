"""Metrica Sports sample data (games 1 and 2, CSV) -> the same `idsse.Match` the rest of the pipeline reads.

Source: github.com/metrica-sports/sample-data ("be responsible with the use of this data; acknowledge the source").
Optical tracking at 25 Hz, coordinates 0-1 with the origin at the top left, pitch 105 x 68 m, tracking and events
synchronised by frame. Games 1 and 2 only; game 3 is in the FIFA/EPTS format and is not read here.

Labels are our own reading of the event list (the data has no per-frame possession), so they are less certain than
IDSSE's flags and are checked by the sanity numbers (dead share, spells per alive minute, median spell) before use:
  * dead from a BALL OUT, a FAULT RECEIVED, a card or a goal (SHOT with subtype GOAL) until the next alive maker;
  * alive otherwise, with the possession of the team of the latest PASS / SHOT / RECOVERY / SET PIECE;
  * BALL LOST and CHALLENGE change nothing (the next RECOVERY or PASS says who has it);
  * rows before the first and after the last event of a half are X.
Ball height is not given (0). Player speed is the finite difference of the position (km/h as IDSSE delivers it).
"""
import os

import numpy as np
import pandas as pd

from mtah import config
from mtah.external import idsse

_HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.environ.get("MTAH_METRICA_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(_HERE))), "work_data", "extra_data", "metrica", "data"))
PITCH = (105.0, 68.0)
GAMES = ("MET1", "MET2")
FPS = 25
ALIVE_MAKERS = ("PASS", "SHOT", "RECOVERY", "SET PIECE")
DEAD_MAKERS = ("BALL OUT", "FAULT RECEIVED", "CARD")


def _raw(game, team):
    path = os.path.join(DIR, "Sample_Game_{}".format(game), "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(game, team))
    return pd.read_csv(path, skiprows=3, header=None)


def read(game):
    """frames (n,), period (n,), players (n, P, 2) in 0-1 (NaN where absent), ball (n, 2), person_team (P,)."""
    home, away = _raw(game, "Home"), _raw(game, "Away")
    assert (home[1].values == away[1].values).all()
    frames = home[1].values.astype(np.int64)
    period = home[0].values.astype(np.int8)

    def block(raw):
        n_players = (raw.shape[1] - 3 - 2) // 2
        xy = raw.iloc[:, 3:3 + 2 * n_players].values.astype(np.float64).reshape(len(raw), n_players, 2)
        ball = raw.iloc[:, 3 + 2 * n_players:3 + 2 * n_players + 2].values.astype(np.float64)
        return xy, ball

    hp, hb = block(home)
    ap, ab = block(away)
    ball = np.where(np.isfinite(hb), hb, ab)
    players = np.concatenate([hp, ap], axis=1)
    team = np.array([idsse.HOME] * hp.shape[1] + [idsse.AWAY] * ap.shape[1], dtype=np.int8)
    return frames, period, players, ball, team


def events(game):
    path = os.path.join(DIR, "Sample_Game_{}".format(game), "Sample_Game_{}_RawEventsData.csv".format(game))
    ev = pd.read_csv(path)
    ev["team"] = ev["Team"].map({"Home": 1, "Away": 2})
    return ev.sort_values(["Start Frame", "End Frame"], kind="stable").reset_index(drop=True)


def label_arrays(ev, frames, period):
    """status (1 alive, 0 dead, -1 none) and possession (0 none, 1 home, 2 away) per tracking row."""
    n = len(frames)
    status = np.full(n, idsse.NO_RECORD, dtype=np.int8)
    possession = np.zeros(n, dtype=np.int8)
    for p in (1, 2):
        rows = np.flatnonzero(period == p)
        sel = ev[ev["Period"] == p]
        if not len(rows) or not len(sel):
            continue
        first, last = sel["Start Frame"].min(), sel["End Frame"].max()
        mark = []                                                  # (frame, kind, team) in time order
        for _, e in sel.iterrows():
            kind, sub = str(e["Type"]), str(e["Subtype"])
            if kind in ALIVE_MAKERS:
                if kind == "SHOT" and "GOAL" in sub.upper():
                    mark.append((int(e["Start Frame"]), "dead", 0))
                else:
                    mark.append((int(e["Start Frame"]), "alive", int(e["team"])))
            elif kind in DEAD_MAKERS:
                mark.append((int(e["Start Frame"]), "dead", 0))
        fr = frames[rows]
        state, owner, k = "alive", 0, 0
        out_s = np.full(len(rows), idsse.NO_RECORD, dtype=np.int8)
        out_p = np.zeros(len(rows), dtype=np.int8)
        for i, f in enumerate(fr):
            while k < len(mark) and mark[k][0] <= f:
                state = mark[k][1]
                owner = mark[k][2] if state == "alive" else 0
                k += 1
            if f < first or f > last:
                continue
            out_s[i] = 1 if state == "alive" else 0
            out_p[i] = owner if state == "alive" else 0
        status[rows] = out_s
        possession[rows] = out_p
    return status, possession


def build(game):
    frames, period, players, ball, team = read(game)
    n = len(frames)
    length, width = PITCH

    def to_m(a):
        return np.stack([(a[..., 0] - 0.5) * length, (0.5 - a[..., 1]) * width], axis=-1)

    pl_m = to_m(players).astype(np.float32)
    ball_m = np.full((n, 3), np.nan, dtype=np.float32)
    ball_m[:, :2] = to_m(ball)
    ball_m[np.isfinite(ball_m[:, 0]), 2] = 0.0
    speed = np.full(players.shape[:2], np.nan, dtype=np.float32)
    speed[1:-1] = np.sqrt(((pl_m[2:] - pl_m[:-2]) ** 2).sum(-1)) / (2.0 / FPS) * 3.6
    pl3 = np.concatenate([pl_m, speed[..., None]], axis=-1)
    status, possession = label_arrays(events(game), frames, period)
    ball_speed = np.full(n, np.nan, dtype=np.float32)
    ball_speed[1:-1] = np.sqrt(((ball_m[2:, :2] - ball_m[:-2, :2]) ** 2).sum(-1)) / (2.0 / FPS) * 3.6
    return idsse.Match(match_id="MET" + str(game), competition="Metrica sample", home_name="Home", away_name="Away",
                       pitch=PITCH, frame_number=frames.astype(np.int32), half=period, ball=ball_m,
                       ball_speed=ball_speed, status=status, possession=possession,
                       person_ids=tuple("p{}".format(i) for i in range(pl3.shape[1])), person_team=team, players=pl3)


def _from_arrays(match_id, c):
    return idsse.Match(match_id=str(match_id), competition="Metrica sample", home_name="Home", away_name="Away",
                       pitch=PITCH, frame_number=c["frame_number"], half=c["half"], ball=c["ball"],
                       ball_speed=c["ball_speed"], status=c["status"], possession=c["possession"],
                       person_ids=tuple(str(p) for p in c["person_ids"]), person_team=c["person_team"],
                       players=c["players"])


def load(match_id):
    """match_id 'MET1' / 'MET2'; cached in cache/external/."""
    game = str(match_id)[3:]
    path = os.path.join(config.CACHE_DIR, "external", "{}_metrica_v1.npz".format(match_id))
    if os.path.isfile(path):
        with np.load(path, allow_pickle=False) as cached:
            return _from_arrays(match_id, cached)
    m = build(game)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    partial = path + ".partial"
    with open(partial, "wb") as handle:
        np.savez(handle, frame_number=m.frame_number, half=m.half, ball=m.ball, ball_speed=m.ball_speed,
                 status=m.status, possession=m.possession, person_ids=np.array(m.person_ids),
                 person_team=m.person_team, players=m.players)
    os.replace(partial, path)
    return m
