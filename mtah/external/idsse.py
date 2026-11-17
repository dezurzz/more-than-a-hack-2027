"""One IDSSE match (DFL position XML + match information XML) -> numpy arrays and H / A / D / X labels.

Why our own reader and not kloppy: kloppy 3.18.0 maps `BallPossession != 1` to the away team, so an
alive frame with no possession flag would silently become A. Those frames are labelled X and counted. Kloppy stays useful as a cross-check (tests/).

Coordinates stay in metres, centred on the pitch (x along the length, y across), as delivered.
Converting to the ATD's normalised 0-1 convention is the next step's job (convert.py).

Rows are the frames of the ball's two FrameSets (first and second half), in order. A frame missing
from a half's ball record is a row with status -1 (no ball record), which labels as X.
"""

import os
from dataclasses import dataclass

import numpy as np
from lxml import etree

from mtah import config
from mtah.external import files, labels as label_map
import possession_metrics as pm

FORMAT_VERSION = 2          # 2: ball_speed added
FPS = 25
HOME, AWAY = 0, 1
HALVES = {"firstHalf": 1, "secondHalf": 2}
BALL_TEAM = "BALL"
NO_RECORD = -1


@dataclass
class Match:
    match_id: str
    competition: str
    home_name: str
    away_name: str
    pitch: tuple              # (length, width) in metres
    frame_number: np.ndarray  # (n,) int32, the XML's N
    half: np.ndarray          # (n,) int8, 1 or 2
    ball: np.ndarray          # (n, 3) float32 x, y, z in metres; NaN where there is no ball record
    ball_speed: np.ndarray    # (n,) float32 as delivered (not used: the ball's speed is taken from its positions)
    status: np.ndarray        # (n,) int8 BallStatus: 1 alive, 0 dead, NO_RECORD no ball record
    possession: np.ndarray    # (n,) int8 BallPossession: 1 home, 2 away, 0 none / not given
    person_ids: tuple         # (P,) PersonId
    person_team: np.ndarray   # (P,) int8 HOME / AWAY
    players: np.ndarray       # (n, P, 3) float32 x, y in metres, speed as delivered (km/h, see convert.py); NaN where not tracked

    @property
    def n_frames(self):
        return len(self.frame_number)

    @property
    def runs(self):
        """Half-open [start, stop) row spans of the two halves."""
        out = []
        for half in (1, 2):
            rows = np.flatnonzero(self.half == half)
            if len(rows):
                out.append((int(rows[0]), int(rows[-1]) + 1))
        return out

    def labels(self, mapping=label_map.MAPPING_SECONDS):
        """H / A / D / X codes. mapping = T seconds of contest resolution; None = raw flags."""
        codes = labels(self.status, self.possession)
        if mapping:
            codes = label_map.resolve_contests(codes, self.runs, mapping)
        return codes


def labels(status, possession):
    """possession_metrics codes (H 0, A 1, D 2, X 3) from BallStatus and BallPossession.

    Dead -> D whatever the possession flag says (the flag names whoever restarts, which the
    challenge's truth throws away too). Alive with possession 1 / 2 -> H / A. Alive with no flag, and
    rows without a ball record -> X.
    """
    status = np.asarray(status)
    possession = np.asarray(possession)
    codes = np.full(len(status), pm.X, dtype=np.int8)
    alive = status == 1
    codes[alive & (possession == 1)] = pm.H
    codes[alive & (possession == 2)] = pm.A
    codes[status == 0] = pm.D
    return codes


def read_info(info_path):
    """Team ids and names, competition and pitch size from the match information XML."""
    root = etree.parse(info_path).getroot()
    general = root.find(".//General")
    environment = root.find(".//Environment")
    return {
        "match_id": general.get("MatchId").replace("DFL-MAT-", ""),
        "competition": general.get("CompetitionName"),
        "home_id": general.get("HomeTeamId"),
        "away_id": general.get("GuestTeamId"),
        "home_name": general.get("HomeTeamName"),
        "away_name": general.get("GuestTeamName"),
        "pitch": (float(environment.get("PitchX")), float(environment.get("PitchY"))),
    }


def _num(value):
    return float(value) if value is not None else np.nan


def parse_positions(positions_path, home_id, away_id):
    """Stream the positions XML. Returns (ball, persons): per half the ball's arrays, per person
    (team, {half: arrays}). Referees and other objects are skipped."""
    ball = {}
    persons = {}
    current = None
    for event, elem in etree.iterparse(positions_path, events=("start", "end"),
                                       tag=("FrameSet", "Frame"), huge_tree=True):
        if elem.tag == "FrameSet":
            if event == "start":
                half = HALVES.get(elem.get("GameSection"))
                team = elem.get("TeamId")
                if half is None or team not in (BALL_TEAM, home_id, away_id):
                    current = None
                else:
                    current = (half, team, elem.get("PersonId"), [])
            else:
                if current is not None:
                    half, team, person, rows = current
                    table = np.array(rows, dtype=np.float64).reshape(-1, 7)
                    if team == BALL_TEAM:
                        ball[half] = table
                    else:
                        entry = persons.setdefault(person, [HOME if team == home_id else AWAY, {}])
                        entry[1][half] = table
                current = None
                elem.clear()
            continue
        if event == "end" and current is not None:
            current[3].append((int(elem.get("N")), _num(elem.get("X")), _num(elem.get("Y")),
                               _num(elem.get("Z")), _num(elem.get("S")),
                               _num(elem.get("BallStatus", "nan")),
                               _num(elem.get("BallPossession", "nan"))))
        if event == "end":
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
    return ball, persons


def assemble(info, ball, persons):
    """Parsed tables -> Match, on the rows of the ball's frame numbers (contiguous per half)."""
    numbers, halves = [], []
    for half in sorted(ball):
        table = ball[half]
        low, high = int(table[:, 0].min()), int(table[:, 0].max())
        numbers.append(np.arange(low, high + 1, dtype=np.int32))
        halves.append(np.full(high - low + 1, half, dtype=np.int8))
    frame_number = np.concatenate(numbers)
    half_of_row = np.concatenate(halves)
    n = len(frame_number)

    def rows_of(half, frames):
        start = int(frame_number[half_of_row == half][0])
        return frames.astype(np.int64) - start + np.flatnonzero(half_of_row == half)[0]

    ball_xyz = np.full((n, 3), np.nan, dtype=np.float32)
    ball_s = np.full(n, np.nan, dtype=np.float32)
    status = np.full(n, NO_RECORD, dtype=np.int8)
    possession = np.zeros(n, dtype=np.int8)
    for half, table in ball.items():
        rows = rows_of(half, table[:, 0])
        ball_xyz[rows] = table[:, 1:4]
        ball_s[rows] = table[:, 4]
        status[rows] = np.nan_to_num(table[:, 5], nan=NO_RECORD).astype(np.int8)
        possession[rows] = np.nan_to_num(table[:, 6], nan=0).astype(np.int8)

    ids = tuple(sorted(persons, key=lambda p: (persons[p][0], p)))
    players = np.full((n, len(ids), 3), np.nan, dtype=np.float32)
    for column, person in enumerate(ids):
        for half, table in persons[person][1].items():
            if half not in ball:
                continue
            rows = rows_of(half, table[:, 0])
            keep = (rows >= 0) & (rows < n) & (half_of_row[np.clip(rows, 0, n - 1)] == half)
            players[rows[keep], column, 0] = table[keep, 1]
            players[rows[keep], column, 1] = table[keep, 2]
            players[rows[keep], column, 2] = table[keep, 4]
    return Match(match_id=info["match_id"], competition=info["competition"],
                 home_name=info["home_name"], away_name=info["away_name"], pitch=info["pitch"],
                 frame_number=frame_number, half=half_of_row, ball=ball_xyz, ball_speed=ball_s,
                 status=status,
                 possession=possession, person_ids=ids,
                 person_team=np.array([persons[p][0] for p in ids], dtype=np.int8),
                 players=players)


def load_files(info_path, positions_path):
    info = read_info(info_path)
    ball, persons = parse_positions(positions_path, info["home_id"], info["away_id"])
    return assemble(info, ball, persons)


def load(match_id, data_dir=None, use_cache=True):
    """One match, checked against the manifest, from cache/external/ when possible."""
    info_path = files.verify(match_id, "info", data_dir)
    positions_path = files.verify(match_id, "positions", data_dir)
    cache_path = os.path.join(config.CACHE_DIR, "external",
                              "{}_v{}_{}.npz".format(match_id, FORMAT_VERSION,
                                                     os.path.getsize(positions_path)))
    info = read_info(info_path)
    if use_cache and os.path.isfile(cache_path):
        with np.load(cache_path, allow_pickle=False) as cached:
            return Match(match_id=info["match_id"], competition=info["competition"],
                         home_name=info["home_name"], away_name=info["away_name"],
                         pitch=info["pitch"], frame_number=cached["frame_number"],
                         half=cached["half"], ball=cached["ball"],
                         ball_speed=cached["ball_speed"], status=cached["status"],
                         possession=cached["possession"],
                         person_ids=tuple(str(p) for p in cached["person_ids"]),
                         person_team=cached["person_team"], players=cached["players"])
    match = load_files(info_path, positions_path)
    if use_cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        partial = cache_path + ".partial"
        with open(partial, "wb") as handle:
            np.savez(handle, frame_number=match.frame_number, half=match.half, ball=match.ball,
                     ball_speed=match.ball_speed, status=match.status, possession=match.possession,
                     person_ids=np.array(match.person_ids), person_team=match.person_team,
                     players=match.players)
        os.replace(partial, cache_path)
    return match
