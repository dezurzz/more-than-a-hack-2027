"""Tahap 3 -- Sportec Open DFL matches converted to the same arrays as the ATD, plus H/A/D labels.

The dataset (Bassek et al. 2025, figshare 28196177, CC BY) is seven Bundesliga matches of
multi-camera tracking at 25 Hz. Download the files of a match into external/dfl/ -- matchinformation
and positions_raw_observed are the ones used here:

    https://springernature.figshare.com/articles/dataset/28196177

Unlike the competition match this data may be read and run on freely: the T&C restrict the
BAR-ATH assets only.

Ground truth comes straight from the ball frames. DFL writes two attributes on every one of them,
BallStatus (1 alive, 0 dead) and BallPossession (1 home, 2 guest), so

    BallStatus 0             -> D
    BallStatus 1, team 1 / 2 -> H / A

which is how the competition labels are described as well (live/dead flag plus team in
possession). DFL's possession is its own definition, not ours, so the labels are close to the
competition's, not identical.

Layout of the result, chosen so label_match() runs on it unchanged:

    frames   the two halves back to back, with HALFTIME_GAP frames of X between them, so the
             scorer sees two graded runs exactly like BAR-ATH (possession_metrics.valid_runs)
    origin   a pitch corner, x along the length, as in pipeline/data.py (DFL is centred)
    speeds   m/s (DFL writes km/h)
    slots    one per player PersonId, NaN while off the pitch; referees dropped

Parsing ~4 million lines takes a while, so each match is cached, compressed, under external/cache/
"""

import os
import re

import numpy as np

from pipeline.config import ROOT, A, H, Config
from pipeline.data import Tracking

DFL_DIR = os.path.join(ROOT, "external", "dfl")
CACHE_DIR = os.path.join(ROOT, "external", "cache")
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")

FPS = 25
HALFTIME_GAP = 25  # frames of X between the halves; any length > 0 separates the graded runs
SECTIONS = ("firstHalf", "secondHalf")
KMH = 1 / 3.6

FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)"[^>]*PersonId="([^"]+)"')
FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"[^>]*? S="([-\d.]+)"')
BALL_FLAGS = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')
ATTRIBUTE = re.compile(r'(\w+)="([^"]*)"')


def match_files(match, dfl_dir=DFL_DIR):
    """(matchinformation, positions) paths of a match, by its short id such as 'J03WMX'."""
    found = {}
    for name in os.listdir(dfl_dir):
        if "DFL-MAT-" + match not in name:
            continue
        if "_matchinformation_" in name:
            found["info"] = os.path.join(dfl_dir, name)
        elif "_positions_raw_observed_" in name:
            found["positions"] = os.path.join(dfl_dir, name)
    missing = [key for key in ("info", "positions") if key not in found]
    if missing:
        raise IOError("match {}: no {} file in {} -- see the download link in pipeline/dfl.py"
                      .format(match, " / ".join(missing), dfl_dir))
    return found["info"], found["positions"]


def read_info(info_path):
    """{'home': TeamId, 'guest': TeamId, 'pitch': (x, y)} from the matchinformation file."""
    with open(info_path, encoding="utf-8") as handle:
        text = handle.read()
    general = dict(ATTRIBUTE.findall(re.search(r"<General [^>]*>", text).group(0)))
    environment = dict(ATTRIBUTE.findall(re.search(r"<Environment [^>]*>", text).group(0)))
    return {"home": general["HomeTeamId"], "guest": general["GuestTeamId"],
            "pitch": (float(environment["PitchX"]), float(environment["PitchY"]))}


def read_positions(positions_path, info):
    """Raw per-section series: {section: {'ball': {N: (x, y, poss, status)}, person: {N: ...}}}.

    Kept as dicts of frame number while reading, because a player's FrameSet covers only the
    frames they were on the pitch and the ball's FrameSet is the only one that spans a section.
    """
    sections = {section: {} for section in SECTIONS}
    teams = {}
    current = None
    with open(positions_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                if current is None:
                    continue
                frame = FRAME.match(line)
                n, x, y, s = (int(frame.group(1)), float(frame.group(2)),
                              float(frame.group(3)), float(frame.group(4)))
                if current[1] == "BALL":
                    flags = BALL_FLAGS.search(line)
                    current[2][n] = (x, y, s, int(flags.group(1)), int(flags.group(2)))
                else:
                    current[2][n] = (x, y, s)
            elif line.startswith("<FrameSet "):
                section, team, person = FRAMESET.match(line).groups()
                if section not in sections or team not in (info["home"], info["guest"], "BALL"):
                    current = None  # referees
                    continue
                key = "ball" if team == "BALL" else person
                if team != "BALL":
                    teams[person] = H if team == info["home"] else A
                current = (section, team, sections[section].setdefault(key, {}))
    return sections, teams


def build_match(sections, teams, pitch):
    """Tracking, labels (one of H/A/D/X per frame) and halves [(half, first, last)] inclusive."""
    people = sorted(teams, key=lambda person: (teams[person], person))
    team = np.array([teams[person] for person in people], dtype=np.int8)
    offset = np.array([pitch[0] / 2, pitch[1] / 2], dtype=np.float32)

    blocks, halves, cursor = [], [], 0
    for half, section in enumerate(SECTIONS, start=1):
        ball_series = sections[section]["ball"]
        first_n, last_n = min(ball_series), max(ball_series)
        n = last_n - first_n + 1

        ball = np.full((n, 2), np.nan, dtype=np.float32)
        labels = np.full(n, "D", dtype="<U1")
        for frame_n, (x, y, _, possession, status) in ball_series.items():
            i = frame_n - first_n
            ball[i] = (x, y)
            if status == 1:
                labels[i] = "H" if possession == 1 else "A"

        players = np.full((n, len(people), 2), np.nan, dtype=np.float32)
        speeds = np.full((n, len(people)), np.nan, dtype=np.float32)
        for slot, person in enumerate(people):
            for frame_n, (x, y, s) in sections[section].get(person, {}).items():
                i = frame_n - first_n
                if 0 <= i < n:
                    players[i, slot] = (x, y)
                    speeds[i, slot] = s * KMH

        if blocks:
            gap = HALFTIME_GAP
            blocks.append((np.full((gap, len(people), 2), np.nan, dtype=np.float32),
                           np.full((gap, len(people)), np.nan, dtype=np.float32),
                           np.full((gap, 2), np.nan, dtype=np.float32),
                           np.full(gap, "X", dtype="<U1")))
            cursor += gap
        blocks.append((players + offset, speeds, ball + offset, labels))
        halves.append((half, cursor, cursor + n - 1))
        cursor += n

    players, speeds, ball, labels = (np.concatenate(parts) for parts in zip(*blocks))
    return Tracking(players, speeds, ball, team), labels, halves


def load_match(match, dfl_dir=DFL_DIR, cache=True):
    """(Tracking, labels, halves, cfg) for one DFL match; cfg carries its pitch size."""
    info_path, positions_path = match_files(match, dfl_dir)
    info = read_info(info_path)
    cfg = Config(FPS=FPS, LAPANGAN_P=info["pitch"][0], LAPANGAN_L=info["pitch"][1],
                 PAKAI_SMART_TAGGING=False)

    cache_path = os.path.join(CACHE_DIR, "dfl_{}.npz".format(match))
    if cache and os.path.isfile(cache_path):
        stored = np.load(cache_path)
        tracking = Tracking(stored["players"], stored["speeds"], stored["ball"], stored["team"])
        halves = [tuple(int(v) for v in row) for row in stored["halves"]]
        return tracking, stored["labels"], halves, cfg

    print("reading {} ...".format(os.path.basename(positions_path)))
    sections, teams = read_positions(positions_path, info)
    tracking, labels, halves = build_match(sections, teams, info["pitch"])
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache_path, players=tracking.players, speeds=tracking.speeds,
                 ball=tracking.ball, team=tracking.team, labels=labels, halves=np.array(halves))
    return tracking, labels, halves, cfg


def available_matches(dfl_dir=DFL_DIR):
    """The MATCHES whose files are present in dfl_dir."""
    names = os.listdir(dfl_dir) if os.path.isdir(dfl_dir) else []
    return [match for match in MATCHES
            if any("DFL-MAT-" + match in name and "_positions_" in name for name in names)]
