"""Tahap 3 -- DFL possession relabelled to the challenge's definition, with DFL's own event data.

DFL flips BallPossession on every touch. The challenge does not (CHALLENGE.md, "Possession
rules"): possession changes only when the other team gains *controlled* possession, and

    a simple touch of the ball without control
    a deflection of a pass that does not intercept it
    an aerial duel won without subsequent control of the ball by that team

do not count; a contested or loose ball stays with the last team. DFL's event feed
(events_raw, in the same figshare article) says which touches were controlled, so a DFL spell of
the other team is kept only if that team has a control event in it -- or if it lasts KEEP_S or
more, the challenge's own proxy ("a player keeps the ball for a few seconds"), which also covers
carries the event feed does not log. Every other spell goes back to the team that had the ball.

Control events, by DFL type:

    Play (pass, cross, set-piece delivery), ShotAtGoal, BallClaiming      the acting team
    OtherBallAction                                                       the acting team,
                                                                          unless a clearance
    TacklingGame  winner, if it won the ball (claimed, retained, dribbled around, PossessionChange)
                  either side that had the ball under control going in

Untouched: stoppages (D comes from BallStatus, the referee's call, as in the challenge), the
timing of every kept change, and any spell right after a stoppage or a half's start -- a restart
always has a clear possessor.
"""

import datetime
import json
import os
import xml.etree.ElementTree as ElementTree

import numpy as np

from pipeline.config import A, H
from pipeline.dfl import (CACHE_DIR, DFL_DIR, FPS, SECTIONS, load_match, match_files,
                          read_info)
from pipeline.runs import value_runs

KEEP_S = 2.0        # an opposing spell this long counts as control even without an event
TOLERANCE_S = 1.0   # event clock vs BallPossession flag: an event this close still supports a spell
WON_BALL = ("ballClaimed", "ballControlRetained", "dribbledAround")


def events_file(match, dfl_dir=DFL_DIR):
    for name in os.listdir(dfl_dir):
        if "DFL-MAT-" + match in name and "_events_raw_" in name:
            return os.path.join(dfl_dir, name)
    raise IOError("match {}: no events_raw file in {}".format(match, dfl_dir))


def parse_time(text):
    return datetime.datetime.fromisoformat(text)


def section_clocks(positions_path):
    """{section: (first frame N, its UTC time)} from the ball's FrameSet of each half."""
    clocks, section = {}, None
    with open(positions_path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<FrameSet ") and 'TeamId="BALL"' in line:
                section = line.split('GameSection="', 1)[1].split('"', 1)[0]
            elif section is not None and line.startswith("<Frame "):
                n = int(line.split('N="', 1)[1].split('"', 1)[0])
                clocks[section] = (n, line.split('T="', 1)[1].split('"', 1)[0])
                section = None
                if len(clocks) == len(SECTIONS):
                    break
    return {key: (n, parse_time(t)) for key, (n, t) in clocks.items()}


def control_teams(event):
    """Team ids that had the ball under control in one <Event> element."""
    out = []
    for element in event.iter():
        tag, a = element.tag, element.attrib
        if tag in ("Play", "ShotAtGoal", "BallClaiming") and "Team" in a:
            out.append(a["Team"])
        elif tag == "OtherBallAction" and a.get("DefensiveClearance") != "true" and "Team" in a:
            out.append(a["Team"])
        elif tag == "TacklingGame":
            if (a.get("WinnerResult") in WON_BALL or a.get("PossessionChange") == "true"
                    or a.get("WinnerRole") == "withBallControl"):
                out.append(a.get("WinnerTeam"))
            if a.get("LoserRole") == "withBallControl":
                out.append(a.get("LoserTeam"))
    return [team for team in out if team]


def control_events(match, halves, dfl_dir=DFL_DIR, cache=True):
    """(frames, teams): every control event as an index into load_match's arrays, and H or A."""
    cache_path = os.path.join(CACHE_DIR, "dfl_{}_control.json".format(match))
    if cache and os.path.isfile(cache_path):
        with open(cache_path) as handle:
            stored = json.load(handle)
        return np.array(stored["frames"], dtype=np.int64), np.array(stored["teams"], dtype=np.int8)

    info_path, positions_path = match_files(match, dfl_dir)
    info = read_info(info_path)
    team_code = {info["home"]: H, info["guest"]: A}
    clocks = section_clocks(positions_path)
    starts = [(clocks[section], first, last) for section, (_, first, last) in zip(SECTIONS, halves)]

    frames, teams = [], []
    for event in ElementTree.parse(events_file(match, dfl_dir)).getroot():
        if "EventTime" not in event.attrib:
            continue
        moment = parse_time(event.attrib["EventTime"])
        for (_, t0), first, last in starts:
            index = first + int(round((moment - t0).total_seconds() * FPS))
            if first <= index <= last:
                for team in control_teams(event):
                    if team in team_code:
                        frames.append(index)
                        teams.append(team_code[team])
                break
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, "w") as handle:
            json.dump({"frames": frames, "teams": teams}, handle)
    return np.array(frames, dtype=np.int64), np.array(teams, dtype=np.int8)


def relabel(labels, halves, frames, teams, keep_s=KEEP_S, tolerance_s=TOLERANCE_S, fps=FPS):
    """Labels with every uncontrolled opposing spell handed back to the team that had the ball."""
    out = labels.copy()
    keep, tolerance = int(keep_s * fps), int(tolerance_s * fps)
    letter = {H: "H", A: "A"}
    controlled = {code: np.sort(frames[teams == code]) for code in (H, A)}
    by_letter = {letter[code]: controlled[code] for code in (H, A)}
    for _, first, last in halves:
        changed = True
        while changed:
            changed = False
            runs = value_runs(out[first:last + 1])
            for i in range(1, len(runs)):
                a, b, value = runs[i]
                previous = runs[i - 1][2]
                if value not in "HA" or previous not in "HA" or b - a >= keep:
                    continue
                events = by_letter[value]
                lo, hi = np.searchsorted(events, [first + a - tolerance, first + b + tolerance])
                if hi > lo:
                    continue
                out[first + a:first + b] = previous
                changed = True
                break  # runs have merged; recompute them
    return out


def load_relabelled(match, **kwargs):
    """load_match with the labels relabelled: (Tracking, labels, halves, cfg)."""
    tracking, labels, halves, cfg = load_match(match)
    frames, teams = control_events(match, halves)
    return tracking, relabel(labels, halves, frames, teams, **kwargs), halves, cfg


LABELS = ("dfl", "challenge")


def loader(labels="dfl"):
    """load_match for DFL's own possession, load_relabelled for the challenge's."""
    if labels not in LABELS:
        raise ValueError("labels must be one of {}".format(LABELS))
    return load_relabelled if labels == "challenge" else load_match
