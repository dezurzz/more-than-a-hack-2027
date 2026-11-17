"""Shared test fixtures. Everything here is synthetic: no test ever opens the match data.

The synthetic ATD has the real file's format (EPTS metadata XML + raw TXT) and its awkward cases:
untracked slots, frames without a ball, a slot with x but no y, a frame with no line at all, and a
trailing blank line. Small enough to compare our code against starter/baseline.py line by line.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah import config  # noqa: E402,F401  (puts the starter modules on sys.path)

N_SLOTS = 6
N_FRAMES = 300
PITCH = (105.3, 68.0)
TEAM_OF_SLOT = [0, 0, 0, 1, 1, 1]
MISSING_FRAME_ID = 50  # no line for this frame at all

META_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Metadata>
  <Teams>
    <Team id="T1"><Name>Home FC</Name></Team>
    <Team id="T2"><Name>Away FC</Name></Team>
  </Teams>
  <Players>
{}
  </Players>
</Metadata>
"""


def _object(rng, x_only=False):
    if x_only:
        return "{:.6f},NaN,{:.3f}".format(rng.rand(), 8 * rng.rand())
    return "{:.6f},{:.6f},{:.3f}".format(rng.rand(), rng.rand(), 8 * rng.rand())


def write_synthetic_atd(folder, seed=0):
    """(meta_path, raw_path) of a small random ATD in folder."""
    rng = np.random.RandomState(seed)
    players = "\n".join('    <Player id="P{}" teamId="T{}"/>'.format(slot + 1, team + 1)
                        for slot, team in enumerate(TEAM_OF_SLOT))
    meta_path = os.path.join(str(folder), "synthetic_atd.xml")
    with open(meta_path, "w") as handle:
        handle.write(META_XML.format(players))

    lines = []
    for frame_id in range(1, N_FRAMES + 1):
        if frame_id == MISSING_FRAME_ID:
            continue
        groups = []
        for _ in range(N_SLOTS):
            draw = rng.rand()
            groups.append("NaN,NaN,NaN" if draw < 0.15 else _object(rng, x_only=draw < 0.18))
        draw = rng.rand()
        ball = "NaN,NaN,NaN" if draw < 0.3 else _object(rng, x_only=draw < 0.32)
        lines.append("{}:{}:{}\n".format(frame_id, ";".join(groups), ball))
    lines.append("\n")

    raw_path = os.path.join(str(folder), "synthetic_atd.txt")
    with open(raw_path, "w") as handle:
        handle.writelines(lines)
    return meta_path, raw_path


@pytest.fixture(autouse=True)
def atd_teams(monkeypatch):
    """The default TEAM_SOURCE "colour" reads the match's shirt cache: never in a test."""
    monkeypatch.setattr(config, "TEAM_SOURCE", "atd")


@pytest.fixture
def synthetic_atd(tmp_path):
    return write_synthetic_atd(tmp_path)
