"""Read the tactical ATD and Smart Tagging into numpy arrays the rest of the pipeline works on.

Same raw format starter/baseline.py reads -- "<frame>:<player>;<player>;...:<ball>", each group
"x,y,speed", NaN for a missing observation -- but kept whole rather than reduced to one nearest
player per frame, because the rules need every position and every speed.

Arrays are indexed by video frame (ATD frame_id - 1, DATA.md) and are in metres. Parsing 199,319
lines takes a minute or two, so the result is cached under data/cache/, which is gitignored with
the rest of data/ -- it is derived from the match data and must never be committed either.
"""

import os
import sys

import numpy as np

from pipeline.config import ROOT, Config

sys.path.insert(0, ROOT)
from games import GAME, require, source_frames  # noqa: E402
from starter.baseline import team_of_each_track  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tagging  # noqa: E402

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"  # first character of "NaN"


class Tracking(object):
    """Everything per frame the rules use. n = video frames, s = track slots.

    players  (n, s, 2) float32  metres, NaN where the slot is not seen
    speeds   (n, s)    float32  m/s as reported by the ATD, NaN where not seen
    ball     (n, 2)    float32  metres, NaN where the ball is not seen
    team     (s,)      int8     H or A for each slot, from the ATD metadata
    """

    def __init__(self, players, speeds, ball, team):
        self.players = players
        self.speeds = speeds
        self.ball = ball
        self.team = team

    def __len__(self):
        return len(self.ball)


def read_atd(raw_path, team, n_frames, cfg=Config()):
    """Parse the raw ATD into a Tracking, coordinates scaled from 0-1 to metres.

    One Python list per line and one numpy row assignment per frame: assigning element by
    element into numpy is several times slower over ~9 million player groups.
    """
    n_slots = len(team)
    players = np.full((n_frames, n_slots * 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    nan_row = [float("nan")] * (n_slots * 3)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # frame_id counts from 1, video frames from 0
            if not 0 <= frame < n_frames:
                continue

            row = list(nan_row)
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if slot >= n_slots or not group or group[0] == MISSING:
                    continue
                fields = group.split(",")
                row[3 * slot:3 * slot + 3] = [float(value) for value in fields[:3]]
            players[frame] = row

            group = parts[2].strip()
            if group and group[0] != MISSING:
                fields = group.split(",")
                ball[frame] = (float(fields[0]), float(fields[1]))

    players = players.reshape(n_frames, n_slots, 3)
    scale = np.array([cfg.LAPANGAN_P, cfg.LAPANGAN_L], dtype=np.float32)
    return Tracking(players=players[:, :, :2] * scale,
                    speeds=np.ascontiguousarray(players[:, :, 2]),
                    ball=ball * scale,
                    team=np.asarray(team, dtype=np.int8))


def load_tracking(game=GAME, data_dir=None, cfg=Config(), cache=True):
    """The tactical ATD as a Tracking, from the cache when the raw file has not changed."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team = team_of_each_track(meta_path)
    n_frames = source_frames("tactical", game)

    stat = os.stat(raw_path)
    key = "{}_{}_{}_{}x{}".format(game, stat.st_size, int(stat.st_mtime),
                                  cfg.LAPANGAN_P, cfg.LAPANGAN_L)
    cache_path = os.path.join(os.path.dirname(os.path.dirname(raw_path)), "cache",
                              "atd_{}.npz".format(key))
    if cache and os.path.isfile(cache_path):
        stored = np.load(cache_path)
        return Tracking(stored["players"], stored["speeds"], stored["ball"], stored["team"])

    print("reading {} ...".format(os.path.basename(raw_path)))
    tracking = read_atd(raw_path, team, n_frames, cfg)
    if cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, players=tracking.players, speeds=tracking.speeds,
                 ball=tracking.ball, team=tracking.team)
    return tracking


def load_events(game=GAME, data_dir=None):
    """Smart Tagging instances, [{code, start_frame, end_frame, tags}], in video frames."""
    return load_smart_tagging(game, data_dir)
