"""Raw ATD (metadata XML + raw TXT) -> numpy arrays on the submission frame index.

Row i of every array is video frame i, which is submission row i and ATD frame_id i + 1.
Coordinates stay normalised 0-1 as delivered; Tracking.players_m() and ball_m() give metres.

Parsing 199319 lines takes a while, so the result is cached in cache/ (gitignored: it is the
match data in another format) and reused while the source file is unchanged.

Errors name the line number and what is wrong with it, never the line itself. An error message is
the thing most likely to be pasted into a chat with an AI assistant, and the T&C forbid sharing
the data there. For the same reason a failed number conversion is re-raised `from None`: Python's
own message would quote the offending value.
"""

import os
from dataclasses import dataclass

import numpy as np

from mtah import config
from games import GAME, GAMES, require, source_frames  # noqa: E402
from starter.baseline import team_names as read_team_names, team_of_each_track  # noqa: E402

# Params
# Bump when the parse changes, so a cache written by older code is never reused.
FORMAT_VERSION = 1

# "<frame_id>:<player>;<player>;...:<ball>", every object "x,y,speed", NaN when missing.
FIELD_SEP = ":"
GROUP_SEP = ";"
VALUE_SEP = ","
VALUES_PER_OBJECT = 3


class AtdFormatError(ValueError):
    """A raw line that does not have the expected shape. Names the line, never quotes it."""

    def __init__(self, line_number, problem):
        super().__init__("ATD raw line {}: {}".format(line_number, problem))


@dataclass
class Tracking:
    players: np.ndarray       # (n_frames, n_slots, 3) x, y, speed; NaN where not tracked
    ball: np.ndarray          # (n_frames, 3) x, y, speed; NaN where not tracked
    present: np.ndarray       # (n_frames,) True where the raw file has a line for the frame
    team_of_slot: np.ndarray  # (n_slots,) 0 home, 1 away (starter.baseline.HOME / AWAY)
    team_names: tuple         # (home, away) club names
    pitch: tuple              # (length, width) in metres
    n_skipped_lines: int      # lines without exactly three ':' fields, skipped like the baseline

    @property
    def n_frames(self):
        return len(self.present)

    def players_m(self):
        """(n_frames, n_slots, 2) player positions in metres."""
        return self.players[:, :, :2] * np.array(self.pitch)

    def ball_m(self):
        """(n_frames, 2) ball position in metres."""
        return self.ball[:, :2] * np.array(self.pitch)


def parse_raw(raw_path, n_slots, n_frames):
    """(players, ball, present, n_skipped_lines) from the raw TXT. See Tracking for shapes."""
    players = np.full((n_frames, n_slots, VALUES_PER_OBJECT), np.nan)
    ball = np.full((n_frames, VALUES_PER_OBJECT), np.nan)
    present = np.zeros(n_frames, dtype=bool)
    n_player_values = n_slots * VALUES_PER_OBJECT
    skipped = 0

    with open(raw_path) as handle:
        for number, line in enumerate(handle, 1):
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                # starter/baseline.py skips these too. Counted, so a surprise shows up in run.py.
                skipped += 1
                continue

            try:
                frame = int(parts[0]) - 1  # frame_id counts from 1, submissions from 0
            except ValueError:
                raise AtdFormatError(number, "frame id is not an integer") from None
            if not 0 <= frame < n_frames:
                raise AtdFormatError(number, "frame id outside 1..{}".format(n_frames))
            if present[frame]:
                raise AtdFormatError(number, "frame id repeats an earlier line")

            player_values = parts[1].replace(GROUP_SEP, VALUE_SEP).split(VALUE_SEP)
            ball_values = parts[2].split(VALUE_SEP)
            if len(player_values) != n_player_values:
                raise AtdFormatError(number, "{} player values, expected {} ({} slots x {})".format(
                    len(player_values), n_player_values, n_slots, VALUES_PER_OBJECT))
            if len(ball_values) != VALUES_PER_OBJECT:
                raise AtdFormatError(number, "{} ball values, expected {}; see "
                                     "scripts/atd_schema.py".format(len(ball_values),
                                                                    VALUES_PER_OBJECT))
            try:
                # float() per value, exactly as the baseline parses, so values are bit-identical.
                players[frame] = np.array([float(v) for v in player_values]).reshape(
                    n_slots, VALUES_PER_OBJECT)
                ball[frame] = [float(v) for v in ball_values]
            except ValueError:
                raise AtdFormatError(number, "a value is not a number") from None
            present[frame] = True
    return players, ball, present, skipped


def load(game=GAME, data_dir=None, source="tactical", use_cache=True):
    """The tracking of one source, from cache/ when possible."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    return load_files(meta_path, raw_path, source_frames(source, game), GAMES[game]["pitch"],
                      config.CACHE_DIR if use_cache else None)


def load_files(meta_path, raw_path, n_frames, pitch, cache_dir=None):
    """load() on explicit paths, which is what the tests use. No cache when cache_dir is None."""
    team_of_slot = team_of_each_track(meta_path)
    cache_path = _cache_path(cache_dir, raw_path, n_frames) if cache_dir else None

    if cache_path and os.path.isfile(cache_path):
        with np.load(cache_path) as cached:
            players, ball, present = cached["players"], cached["ball"], cached["present"]
            skipped = int(cached["n_skipped_lines"])
    else:
        players, ball, present, skipped = parse_raw(raw_path, len(team_of_slot), n_frames)
        if cache_path:
            os.makedirs(cache_dir, exist_ok=True)
            # Write then rename, so an interrupted run never leaves a half-written cache behind.
            partial = cache_path + ".partial"
            with open(partial, "wb") as handle:
                np.savez(handle, players=players, ball=ball, present=present,
                         n_skipped_lines=skipped)
            os.replace(partial, cache_path)

    return Tracking(players=players, ball=ball, present=present, team_of_slot=team_of_slot,
                    team_names=read_team_names(meta_path), pitch=tuple(pitch),
                    n_skipped_lines=skipped)


def _cache_path(cache_dir, raw_path, n_frames):
    """Cache file keyed on the parser version and the source file's size and modification time."""
    stat = os.stat(raw_path)
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    return os.path.join(cache_dir, "{}_v{}_{}_{}_{}.npz".format(
        stem, FORMAT_VERSION, n_frames, stat.st_size, stat.st_mtime_ns))
