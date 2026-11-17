"""Mapping of external (IDSSE) labels to the challenge's convention.

DFL flags possession at every touch; the challenge counts controlled possession only (CHALLENGE.md
"what does not count"). A run of consecutive H / A spells that are each shorter than T is a contest:

- the spell before it (a long one) and the spell after it are the same team, or the ball goes dead
  (or the half ends) after it: the whole contest goes to the team before (touches, deflections);
- a different team takes the ball after it: control passes at the first touch of that team,
  the contest goes to the team before until then and to the winner from there on;
- right after a restart (or at the start of a half) the first spell of the contest keeps its team,
  since a restart has a clear possessor, and the rest is resolved as above.

Dead ball and spells of T or longer never change. Applies to external labels only, never to our
BAR-ATH output.
"""

import numpy as np

import possession_metrics as pm

FPS = 25
MAPPING_SECONDS = 1.5
SPELL = (pm.H, pm.A)


def resolve_contests(codes, runs, min_seconds=MAPPING_SECONDS, fps=FPS):
    """Copy of codes (possession_metrics codes) with contests resolved inside each run.

    min_seconds 0 or None changes nothing. Nothing crosses a run boundary.
    """
    out = np.array(codes, copy=True)
    if not min_seconds:
        return out
    min_frames = int(round(min_seconds * fps))
    for start, stop in runs:
        segs = pm.segments(codes, [(start, stop)])
        i = 0
        while i < len(segs):
            a, b, k = segs[i]
            if k not in SPELL or b - a >= min_frames:
                i += 1
                continue
            j = i
            while j < len(segs) and segs[j][2] in SPELL and segs[j][1] - segs[j][0] < min_frames:
                j += 1                                  # segs[i:j] is one contest
            before = segs[i - 1][2] if i > 0 else pm.X
            after = segs[j][2] if j < len(segs) else pm.X
            contest = segs[i:j]
            if before not in SPELL:                     # after a restart / at a half's start
                before = contest[0][2]
                contest = contest[1:]
            if after in SPELL and after != before:
                first = next((n for n, s in enumerate(contest) if s[2] == after), len(contest))
                for n, (sa, sb, _) in enumerate(contest):
                    out[sa:sb] = before if n < first else after
            else:
                for sa, sb, _ in contest:
                    out[sa:sb] = before
            i = j
    return out
