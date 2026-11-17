"""Viterbi decoding of per-frame evidence into H / A / D with switching penalties (plan P3).

A half is a path through three states. Each frame adds its evidence for the state the path is in
(log-probability-like scores, higher = more likely) and every change of state pays a penalty. The
path with the best total wins. Two properties follow, and they are why this replaces the majority
vote:

  - A change is taken only when the evidence after it outweighs its penalty, so one noisy frame
    cannot split a possession, but a short possession with strong evidence survives.
  - The whole half is decoded at once, so a change lands where the evidence turns, not half a
    window later: no smoothing lag.

Determinism: the inner loop is plain Python floats with fixed tie-breaking (lowest state index
wins), and scores are rounded to SCORE_DECIMALS first, so a last-bit difference in np.log between
two CPUs cannot flip a decision. Runs in about a second per half.
"""

import numpy as np

STATES = ("H", "A", "D")
H, A, D = 0, 1, 2

# Params
SCORE_DECIMALS = 6
PROBABILITY_FLOOR = 1e-4  # no single frame may be certain; keeps every log finite


def transition_costs(switch, to_dead, from_dead):
    """3x3 cost of moving from state i (row) to j (column). Staying costs nothing.

    switch:    H <-> A, a change of possession
    to_dead:   H or A -> D, the ball going out of play
    from_dead: D -> H or A, a restart

    switch must be below to_dead + from_dead. Otherwise a change of possession is cheaper through
    one or two frames of D (H -> D -> A costs to_dead + from_dead plus a few frames of D evidence)
    than directly, and the decoder routes every change through a fake stoppage: the switching
    penalty stops doing anything and the number of D stretches explodes.
    """
    if switch >= to_dead + from_dead:
        raise ValueError("switch {:g} must be below to_dead + from_dead = {:g}: changes of "
                         "possession would be routed through fake dead-ball frames".format(
                             switch, to_dead + from_dead))
    return ((0.0, switch, to_dead),
            (switch, 0.0, to_dead),
            (from_dead, from_dead, 0.0))


def scores_from_probabilities(p_dead, p_home):
    """(n, 3) log scores for H, A, D from P(dead) and P(home | alive).

    NaN means "no evidence", and the two NaNs are not the same thing:
      - p_home NaN becomes 0.5: no opinion on the team, so the path keeps its current one.
      - p_dead NaN becomes the floor: no evidence of a stoppage means alive. Mapping it to 0.5
        instead would make D (log 0.5) beat H and A (log 0.5 + log p_home < log 0.5) on every
        such frame, turning any stretch without evidence into dead ball. A missing ball must not
        mean D (the baseline's mistake).
    """
    p_dead = np.clip(np.nan_to_num(np.asarray(p_dead, dtype=np.float64), nan=PROBABILITY_FLOOR),
                     PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
    p_home = np.clip(np.nan_to_num(np.asarray(p_home, dtype=np.float64), nan=0.5),
                     PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
    alive = np.log(1.0 - p_dead)
    return np.column_stack([alive + np.log(p_home), alive + np.log(1.0 - p_home), np.log(p_dead)])


def viterbi(scores, costs):
    """Best state path (int8 codes) for one contiguous span. scores is (n, 3)."""
    scores = np.round(np.asarray(scores, dtype=np.float64), SCORE_DECIMALS)
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite; use scores_from_probabilities for NaN evidence")
    n = len(scores)
    if not n:
        return np.zeros(0, dtype=np.int8)

    rows = scores.tolist()
    pointers = bytearray(3 * n)
    previous = list(rows[0])
    for t in range(1, n):
        evidence = rows[t]
        current = [0.0, 0.0, 0.0]
        for j in range(3):
            best_i, best = 0, previous[0] - costs[0][j]
            for i in (1, 2):
                value = previous[i] - costs[i][j]
                if value > best:  # strict: ties keep the lower index, on every machine
                    best_i, best = i, value
            current[j] = best + evidence[j]
            pointers[3 * t + j] = best_i
        previous = current

    path = np.empty(n, dtype=np.int8)
    state = max(range(3), key=lambda k: (previous[k], -k))
    for t in range(n - 1, 0, -1):
        path[t] = state
        state = pointers[3 * t + state]
    path[0] = state
    return path


# Optional evidence gate. Off everywhere by default.
GATE_FLOOR = 0.2
GATE_SLOPE = 2.0


def evidence_gate(probabilities):
    """(n,) gate g per step from (n, 3) class probabilities.

    g[t] = max(0.2, 1 - 2 * sum_k |p[t,k] - p[t-1,k]|); g[0] = 1 (nothing before it). A steady
    probability gives 1; a sharp reversal gives the floor, which makes a change of state at that
    step cheaper.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    gate = np.ones(len(p))
    if len(p) > 1:
        gate[1:] = np.maximum(GATE_FLOOR, 1.0 - GATE_SLOPE * np.abs(np.diff(p, axis=0)).sum(axis=1))
    return gate


def viterbi_gated(scores, costs, gate):
    """Like `viterbi`, but every transition cost at step t is multiplied by gate[t]."""
    scores = np.round(np.asarray(scores, dtype=np.float64), SCORE_DECIMALS)
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite; use scores_from_probabilities for NaN evidence")
    n = len(scores)
    if not n:
        return np.zeros(0, dtype=np.int8)
    gate = np.asarray(gate, dtype=np.float64)
    if len(gate) != n:
        raise ValueError("gate must have one value per step")

    rows = scores.tolist()
    g_list = gate.tolist()
    pointers = bytearray(3 * n)
    previous = list(rows[0])
    for t in range(1, n):
        evidence = rows[t]
        g = g_list[t]
        current = [0.0, 0.0, 0.0]
        for j in range(3):
            best_i, best = 0, previous[0] - costs[0][j] * g
            for i in (1, 2):
                value = previous[i] - costs[i][j] * g
                if value > best:
                    best_i, best = i, value
            current[j] = best + evidence[j]
            pointers[3 * t + j] = best_i
        previous = current

    path = np.empty(n, dtype=np.int8)
    state = max(range(3), key=lambda k: (previous[k], -k))
    for t in range(n - 1, 0, -1):
        path[t] = state
        state = pointers[3 * t + state]
    path[0] = state
    return path


def decode(scores, costs, runs):
    """H/A/D labels over the given half-open runs (one per half), X everywhere else.

    Each run is decoded on its own, so nothing ever carries across halftime.
    """
    labels = np.full(len(scores), "X", dtype="<U1")
    names = np.array(STATES)
    for start, stop in runs:
        labels[start:stop] = names[viterbi(scores[start:stop], costs)]
    return labels
