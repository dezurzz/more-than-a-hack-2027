"""MS-TCN stage -- turn per-frame H/A/D probabilities into possessions.

Argmax per frame flickers, and tidy_spells() afterwards only hands short spells to a neighbour.
Decoding instead picks the single labelling of a half that best explains all its probabilities
while paying a price for every change of label and never letting a spell end before it is
MIN_S long -- a hidden semi-Markov reading of "a possession is an interval":

    score(labels) = sum_t log p_t(label_t) - SWITCH * (number of changes)
    subject to    every spell lasts >= min_s (the first spell of a half excepted)

Viterbi finds the maximum exactly. It runs at STEP frames per step (probabilities averaged in
log space over the step, so the switch price keeps its meaning at any step) and the result is
spread back to frames, which costs at most STEP / 2 frames of boundary precision -- far below
the boundary error of the model itself.

With a boundary head (pipeline.tcn) the switch price varies in time: at step t it is
SWITCH * (1 - BETA * b_t), b_t the largest boundary probability in the step, so a change of
label is cheap where the model sees a moment of change and full price elsewhere.

Also here: test-time flips. The model was trained with team swaps and pitch mirrors, so it can be
asked the same question four ways and the answers averaged.
"""

import numpy as np

from pipeline import features
from pipeline.tcn import CLASSES

STEP = 5                      # frames per Viterbi step (0.2 s at 25 Hz)
FLIPS = ((False, False, False), (True, False, False), (False, True, False), (True, True, True))


def flipped_probabilities(predictor, table, halves, flips=FLIPS):
    """Mean probabilities over (swap teams, mirror x, mirror y) variants of the same frames."""
    total = np.zeros((len(table), len(CLASSES)))
    for swap, along_x, along_y in flips:
        variant = features.mirror(table, along_x=along_x, along_y=along_y)
        if swap:
            variant = features.swap_teams(variant)
        p = predictor.probabilities(variant, halves)
        total += p[:, [1, 0, 2]] if swap else p
    return total / len(flips)


def flipped_boundary(predictor, table, halves, flips=FLIPS):
    """Mean boundary probability over the same flips; a change of label survives every flip."""
    total = np.zeros(len(table))
    for swap, along_x, along_y in flips:
        variant = features.mirror(table, along_x=along_x, along_y=along_y)
        total += predictor.boundary(features.swap_teams(variant) if swap else variant, halves)
    return total / len(flips)


def viterbi(log_p, switch, min_steps):
    """Best class per step under a switch price and a minimum spell length, in steps.

    switch is one price, or one per step (the price of entering a new spell at that step).
    """
    n, classes = log_p.shape
    k = max(1, int(min_steps))
    delta = np.full((classes, k), -np.inf)
    delta[:, k - 1] = log_p[0]               # the first spell of a half has no minimum
    prices = np.broadcast_to(np.asarray(switch, dtype=float), (n,))
    off_diagonal = 1.0 - np.eye(classes)
    on_diagonal = np.where(np.eye(classes) > 0, np.inf, 0.0)
    source = np.zeros((n, classes), dtype=np.int8)   # where a new spell of c came from
    grew = np.zeros((n, classes), dtype=bool)         # (c, k-1) reached from (c, k-2), not stayed
    stayed = np.zeros((n, classes), dtype=bool)       # k == 1 only: stayed rather than switched
    for t in range(1, n):
        candidates = delta[:, k - 1][:, None] - (prices[t] * off_diagonal + on_diagonal)
        source[t] = candidates.argmax(0)
        switch_in = candidates.max(0)
        new = np.empty_like(delta)
        if k == 1:
            stay = delta[:, 0]
            stayed[t] = stay >= switch_in
            new[:, 0] = np.where(stayed[t], stay, switch_in)
        else:
            new[:, 0] = switch_in
            new[:, 1:k - 1] = delta[:, :k - 2]
            grew[t] = delta[:, k - 2] > delta[:, k - 1]
            new[:, k - 1] = np.where(grew[t], delta[:, k - 2], delta[:, k - 1])
        delta = new + log_p[t][:, None]

    out = np.empty(n, dtype=np.int8)
    c, j = int(np.argmax(delta[:, k - 1])), k - 1   # a half ends on a full-length spell
    for t in range(n - 1, -1, -1):
        out[t] = c
        if t == 0:
            break
        if k == 1:
            if not stayed[t, c]:
                c = int(source[t, c])
        elif j == 0:
            c, j = int(source[t, c]), k - 1
        elif j < k - 1 or grew[t, c]:
            j -= 1
    return out


def combine(team_probabilities, dead_probabilities):
    """Dead ball from one model, which team has the ball (when it is live) from another.

    P(D) is the dead model's; the rest, 1 - P(D), is split between H and A in the team model's
    proportions. Made for tcn2 (dead ball, which it gets right on the ATD when the ball stays in
    view) with tcn6 (turnovers): see pipeline.disagree.
    """
    h, a, d = (CLASSES.index(c) for c in "HAD")
    out = np.empty_like(team_probabilities)
    live = 1.0 - dead_probabilities[:, d]
    held = team_probabilities[:, h] + team_probabilities[:, a]
    share_h = np.where(held > 0, team_probabilities[:, h] / np.maximum(held, 1e-12), 0.5)
    out[:, d] = dead_probabilities[:, d]
    out[:, h] = live * share_h
    out[:, a] = live * (1.0 - share_h)
    return out


def decode(probabilities, halves, switch=8.0, min_s=1.0, fps=25, step=STEP, boundary=None,
           beta=0.0):
    """H/A/D inside the halves by Viterbi, X outside; boundary (n,) + beta vary the switch price."""
    out = np.full(len(probabilities), "X", dtype="<U1")
    letters = np.array(CLASSES)
    log_all = np.log(np.clip(probabilities, 1e-6, 1.0))
    for _, first, last in halves:
        log_p = log_all[first:last + 1]
        n = len(log_p)
        steps = -(-n // step)
        padded = np.concatenate([log_p, np.repeat(log_p[-1:], steps * step - n, axis=0)])
        coarse = padded.reshape(steps, step, -1).sum(1)
        price = switch
        if boundary is not None and beta:
            b = boundary[first:last + 1]
            b = np.concatenate([b, np.repeat(b[-1:], steps * step - n)]).reshape(steps, step).max(1)
            price = switch * (1.0 - beta * b)
        best = viterbi(coarse, price, round(min_s * fps / step))
        out[first:last + 1] = letters[np.repeat(best, step)[:n]]
    return out
