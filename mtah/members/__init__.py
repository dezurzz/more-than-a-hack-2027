"""Inference code of the two teammates' models , vendored so one `run.py` runs all three.

    mtah.members.dzaky   Dzaky's hybrid MS-TCN (tcn6 for the team, tcn2 for the dead ball)
    mtah.members.wira    Wira's initial TCN (tcn1800, trained on IDSSE alone)

Each has an ATTRIBUTION.md saying where the code came from. Both give per-frame probabilities of the
competition match; mtah.ensemble combines them with ours.
"""

import contextlib


@contextlib.contextmanager
def torch_scope(threads, deterministic=True):
    """torch's thread count and deterministic flag for one member, restored afterwards.

    The members set these process-wide (4 threads), ours needs 1; leaking them would change our own
    member's floating-point sums.
    """
    import torch
    saved = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)
    try:
        yield
    finally:
        torch.set_num_threads(saved[0])
        torch.use_deterministic_algorithms(saved[1])
