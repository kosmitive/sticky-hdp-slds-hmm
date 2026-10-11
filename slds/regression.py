"""Per-mode regression pairs for the dynamics updates.

Pairs exist only for ``t >= 1``. The first stored state has no parent, so it
is not an inverse-Wishart or matrix-normal observation.
"""

from __future__ import annotations

import numpy as np


def split_regression_pairs(
    modes: np.ndarray,
    states: np.ndarray,
    mode_count: int,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Stack parent and child states for each mode.

    Args:
        modes: Mode sequence, shape ``(traces, steps)``.
        states: Latent states, shape ``(traces, steps, latent)``.
        mode_count: Truncation level, including modes that never occur.

    Returns:
        Parent matrices, child matrices, and the pair count per mode.
        Matrix ``k`` has shape ``(latent, count_k)``.
    """

    latent_dim = states.shape[-1]
    counts = np.bincount(modes[:, 1:].reshape(-1), minlength=mode_count)
    cursor = np.zeros(mode_count, dtype=np.int64)
    parents = [np.empty((latent_dim, int(count))) for count in counts]
    children = [np.empty((latent_dim, int(count))) for count in counts]
    trace_count, step_count, _ = states.shape
    for trace in range(trace_count):
        for step in range(1, step_count):
            mode = int(modes[trace, step])
            index = int(cursor[mode])
            children[mode][:, index] = states[trace, step]
            parents[mode][:, index] = states[trace, step - 1]
            cursor[mode] += 1
    return parents, children, counts
