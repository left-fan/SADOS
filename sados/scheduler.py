"""Dynamic objective scheduler utilities."""

from __future__ import annotations


def linear_schedule(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return float(end)
    ratio = min(max(step / (total_steps - 1), 0.0), 1.0)
    return float(start + ratio * (end - start))


def dynamic_supervised_weight(step: int, total_steps: int, initial_weight: float = 0.4) -> float:
    """Linearly increase supervised weight from ``initial_weight`` to 1.

    ``step`` is zero-based. The returned weight multiplies the supervised
    objective; the contrastive objective receives ``1 - s(t)``. With the
    default ``initial_weight=0.4``, adaptation starts with supervised learning
    at 0.4 and contrastive learning at 0.6, then ends at 1.0 and 0.0.
    """

    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    c0 = min(max(float(initial_weight), 0.0), 1.0)
    # Eq. 16: s(t) = min(1, c0 + t / T), with zero-based t.
    return min(1.0, c0 + max(0, int(step)) / total_steps)


def dynamic_contrastive_weight(
    step: int,
    total_steps: int,
    initial_supervised_weight: float = 0.4,
) -> float:
    return 1.0 - dynamic_supervised_weight(step, total_steps, initial_supervised_weight)
