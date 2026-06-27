"""Edit-budget LR scheduler for the reflector."""

import math


class LRScheduler:
    """Cosine / linear / constant decay of the per-step edit budget (integer)."""

    def __init__(self, edit_budget: int, min_edit_budget: int, mode: str = "cosine") -> None:
        self._max_lr = max(1, edit_budget)
        self._min_lr = max(1, min(min_edit_budget, self._max_lr))
        self._mode = mode

    def get(self, iteration: int, total_iterations: int) -> int:
        """Return the edit budget for the given iteration (clamped integer)."""
        if self._mode == "constant" or total_iterations <= 1:
            return self._max_lr

        t = max(0, min(iteration, total_iterations))
        n = total_iterations

        if self._mode == "linear":
            lr = self._max_lr + (self._min_lr - self._max_lr) * (t / n)
        elif self._mode == "cosine":
            lr = self._min_lr + 0.5 * (self._max_lr - self._min_lr) * (1.0 + math.cos(math.pi * t / n))
        else:
            lr = self._max_lr

        return max(self._min_lr, min(self._max_lr, round(lr)))
