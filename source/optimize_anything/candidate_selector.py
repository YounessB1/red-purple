"""Custom candidate selectors for the red-purple GEPA loop."""

import random

from gepa.core.state import GEPAState
from gepa.proposer.reflective_mutation.base import CandidateSelector


class ValAvgProportionalSelector(CandidateSelector):
    """Sample parent proportionally to val-average score.

    Each candidate's selection probability = score / sum(scores).
    Falls back to uniform when all scores are zero.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng if rng is not None else random.Random(0)

    def select_candidate_idx(self, state: GEPAState) -> int:
        scores = state.program_full_scores_val_set
        total = sum(scores)
        if total <= 0:
            return self.rng.randint(0, len(scores) - 1)
        cumulative, threshold = 0.0, self.rng.random() * total
        for idx, s in enumerate(scores):
            cumulative += s
            if cumulative >= threshold:
                return idx
        return len(scores) - 1
