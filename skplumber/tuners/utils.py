from typing import NamedTuple


class TuneResult(NamedTuple):
    best_score: float
    """The best validation score the tuner was able to find."""
    n_evals: int
    """The number of evaluations the tuner completed while optimizing."""
    did_improve: bool
    """Whether the tuner improved upon the starting hyperparameter configuration"""
