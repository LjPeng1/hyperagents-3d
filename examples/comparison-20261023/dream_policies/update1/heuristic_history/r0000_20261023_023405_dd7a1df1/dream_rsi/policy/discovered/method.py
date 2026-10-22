# DREAM-RSI incumbent: parallel-refine baseline
"""Incumbent policy before any promotion: the parallel-refine baseline.

Replays the hand-designed wave-parallel explore + refine policy with the
online grid caps, so the first offline phase compares the proposals against
the policy that generated the trace.
"""

from dream_rsi.policy.baseline import ParallelRefineMethod

GRID_CAPS = {'max_width': 2, 'max_refine': 1}


class OptimalPolicy(ParallelRefineMethod):
    NAME = "parallel_refine"

    def __init__(self, config=None):
        super().__init__({**GRID_CAPS, **(config or {})})
