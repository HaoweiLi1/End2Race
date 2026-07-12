#!/usr/bin/env python3
"""Multi-objective macro buffer accounting regression."""

import numpy as np

from bplus_v22.buffer import MacroRecord, MultiObjectiveMacroBuffer
from bplus_v22.macro import aggregate_micro_signals


def main() -> None:
    buffer = MultiObjectiveMacroBuffer(3)
    first = aggregate_micro_signals(
        np.ones(10), np.zeros(10), np.zeros(10), np.zeros(10, bool), np.zeros(10, bool)
    )
    terminal = aggregate_micro_signals(
        [0.0, -1.0], [0.0, 1.0], [0.0, 0.0], [False, True], [False, False]
    )
    truncation = aggregate_micro_signals(
        [0.5], [0.0], [1.0], [False], [True]
    )
    buffer.add(MacroRecord(first, -0.1, 0.2, 0.0, 0.1))
    buffer.add(MacroRecord(terminal, -0.2, 0.3, 0.2, 0.0))
    buffer.add(
        MacroRecord(
            truncation,
            -0.3,
            0.1,
            0.0,
            0.2,
            reward_trunc_next_value=0.7,
            collision_trunc_next_value=0.4,
            performance_trunc_next_value=0.8,
        )
    )
    output = buffer.compute_advantages(
        {"reward": 0.0, "collision": 0.0, "performance": 0.0}
    )
    for channel in ("reward", "collision", "performance"):
        assert output[f"{channel}_advantage"].shape == (3,)
        assert output[f"{channel}_return"].shape == (3,)
        assert np.all(np.isfinite(output[f"{channel}_return"]))
    assert output["macro_length"].tolist() == [10, 2, 1]
    assert output["log_prob"].tolist() == np.asarray([-0.1, -0.2, -0.3], np.float32).tolist()
    try:
        buffer.add(MacroRecord(first, 0.0, 0.0, 0.0, 0.0))
        raise AssertionError("macro buffer overflow accepted")
    except RuntimeError:
        pass
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
