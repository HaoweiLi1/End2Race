#!/usr/bin/env python3
"""Small contracts for the B4 post-hoc analysis helpers."""

import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_b4_substantive_negative import (
    factorized_normal_log_prob,
    outcome_flags,
)


def main() -> None:
    # Transition flags are intentionally independent.  A collision-to-overtake
    # row is both a fixed collision and a gained overtake; an ``elif`` chain
    # would undercount the paired product diagnostics.
    flags = outcome_flags("collision", "overtaking")
    assert flags == {
        "fixed_collision": True,
        "new_collision": False,
        "persistent_collision": False,
        "gained_overtake": True,
        "lost_overtake": False,
    }
    flags = outcome_flags("overtaking", "collision")
    assert flags["new_collision"]
    assert flags["lost_overtake"]

    mean = np.asarray([[0.1, 5.0], [-0.2, 4.0]], dtype=np.float64)
    raw = np.asarray([[0.1, 5.0], [-0.17, 4.2]], dtype=np.float64)
    std = np.asarray([0.03, 0.20], dtype=np.float64)
    got = factorized_normal_log_prob(raw, mean, std)
    expected = []
    for action, location in zip(raw, mean):
        expected.append(
            sum(
                -0.5 * ((value - mu) / sigma) ** 2
                - math.log(sigma)
                - 0.5 * math.log(2.0 * math.pi)
                for value, mu, sigma in zip(action, location, std)
            )
        )
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)
    print("B4 substantive-negative analysis helper tests passed.")


if __name__ == "__main__":
    main()
