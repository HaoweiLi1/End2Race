#!/usr/bin/env python3
"""Unit test for locked D2R TTC interval accounting."""

import numpy as np

from d2r.summary import _ttc_bins


def main():
    target = np.array([0.0, 0.249, 0.25, 0.499, 0.5, 0.999, 1.0, 1.499, 1.5, 1.999, 2.0])
    predicted = target + 0.2
    report = _ttc_bins(target, predicted)
    counts = [report[token]["count"] for token in ("000_025", "025_050", "050_100", "100_150", "150_200")]
    if counts != [2, 2, 2, 2, 2]:
        raise AssertionError(f"FAIL TTC bin counts: {counts}")
    if not all(np.isclose(report[token]["mae"], 0.2) for token in report):
        raise AssertionError("FAIL TTC bin MAE")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
