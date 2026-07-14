import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_b5_statistical_noise.py"
SPEC = importlib.util.spec_from_file_location("analyze_b5_statistical_noise", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_exact_mcnemar_matches_b5_iter10_counts():
    assert abs(MODULE.exact_mcnemar_two_sided(9, 7) - 0.803619384765625) < 1e-15
    assert abs(MODULE.exact_mcnemar_two_sided(12, 7) - 0.359283447265625) < 1e-15


def test_joint_sign_flip_preserves_snapshot_dependence():
    vectors = [(1, 1, 0), (1, -1, 1)]
    distribution = MODULE.sign_flip_distribution(vectors)
    assert sum(distribution.values()) == 4
    assert distribution[(2, 0, 1)] == 1
    assert distribution[(-2, 0, -1)] == 1
    observed = (2, 0, 1)
    max_p = sum(
        count for state, count in distribution.items() if max(state) >= max(observed)
    ) / sum(distribution.values())
    assert max_p == 0.5
