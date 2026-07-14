import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_b5_objective_alignment.py"
SPEC = importlib.util.spec_from_file_location("analyze_b5_objective_alignment", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_opened_prevalence_weights_preserve_mean_episode_mass():
    value = sum(
        MODULE.TRAIN_COUNTS[outcome] * MODULE.PREVALENCE[outcome]
        for outcome in MODULE.OUTCOMES
    )
    assert abs(value - 16.0) < 1e-12
    assert MODULE.PREVALENCE == {
        "collision": 0.10666666666666667,
        "overtake": 1.5199999999999998,
        "follow": 1.56,
    }


def test_function_metric_uses_declared_action_scales():
    direction = torch.tensor([[0.03, 0.0], [0.0, 0.20]]).numpy()
    reference = direction.copy()
    weight = torch.tensor([0.5, 0.5], dtype=torch.float64)
    result = MODULE.weighted_direction_metrics(direction, reference, weight)
    assert abs(result["direction_norm"] - 1.0) < 1e-7
    assert abs(result["cosine"] - 1.0) < 1e-12
