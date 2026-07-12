#!/usr/bin/env python3
"""Unit tests for the single locked D2R geometry network."""

import numpy as np
import torch

from d2r import TTC_BIN_COUNT
from d2r.model import D2RGeometryNet, decode_ttc_logits, initialize_classification_bias


def check(name, condition):
    if not condition:
        raise AssertionError(f"FAIL {name}")


def main():
    torch.manual_seed(20260711)
    model = D2RGeometryNet().eval()
    lidar = torch.rand(3, 8, 360)
    bc = torch.randn(3, 1680)
    scalar = torch.randn(3, 24)
    with torch.no_grad():
        output = model(lidar, bc, scalar)
    check("collision-shape", output["collision_logits"].shape == (3, 6))
    check("ttc-shape", output["ttc_logits"].shape == (3, TTC_BIN_COUNT))
    check("geometry-shape", all(output[name].shape == (3,) for name in ("rel_s", "lateral_gap", "closing_rate")))
    check("rel-range", torch.all(torch.abs(output["rel_s"]) <= 10.0))
    check("lateral-range", torch.all((output["lateral_gap"] >= 0.0) & (output["lateral_gap"] <= 2.0)))
    check("closing-range", torch.all(torch.abs(output["closing_rate"]) <= 5.0))
    predicted_ttc = decode_ttc_logits(output["ttc_logits"])
    check("ttc-range", torch.all((predicted_ttc >= 0.05) & (predicted_ttc <= 4.95)))

    # Circular convolutions must commute with a beam roll before pooling.
    with torch.no_grad():
        encoded = model.encode_beams(lidar, pool=False)
        shifted = model.encode_beams(torch.roll(lidar, 17, dims=2), pool=False)
    check("circular-equivariance", torch.allclose(shifted, torch.roll(encoded, 17, dims=2), atol=2e-6, rtol=1e-5))

    logits = torch.full((2, TTC_BIN_COUNT), -100.0)
    logits[0, 0] = 100.0
    logits[1, -1] = 100.0
    decoded = decode_ttc_logits(logits).numpy()
    check("ttc-decode-centers", np.allclose(decoded, [0.05, 4.95], atol=1e-6))

    prevalence = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06], dtype=np.float32)
    initialize_classification_bias(model, prevalence)
    observed = torch.sigmoid(model.collision_head.bias).detach().numpy()
    check("bias-prevalence", np.allclose(observed, prevalence, atol=1e-7))
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
