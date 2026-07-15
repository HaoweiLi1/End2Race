# D2R-G Spatiotemporal Geometry — Implementation Plan

Version: `d2r-g-plan-1`  
Specification: `../specs/2026-07-11-d2r-spatiotemporal-geometry.md`

## 1. Red-green sequence

1. History-index tests: every tap is causal and clamps within its own episode.
2. Circular-convolution tests: beam-roll equivariance before pooling and
   fixed architecture/output shapes.
3. TTC-bin tests: boundary assignment, expected-center decoding, finite loss,
   and critical-region weighting.
4. Sampler tests: all event/critical frames retained, background stride 20,
   inverse-probability weight 20, and no held-out episode frame retained.
5. Registry/release tests: exactly 1,928 D2R-G non-test rows, no test/final
   pool identity, append idempotence, atomic partial promotion, and manifest
   corruption detection.
6. Remote source-hash check and tests in the pinned end2race interpreter.
7. One tiny fit/predict smoke with no registry mutation, followed by a fresh
   registered one-outer-fold engineering smoke only if all structural tests
   pass.
8. Full five-outer/three-inner OOF in a detached remote run. Open no test
   unless the complete OOF gate passes.
9. Independent validation, synthesis, handoff update, and explicit allowlist
   sync.

## 2. Planned files

- `d2r/__init__.py`: locked constants and train configuration;
- `d2r/data.py`: causal history view, labels, sampling, and registry rows;
- `d2r/model.py`: single circular spatiotemporal geometry encoder;
- `d2r/train.py`: nested grouped fitting, calibration, prediction, metrics;
- `d2r/release.py`: independent accounting and atomic release;
- `d2r_cli.py`: testable smoke/OOF/validate entry points;
- `tests/test_d2r_data.py`, `tests/test_d2r_model.py`, and
  `tests/test_d2r_release.py`.

All local edits use `apply_patch`. Runtime bytecode and Numba caches are
routed to `/tmp`. Remote synchronization is an explicit file allowlist.

## 3. Operational gates

- No registry append before all structural tests and source hashes pass.
- No full OOF before the registered engineering smoke independently validates.
- No test marker, test registry row, test NPZ, test feature, or test prediction
  unless complete OOF passes.
- No D3/PPO process in this plan.
