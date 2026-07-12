#!/usr/bin/env python3
"""Registry planning and portable sidecar-bundle invariants."""

from dataclasses import asdict
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from bplus_v22.sidecar import (
    EXPECTED_REGISTRY_BEFORE_SHA256,
    REGISTRY_DECISION_EFFECT,
    REGISTRY_RELPATH,
    REGISTRY_STAGE,
    REGISTRY_USE_CLASS,
    _bundle_payload,
    _episode_rows,
    _registry_plan_live_state,
    _validate_bundle,
    file_sha256,
    load_sidecar_bundle,
    make_actor_pretrain_registry_rows,
)
from d0.identity import append_opened_registry
from d2r import LOCKED_CONFIG, SEED
from d2r.model import D2RGeometryNet


LOSS_NAMES = (
    "loss",
    "classification_loss",
    "ttc_loss",
    "rel_loss",
    "lateral_loss",
    "closing_loss",
)


def train_report() -> dict:
    history = []
    for epoch in range(6):
        row = {"epoch": epoch, "batches": 3}
        row.update({name: float(epoch + 1) / 10.0 for name in LOSS_NAMES})
        history.append(row)
    return {
        "schema": "d2r-train-report-1",
        "config": asdict(LOCKED_CONFIG),
        "seed": SEED,
        "sampled_frame_count": 123,
        "sampling_weight_sum": 456.0,
        "class_counts": {},
        "initial_prevalence": [0.01] * 6,
        "micro_max_batches_per_epoch": None,
        "history": history,
    }


def main() -> None:
    root = Path(".").resolve()
    episodes = _episode_rows(root)
    rows = make_actor_pretrain_registry_rows(episodes)
    assert len(rows) == 1928
    assert {row["stage"] for row in rows} == {REGISTRY_STAGE}
    assert {row["use_class"] for row in rows} == {REGISTRY_USE_CLASS}
    assert {row["decision_effect"] for row in rows} == {REGISTRY_DECISION_EFFECT}
    assert {row["final_pool"] for row in rows} == {"false"}

    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        before = temporary / "before.tsv"
        after = temporary / "after.tsv"
        # Use the immutable pre-append snapshot, not mutable live registry
        # state, so this regression remains valid after later stages append.
        shutil.copyfile(
            root
            / "Experiments/B1_route_r2_scaffold/artifacts/"
            "registry_plan_20260712_075931/registry_before.snapshot.tsv",
            before,
        )
        shutil.copyfile(before, after)
        assert file_sha256(before) == EXPECTED_REGISTRY_BEFORE_SHA256
        assert (
            _registry_plan_live_state(
                before,
                rows,
                EXPECTED_REGISTRY_BEFORE_SHA256,
                "0" * 64,
            )
            == "ready"
        )
        appended = append_opened_registry(after, rows)
        assert (appended.appended, appended.skipped, appended.total) == (
            1928,
            0,
            12019,
        )
        after_sha = file_sha256(after)
        assert (
            _registry_plan_live_state(
                after,
                rows,
                EXPECTED_REGISTRY_BEFORE_SHA256,
                after_sha,
            )
            == "already_appended"
        )
        repeated = append_opened_registry(after, rows)
        assert (repeated.appended, repeated.skipped, repeated.total) == (
            0,
            1928,
            12019,
        )
        assert file_sha256(after) == after_sha

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            model = D2RGeometryNet().eval()
        mean = np.zeros(1680, dtype=np.float32)
        std = np.ones(1680, dtype=np.float32)
        bundle = _bundle_payload(model, mean, std, train_report())
        release = temporary / "bundle_release"
        release.mkdir()
        torch.save(bundle, release / "sidecar_bundle.pt")
        details = _validate_bundle(release)
        state, observed_mean, observed_std, loaded = load_sidecar_bundle(release)
        assert details["epochs"] == 6 and details["sampled_frames"] == 123
        assert details["state_dict_sha256"] == loaded["state_dict_sha256"]
        assert set(state) == set(model.state_dict())
        assert torch.equal(observed_mean, torch.zeros(1680))
        assert torch.equal(observed_std, torch.ones(1680))

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
