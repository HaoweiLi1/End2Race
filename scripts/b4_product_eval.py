#!/usr/bin/env python3
"""Resumable BC-compatible 3x4x50 evaluation for B4 plain End2Race actors.

Each shard owns ten of the fifty physical startpoints, so five shards form an
exact 600-case Cartesian grid.  ``run`` loads one strict plain End2Race actor
per worker and reuses the original ``eval_multiagent.evaluate_segment`` loop.
``merge`` rejects missing/duplicate rows and reports paired BC transitions.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import socket
import sys
from typing import Any, Mapping, Sequence


SCHEMA = "end2race-b4-product-eval-shard-1"
SUMMARY_SCHEMA = "end2race-b4-product-eval-merge-1"
MAP_NAME = "Austin"
EGO_RACELINE = "raceline1"
OPP_RACELINES = ("raceline0", "raceline1", "raceline2")
OPP_SPEED_SCALES = (0.5, 0.6, 0.7, 0.8)
STARTPOINT_COUNT = 50
SHARD_COUNT = 5
INTERVAL_IDX = 15
SIM_DURATION = 8.0
CASES_PER_SHARD = 120
TOTAL_CASES = 600
VALID_OUTCOMES = {"collision", "following", "overtaking"}

_WORKER: dict[str, Any] = {}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _startpoints(repo: Path) -> tuple[int, ...]:
    path = repo / f"f1tenth_racetracks/{MAP_NAME}/{EGO_RACELINE}.csv"
    # Literal evaluate.sh contract: ``tail -n +3 | wc -l``.
    max_waypoints = sum(1 for _ in path.open(encoding="utf-8")) - 2
    values = tuple(
        ordinal * max_waypoints // (STARTPOINT_COUNT - 1)
        for ordinal in range(STARTPOINT_COUNT)
    )
    if len(values) != STARTPOINT_COUNT or len(set(values)) != STARTPOINT_COUNT:
        raise ValueError("B4 product evaluation startpoint grid drift")
    return values


def enumerate_cases(repo: Path) -> tuple[dict[str, Any], ...]:
    cases: list[dict[str, Any]] = []
    for ordinal, ego_idx in enumerate(_startpoints(repo)):
        for opp_raceline in OPP_RACELINES:
            for speed in OPP_SPEED_SCALES:
                cases.append(
                    {
                        "case_id": (
                            f"sp{ordinal:02d}_{opp_raceline.replace('raceline', 'ol')}_"
                            f"s{speed:.1f}"
                        ),
                        "startpoint_ordinal": ordinal,
                        "ego_idx": ego_idx,
                        "opp_raceline": opp_raceline,
                        "opp_speedscale": speed,
                        "shard_index": ordinal % SHARD_COUNT,
                    }
                )
    if len(cases) != TOTAL_CASES or len({row["case_id"] for row in cases}) != TOTAL_CASES:
        raise AssertionError("B4 product evaluation Cartesian grid drift")
    return tuple(cases)


def _valid_metric(
    path: Path,
    case: Mapping[str, Any],
    *,
    variant: str,
    model_sha256: str,
) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    npz_relpath = value.get("npz_relpath")
    npz = (
        path.parent.parent / str(npz_relpath)
        if isinstance(npz_relpath, str)
        else Path("/__invalid_b4_npz__")
    )
    return (
        value.get("case_id") == case["case_id"]
        and value.get("variant") == variant
        and value.get("model_sha256") == model_sha256
        and value.get("outcome") in VALID_OUTCOMES
        and value.get("startpoint_ordinal") == case["startpoint_ordinal"]
        and value.get("ego_idx") == case["ego_idx"]
        and value.get("opp_raceline") == case["opp_raceline"]
        and float(value.get("opp_speedscale", -1.0)) == case["opp_speedscale"]
        and npz.is_file()
        and not npz.is_symlink()
        and value.get("npz_sha256") == _sha256(npz)
    )


def _init_worker(
    model_path: str,
    device_name: str,
    cache_root: str,
    variant: str,
    model_sha256: str,
) -> None:
    cache = Path(cache_root) / f"worker-{os.getpid()}"
    cache.mkdir(parents=True, exist_ok=False)
    os.environ["NUMBA_CACHE_DIR"] = str(cache.resolve())
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    import torch
    import torch.nn as nn
    from bplus_v22.b4_direct import load_strict_plain_actor

    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("B4 product evaluator requested unavailable CUDA")
        torch.cuda.set_device(device)

    class AccountingActor(nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor
            self.reset_runtime()

        @property
        def gru(self):
            return self.actor.gru

        def reset_runtime(self) -> None:
            self.steps = 0
            self.speed_projection_count = 0
            self.steer_projection_count = 0
            self.max_abs_speed_projection_delta = 0.0
            self.max_abs_steer_projection_delta = 0.0

        def forward(self, lidar, previous_speed, hidden):
            action, next_hidden = self.actor(lidar, previous_speed, hidden)
            requested = action[:, -1, :]
            steer = requested[:, 0]
            speed = requested[:, 1]
            steer_delta = torch.abs(torch.clamp(steer, -0.52, 0.52) - steer)
            speed_delta = torch.abs(torch.clamp(speed, 0.0, 20.0) - speed)
            self.steer_projection_count += int(torch.count_nonzero(steer_delta).item())
            self.speed_projection_count += int(torch.count_nonzero(speed_delta).item())
            self.max_abs_steer_projection_delta = max(
                self.max_abs_steer_projection_delta,
                float(torch.max(steer_delta).item()),
            )
            self.max_abs_speed_projection_delta = max(
                self.max_abs_speed_projection_delta,
                float(torch.max(speed_delta).item()),
            )
            self.steps += int(requested.shape[0])
            return action, next_hidden

        def accounting(self) -> dict[str, Any]:
            return {
                "deterministic_steps": self.steps,
                "deterministic_speed_projection_count": self.speed_projection_count,
                "deterministic_steer_projection_count": self.steer_projection_count,
                "max_abs_deterministic_speed_projection_delta": (
                    self.max_abs_speed_projection_delta
                ),
                "max_abs_deterministic_steer_projection_delta": (
                    self.max_abs_steer_projection_delta
                ),
            }

    actor = AccountingActor(load_strict_plain_actor(model_path, device)).to(device)
    actor.eval()
    _WORKER.update(
        {
            "model": actor,
            "model_path": model_path,
            "device": device,
            "variant": variant,
            "model_sha256": model_sha256,
        }
    )


def _run_case(payload: tuple[dict[str, Any], str]) -> dict[str, Any]:
    case, output_text = payload
    output = Path(output_text)
    metrics_path = output / "metrics" / f"{case['case_id']}.json"
    log_path = output / "logs" / f"{case['case_id']}.log"
    npz_path = output / "npz" / f"{case['case_id']}.npz"
    for directory in (metrics_path.parent, log_path.parent, npz_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    from contextlib import redirect_stderr, redirect_stdout
    from eval_multiagent import evaluate_segment

    model = _WORKER["model"]
    model.reset_runtime()
    with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
        result = evaluate_segment(
            model,
            _WORKER["device"],
            0.0,
            MAP_NAME,
            int(case["ego_idx"]),
            INTERVAL_IDX,
            EGO_RACELINE,
            str(case["opp_raceline"]),
            float(case["opp_speedscale"]),
            SIM_DURATION,
            False,
            str(_WORKER["model_path"]),
            result_tag=f"b4_{_WORKER['variant']}_shard{case['shard_index']}",
        )
    source_npz = Path(str(result["npz_path"]))
    if not source_npz.is_file() or npz_path.exists():
        raise RuntimeError(f"B4 product evaluator NPZ contract failed: {case['case_id']}")
    os.replace(source_npz, npz_path)
    result.update(
        {
            "schema": SCHEMA,
            "case_id": case["case_id"],
            "variant": _WORKER["variant"],
            "model_sha256": _WORKER["model_sha256"],
            "startpoint_ordinal": case["startpoint_ordinal"],
            "shard_index": case["shard_index"],
            "npz_path": str(npz_path.resolve()),
            "npz_relpath": f"npz/{case['case_id']}.npz",
            "npz_sha256": _sha256(npz_path),
            **model.accounting(),
        }
    )
    _write_json(metrics_path, result)
    return {
        "case_id": case["case_id"],
        "outcome": result["outcome"],
        "speed_projection_count": result["deterministic_speed_projection_count"],
    }


def run_shard(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    output = Path(args.output).resolve()
    model = Path(args.model_path).resolve()
    if not model.is_file() or model.is_symlink():
        raise ValueError("B4 product evaluation model is not one regular file")
    if not 0 <= args.shard_index < SHARD_COUNT:
        raise ValueError("B4 product evaluation shard index is invalid")
    if args.workers <= 0:
        raise ValueError("B4 product evaluation worker count must be positive")
    output.mkdir(parents=True, exist_ok=True)
    model_sha = _sha256(model)
    all_cases = enumerate_cases(repo)
    cases = tuple(row for row in all_cases if row["shard_index"] == args.shard_index)
    if len(cases) != CASES_PER_SHARD:
        raise AssertionError("B4 product evaluation shard size drift")
    manifest = {
        "schema": SCHEMA,
        "variant": args.variant,
        "model_path": str(model),
        "model_sha256": model_sha,
        "source_commit": args.source_commit,
        "producer_host_id": args.producer_host_id,
        "producer_hostname": socket.gethostname(),
        "producer_gpu_uuid": args.gpu_uuid,
        "device": args.device,
        "map_name": MAP_NAME,
        "ego_raceline": EGO_RACELINE,
        "opponent_racelines": list(OPP_RACELINES),
        "opponent_speed_scales": list(OPP_SPEED_SCALES),
        "startpoint_count": STARTPOINT_COUNT,
        "shard_count": SHARD_COUNT,
        "shard_index": args.shard_index,
        "case_count": len(cases),
        "assignment": "startpoint_ordinal_mod_5",
        "interval_idx": INTERVAL_IDX,
        "sim_duration": SIM_DURATION,
        "cases": list(cases),
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("B4 product evaluation resume manifest drift")
    else:
        _write_json(manifest_path, manifest)

    pending = [
        case
        for case in cases
        if not _valid_metric(
            output / "metrics" / f"{case['case_id']}.json",
            case,
            variant=args.variant,
            model_sha256=model_sha,
        )
    ]
    if (output / "COMPLETE").exists() and pending:
        raise ValueError("B4 product evaluation COMPLETE shard is incomplete")
    if pending:
        cache_root = output / "numba_cache"
        cache_root.mkdir(exist_ok=True)
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.workers, len(pending)),
            mp_context=context,
            initializer=_init_worker,
            initargs=(
                str(model),
                args.device,
                str(cache_root),
                args.variant,
                model_sha,
            ),
        ) as executor:
            futures = {
                executor.submit(_run_case, (dict(case), str(output))): case
                for case in pending
            }
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                future.result()
                completed += 1
                if completed % 10 == 0 or completed == len(futures):
                    print(
                        f"{args.variant} shard{args.shard_index}: "
                        f"{completed}/{len(futures)} newly completed",
                        flush=True,
                    )

    metrics = []
    for case in cases:
        path = output / "metrics" / f"{case['case_id']}.json"
        if not _valid_metric(path, case, variant=args.variant, model_sha256=model_sha):
            raise RuntimeError(f"B4 product evaluation incomplete: {case['case_id']}")
        metrics.append(json.loads(path.read_text(encoding="utf-8")))
    counts = {
        outcome: sum(row["outcome"] == outcome for row in metrics)
        for outcome in sorted(VALID_OUTCOMES)
    }
    summary = {
        "schema": SCHEMA,
        "passed": True,
        "variant": args.variant,
        "model_sha256": model_sha,
        "shard_index": args.shard_index,
        "case_count": len(metrics),
        "counts": counts,
        "deterministic_speed_projection_count": sum(
            int(row["deterministic_speed_projection_count"]) for row in metrics
        ),
        "deterministic_steer_projection_count": sum(
            int(row["deterministic_steer_projection_count"]) for row in metrics
        ),
    }
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary_path.unlink()
    _write_json(summary_path, summary)
    complete = output / "COMPLETE"
    if not complete.exists():
        complete.write_text(_sha256(summary_path) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


def _load_variant(root: Path, variant: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    model_shas: set[str] = set()
    for shard_index in range(SHARD_COUNT):
        shard = root / variant / f"shard{shard_index}"
        manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((shard / "summary.json").read_text(encoding="utf-8"))
        if (
            not (shard / "COMPLETE").is_file()
            or manifest.get("schema") != SCHEMA
            or manifest.get("variant") != variant
            or manifest.get("shard_index") != shard_index
            or manifest.get("case_count") != CASES_PER_SHARD
            or summary.get("passed") is not True
            or summary.get("case_count") != CASES_PER_SHARD
        ):
            raise ValueError(f"B4 product shard envelope failed: {variant}/shard{shard_index}")
        model_shas.add(str(manifest["model_sha256"]))
        for case in manifest["cases"]:
            metric_path = shard / "metrics" / f"{case['case_id']}.json"
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            if not _valid_metric(
                metric_path,
                case,
                variant=variant,
                model_sha256=str(manifest["model_sha256"]),
            ):
                raise ValueError(f"B4 product metric failed: {variant}/{case['case_id']}")
            if case["case_id"] in rows:
                raise ValueError("B4 product evaluation duplicate case")
            rows[case["case_id"]] = metric
    if len(model_shas) != 1 or len(rows) != TOTAL_CASES:
        raise ValueError(f"B4 product variant inventory failed: {variant}")
    return rows


def merge(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    variants = tuple(args.variant)
    if not variants or variants[0] != "BC" or len(set(variants)) != len(variants):
        raise ValueError("B4 product merge variants must be unique and start with BC")
    by_variant = {variant: _load_variant(root, variant) for variant in variants}
    keys = set(by_variant["BC"])
    if any(set(rows) != keys for rows in by_variant.values()):
        raise ValueError("B4 product paired case inventory drift")

    def counts(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
        return {
            "episodes": len(rows),
            "collision": sum(row["outcome"] == "collision" for row in rows.values()),
            "overtake": sum(row["outcome"] == "overtaking" for row in rows.values()),
            "follow": sum(row["outcome"] == "following" for row in rows.values()),
            "ego_collision": sum(bool(row["ego_collision"]) for row in rows.values()),
            "opp_collision": sum(bool(row["opp_collision"]) for row in rows.values()),
            "speed_projection": sum(
                int(row["deterministic_speed_projection_count"])
                for row in rows.values()
            ),
        }

    baseline = counts(by_variant["BC"])
    overtake_floor = math.ceil(0.95 * baseline["overtake"])
    candidates: dict[str, dict[str, Any]] = {}
    for variant in variants[1:]:
        row_counts = counts(by_variant[variant])
        fixed = new = gained = lost = 0
        for key in keys:
            before = by_variant["BC"][key]["outcome"]
            after = by_variant[variant][key]["outcome"]
            fixed += before == "collision" and after != "collision"
            new += before != "collision" and after == "collision"
            gained += before != "overtaking" and after == "overtaking"
            lost += before == "overtaking" and after != "overtaking"
        feasible = (
            row_counts["speed_projection"] == 0
            and row_counts["overtake"] >= overtake_floor
            and row_counts["collision"] < baseline["collision"]
            and fixed > new
        )
        candidates[variant] = {
            **row_counts,
            "fixed_collision": fixed,
            "new_collision": new,
            "gained_overtake": gained,
            "lost_overtake": lost,
            "overtake_floor": overtake_floor,
            "overtake_guardrail_pass": row_counts["overtake"] >= overtake_floor,
            "collision_strict_improve": row_counts["collision"] < baseline["collision"],
            "fixed_gt_new": fixed > new,
            "feasible": feasible,
            "collision_risk_ratio": (
                None
                if baseline["collision"] == 0
                else row_counts["collision"] / baseline["collision"]
            ),
        }
    feasible_variants = [name for name, value in candidates.items() if value["feasible"]]

    def iteration(name: str) -> int:
        try:
            return int(name.rsplit("iter", 1)[1])
        except (IndexError, ValueError):
            return 10**9

    selected = (
        min(
            feasible_variants,
            key=lambda name: (
                candidates[name]["collision"],
                -candidates[name]["overtake"],
                iteration(name),
            ),
        )
        if feasible_variants
        else None
    )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "integrity_passed": True,
        "grid": {
            "map_name": MAP_NAME,
            "opponent_racelines": list(OPP_RACELINES),
            "opponent_speed_scales": list(OPP_SPEED_SCALES),
            "startpoint_count": STARTPOINT_COUNT,
            "episode_count_per_variant": TOTAL_CASES,
        },
        "outcome_contract": "original eval_multiagent terminal state_label",
        "bc": baseline,
        "overtake_floor_95pct": overtake_floor,
        "candidates": candidates,
        "selected_variant": selected,
        "verdict": "SURVIVOR" if selected is not None else "B4_SUBSTANTIVE_NEGATIVE",
    }
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "summary.json", summary)
    fields = (
        "case_id",
        "startpoint_ordinal",
        "ego_idx",
        "opp_raceline",
        "opp_speedscale",
        "variant",
        "outcome",
        "ego_collision",
        "opp_collision",
        "deterministic_speed_projection_count",
        "model_sha256",
    )
    with (output / "paired_rows.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key in sorted(keys):
            for variant in variants:
                writer.writerow({name: by_variant[variant][key].get(name, "") for name in fields})
    report = [
        "# B4 product-grid evaluation",
        "",
        f"- Grid: {len(OPP_RACELINES)} racelines x {len(OPP_SPEED_SCALES)} speeds x {STARTPOINT_COUNT} startpoints = {TOTAL_CASES}",
        f"- BC: collision={baseline['collision']}, overtake={baseline['overtake']}, follow={baseline['follow']}",
        f"- 95% overtake floor: {overtake_floor}",
        "",
        "| variant | collision | overtake | follow | fixed | new | gained | lost | speed projection | feasible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant, value in candidates.items():
        report.append(
            f"| {variant} | {value['collision']} | {value['overtake']} | {value['follow']} | "
            f"{value['fixed_collision']} | {value['new_collision']} | "
            f"{value['gained_overtake']} | {value['lost_overtake']} | "
            f"{value['speed_projection']} | {value['feasible']} |"
        )
    report.extend(["", f"Selected: `{selected}`", f"Verdict: **{summary['verdict']}**", ""])
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--repo", default=".")
    run.add_argument("--model-path", required=True)
    run.add_argument("--variant", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--workers", type=int, default=2)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--source-commit", required=True)
    run.add_argument("--producer-host-id", required=True)
    run.add_argument("--gpu-uuid", required=True)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--root", required=True)
    merge_parser.add_argument("--variant", action="append", required=True)
    merge_parser.add_argument("--output", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.action == "run":
        return run_shard(args)
    return merge(args)


if __name__ == "__main__":
    sys.exit(main())
