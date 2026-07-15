"""Prospective Task-8 training/development manifest release."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

from bplus_v22 import ARMS, OWNER_DECISION, PILOT_SEEDS
from bplus_v22.release import file_sha256, validate_source_preflight
from d0.identity import REGISTRY_FIELDS, validate_registry_row


METADATA_RELPATH = (
    "Experiments/A3_d2_representation/artifacts/"
    "non_test_full_20260711_175713/episode_metadata.tsv"
)
METADATA_SHA256 = "468d8be50aecad19f89fbf2c35dc421acb4244a61f957f77dcfff1acd227eda3"
SPLIT_RELPATH = (
    "Experiments/A3_d2_representation/artifacts/split_lock/scenario_split.tsv"
)
SPLIT_SHA256 = "2f8146d7be0e36c3abcc084dcdbfa9e3df85983c37c6249294ab19b1431c49f3"
WARMSTART_MANIFEST_RELPATH = (
    "Experiments/B1_route_r2_scaffold/artifacts/"
    "warmstart_remediation_manifest_20260712_100032"
)
WARMSTART_MANIFEST_SHA256 = (
    "72b3ef0e25a41984e256454218e36640bd9e045430671b57af570e7d1896f24e"
)
D25_RESULTS_RELPATH = (
    "Experiments/A4_d25_counterfactual/artifacts/"
    "full_oracle_20260711_185500/case_results.tsv"
)
D25_RESULTS_SHA256 = "0ef0a09adba1d46d76151187a4d295ce149ad8409a458befc7950d7d3f7b7c1b"
REGISTRY_RELPATH = "Experiments/A0_project_registry/opened_registry.tsv"
REGISTRY_SHA256 = "aff5f03db06836c6c51ff53944ed2ec2e521fbe777cc7d26228a15a9362d0b0d"
REGISTRY_BASE_SNAPSHOT_RELPATH = (
    "Experiments/B1_route_r2_scaffold/artifacts/"
    "warmstart_remediation_manifest_20260712_100032/registry_after.expected.tsv"
)

PANEL_REPRESENTATIVE = "representative_preservation"
PANEL_SKILL_F = "skill_F"
PANEL_SKILL_S = "skill_S"
DEVELOPMENT_PANELS = (PANEL_REPRESENTATIVE, PANEL_SKILL_F, PANEL_SKILL_S)
SNAPSHOT_IDS = ("WARMSTART", "PPO_ITER_0020", "PPO_ITER_0040")
DOMAIN = b"end2race:bplus-v2.2:task8:v1\0"

SCENARIO_FIELDS = (
    "manifest_order",
    "panel",
    "estimate_class",
    "mechanism_enriched",
    "held_out_policy_generalization",
    "l2_id",
    "l3_id",
    "l4_id",
    "map_name",
    "skill",
    "opponent_raceline",
    "speedscale_hex",
    "resolved_ego_idx",
    "npz_relpath",
    "npz_sha256",
    "frame_count",
    "selection_sha256",
)
TRAIN_FIELDS = (
    "training_order",
    "l2_id",
    "l3_id",
    "l4_id",
    "map_name",
    "skill",
    "opponent_raceline",
    "speedscale_hex",
    "resolved_ego_idx",
    "npz_relpath",
    "npz_sha256",
    "frame_count",
    "selection_sha256",
)
WITNESS_FIELDS = (
    "panel_order",
    "panel",
    "mechanism_enriched",
    "held_out_policy_generalization",
    "l2_id",
    "l4_id",
    "map_name",
    "skill",
    "witness_branch_id",
    "source_npz_relpath",
    "source_npz_sha256",
)
RECOVERABILITY_FIELDS = (
    "panel_order",
    "panel",
    "mechanism_enriched",
    "held_out_policy_generalization",
    "l2_id",
    "l4_id",
    "map_name",
    "skill",
    "status",
)
SMOKE_FIELDS = (
    "smoke_order",
    "selection_stratum",
    "l2_id",
    "l3_id",
    "l4_id",
    "map_name",
    "skill",
    "opponent_raceline",
    "speedscale_hex",
    "resolved_ego_idx",
    "npz_relpath",
    "npz_sha256",
    "frame_count",
)
JOB_FIELDS = (
    "job_order",
    "arm",
    "snapshot_id",
    "pilot_seed",
    "scenario_count",
    "scenario_manifest_sha256",
    "ordered_l2_sha256",
)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _validate_registry_is_exact_or_strict_append(root: Path) -> str:
    """Preserve Task-8's frozen registry while allowing prospective reuse rows."""

    live = root / REGISTRY_RELPATH
    observed_sha = file_sha256(live)
    if observed_sha == REGISTRY_SHA256:
        return observed_sha
    baseline = root / REGISTRY_BASE_SNAPSHOT_RELPATH
    if file_sha256(baseline) != REGISTRY_SHA256:
        raise ValueError("Task-8 registry baseline snapshot drift")
    with baseline.open(newline="", encoding="utf-8") as handle:
        base_reader = csv.DictReader(handle, delimiter="\t")
        if tuple(base_reader.fieldnames or ()) != REGISTRY_FIELDS:
            raise ValueError("Task-8 registry baseline header drift")
        base_rows = [validate_registry_row(row) for row in base_reader]
    with live.open(newline="", encoding="utf-8") as handle:
        live_reader = csv.DictReader(handle, delimiter="\t")
        if tuple(live_reader.fieldnames or ()) != REGISTRY_FIELDS:
            raise ValueError("Task-8 live registry header drift")
        live_rows = [validate_registry_row(row) for row in live_reader]
    base_by_id = {row["row_id"]: row for row in base_rows}
    live_by_id = {row["row_id"]: row for row in live_rows}
    if (
        len(base_by_id) != len(base_rows)
        or len(live_by_id) != len(live_rows)
        or len(live_rows) <= len(base_rows)
        or any(live_by_id.get(row_id) != row for row_id, row in base_by_id.items())
    ):
        raise ValueError("Task-8 live registry is not a strict append-only superset")
    return observed_sha


def _write_tsv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _score(domain: str, l2_id: str) -> str:
    return hashlib.sha256(DOMAIN + domain.encode("utf-8") + b"\0" + l2_id.encode()).hexdigest()


def _take_diverse(rows: list[dict[str, str]], count: int, domain: str) -> list[dict[str, str]]:
    ranked = sorted(rows, key=lambda row: (_score(domain, row["l2_id"]), row["l2_id"]))
    first: list[dict[str, str]] = []
    later: list[dict[str, str]] = []
    seen_l4: set[str] = set()
    for row in ranked:
        if row["l4_id"] in seen_l4:
            later.append(row)
        else:
            first.append(row)
            seen_l4.add(row["l4_id"])
    selected = (first + later)[:count]
    if len(selected) != count:
        raise ValueError(f"Task-8 stratum lacks {count} rows: {domain}")
    return selected


def _scenario_row(row: dict[str, str], order: int, panel: str) -> dict[str, str]:
    distributional = panel == PANEL_REPRESENTATIVE
    return {
        "manifest_order": str(order),
        "panel": panel,
        "estimate_class": "distributional_development" if distributional else "mechanism_enriched",
        "mechanism_enriched": str(not distributional).lower(),
        "held_out_policy_generalization": "false",
        "l2_id": row["l2_id"],
        "l3_id": row["l3_id"],
        "l4_id": row["l4_id"],
        "map_name": row["map_name"],
        "skill": row["skill"],
        "opponent_raceline": row["opponent_raceline"],
        "speedscale_hex": row["speedscale_hex"],
        "resolved_ego_idx": row["resolved_ego_idx"],
        "npz_relpath": row["npz_relpath"],
        "npz_sha256": row["npz_sha256"],
        "frame_count": row["frame_count"],
        "selection_sha256": _score(f"development:{panel}", row["l2_id"]),
    }


def build_manifests(repo_root: str | Path = ".") -> dict[str, list[dict[str, str]]]:
    root = Path(repo_root).resolve()
    sources = {
        METADATA_RELPATH: METADATA_SHA256,
        SPLIT_RELPATH: SPLIT_SHA256,
        f"{WARMSTART_MANIFEST_RELPATH}/output_manifest.sha256": WARMSTART_MANIFEST_SHA256,
        D25_RESULTS_RELPATH: D25_RESULTS_SHA256,
    }
    for relative, expected in sources.items():
        if file_sha256(root / relative) != expected:
            raise ValueError(f"Task-8 frozen source hash drift: {relative}")
    _validate_registry_is_exact_or_strict_append(root)
    metadata = _read_tsv(root / METADATA_RELPATH)
    split = _read_tsv(root / SPLIT_RELPATH)
    non_test = {row["l2_id"] for row in split if row["split"] == "non_test"}
    sealed = {row["l2_id"] for row in split if row["split"] == "test"}
    if len(metadata) != 1928 or {row["l2_id"] for row in metadata} != non_test:
        raise ValueError("Task-8 metadata/non-test split mismatch")
    if set(row["l2_id"] for row in metadata) & sealed:
        raise ValueError("Task-8 metadata contains sealed-test scenario")

    used: set[str] = set()
    panels: dict[str, list[dict[str, str]]] = {}
    for panel, skill in ((PANEL_SKILL_F, "skill_F"), (PANEL_SKILL_S, "skill_S")):
        selected: list[dict[str, str]] = []
        for map_name in ("Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring"):
            raceline_quotas = {"raceline1": 24} if skill == "skill_F" else {
                "raceline0": 12,
                "raceline2": 12,
            }
            for raceline, quota in raceline_quotas.items():
                candidates = [
                    row for row in metadata
                    if row["skill"] == skill
                    and row["map_name"] == map_name
                    and row["opponent_raceline"] == raceline
                    and row["l2_id"] not in used
                ]
                chosen = _take_diverse(candidates, quota, f"{panel}:{map_name}:{raceline}")
                selected.extend(chosen)
                used.update(row["l2_id"] for row in chosen)
        panels[panel] = selected

    representative: list[dict[str, str]] = []
    quotas = {"other": 12, "skill_F": 4, "skill_S": 8}
    for map_name in ("Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring"):
        for skill, count in quotas.items():
            candidates = [
                row for row in metadata
                if row["map_name"] == map_name
                and row["skill"] == skill
                and row["l2_id"] not in used
            ]
            chosen = _take_diverse(candidates, count, f"{PANEL_REPRESENTATIVE}:{map_name}:{skill}")
            representative.extend(chosen)
            used.update(row["l2_id"] for row in chosen)
    panels[PANEL_REPRESENTATIVE] = representative

    development: list[dict[str, str]] = []
    for panel in DEVELOPMENT_PANELS:
        ordered = sorted(
            panels[panel], key=lambda row: (_score(f"development:{panel}", row["l2_id"]), row["l2_id"])
        )
        base_order = len(development)
        development.extend(
            _scenario_row(row, base_order + index, panel)
            for index, row in enumerate(ordered)
        )
    if len(development) != 288 or len({row["l2_id"] for row in development}) != 288:
        raise AssertionError("Task-8 development panels are not 288 unique scenarios")
    if [int(row["manifest_order"]) for row in development] != list(range(288)):
        raise AssertionError("Task-8 development manifest order is not contiguous")

    training_source = sorted(
        (row for row in metadata if row["l2_id"] not in used),
        key=lambda row: (_score("training", row["l2_id"]), row["l2_id"]),
    )
    training = [
        {
            "training_order": str(index),
            **{name: row[name] for name in TRAIN_FIELDS[1:-1]},
            "selection_sha256": _score("training", row["l2_id"]),
        }
        for index, row in enumerate(training_source)
    ]
    if len(training) != 1640:
        raise AssertionError("Task-8 training/development partition drift")

    warm_rows = _read_tsv(root / WARMSTART_MANIFEST_RELPATH / "episodes.tsv")
    witness_source = [row for row in warm_rows if row["role"] == "witness"]
    witness = [
        {
            "panel_order": str(index),
            "panel": "d25_witness_training",
            "mechanism_enriched": "true",
            "held_out_policy_generalization": "false",
            **{name: row[name] for name in WITNESS_FIELDS[4:]},
        }
        for index, row in enumerate(sorted(witness_source, key=lambda row: row["l2_id"]))
    ]
    if len(witness) != 67:
        raise AssertionError("Task-8 witness panel must contain exactly 67 training witnesses")

    witness_l4 = {row["l4_id"] for row in witness}
    recoverable = [
        row for row in _read_tsv(root / D25_RESULTS_RELPATH)
        if row["status"] == "recovered_confirmed_safe_pass" and row["l4_id"] not in witness_l4
    ]
    recoverability = [
        {
            "panel_order": str(index),
            "panel": "l4_disjoint_recoverability",
            "mechanism_enriched": "true",
            "held_out_policy_generalization": "false",
            **{name: row[name] for name in RECOVERABILITY_FIELDS[4:]},
        }
        for index, row in enumerate(sorted(recoverable, key=lambda row: row["l2_id"]))
    ]

    smoke_source: list[dict[str, str]] = []
    for map_name in ("Austin", "Hockenheim", "MoscowRaceway", "Nuerburgring"):
        for panel in (PANEL_SKILL_F, PANEL_SKILL_S):
            candidates = [
                row for row in development if row["map_name"] == map_name and row["panel"] == panel
            ]
            smoke_source.append(min(candidates, key=lambda row: _score(f"smoke:{map_name}:{panel}", row["l2_id"])))
    smoke = [
        {
            "smoke_order": str(index),
            "selection_stratum": f"{row['map_name']}:{row['panel']}",
            **{name: row[name] for name in SMOKE_FIELDS[2:]},
        }
        for index, row in enumerate(smoke_source)
    ]
    return {
        "development": development,
        "training": training,
        "witness": witness,
        "recoverability": recoverability,
        "smoke": smoke,
    }


def _ordered_l2_sha(rows: list[dict[str, str]]) -> str:
    return hashlib.sha256("\n".join(row["l2_id"] for row in rows).encode("utf-8") + b"\n").hexdigest()


def _jobs(development: list[dict[str, str]], manifest_sha: str) -> list[dict[str, str]]:
    rows = []
    for arm in ARMS:
        for snapshot in SNAPSHOT_IDS:
            for seed in PILOT_SEEDS:
                rows.append({
                    "job_order": str(len(rows)),
                    "arm": arm,
                    "snapshot_id": snapshot,
                    "pilot_seed": str(seed),
                    "scenario_count": str(len(development)),
                    "scenario_manifest_sha256": manifest_sha,
                    "ordered_l2_sha256": _ordered_l2_sha(development),
                })
    return rows


def _write_output_manifest(directory: Path) -> None:
    paths = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.name not in {"output_manifest.sha256", "COMPLETE"}
    )
    (directory / "output_manifest.sha256").write_text(
        "\n".join(f"{file_sha256(path)}  {path.relative_to(directory).as_posix()}" for path in paths) + "\n",
        encoding="utf-8",
    )


def create_manifest_release(
    repo_root: str | Path,
    source_preflight_dir: str | Path,
    output_dir: str | Path,
    created_at: str,
) -> dict:
    root = Path(repo_root).resolve()
    if not validate_source_preflight(source_preflight_dir, root)["passed"]:
        raise ValueError("Task-8 source preflight is invalid")
    built = build_manifests(root)
    output = Path(output_dir)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("Task-8 output/partial exists")
    partial.mkdir(parents=True)
    try:
        _write_tsv(partial / "development_scenarios.tsv", built["development"], SCENARIO_FIELDS)
        _write_tsv(partial / "training_scenarios.tsv", built["training"], TRAIN_FIELDS)
        _write_tsv(partial / "d25_witness_training.tsv", built["witness"], WITNESS_FIELDS)
        _write_tsv(partial / "l4_disjoint_recoverability.tsv", built["recoverability"], RECOVERABILITY_FIELDS)
        _write_tsv(partial / "no_learning_smoke.tsv", built["smoke"], SMOKE_FIELDS)
        dev_sha = file_sha256(partial / "development_scenarios.tsv")
        jobs = _jobs(built["development"], dev_sha)
        _write_tsv(partial / "evaluation_jobs.tsv", jobs, JOB_FIELDS)
        config = {
            "schema": "bplus-v2.2-task8-manifest-release-1",
            "created_at": str(created_at),
            "owner_decision": OWNER_DECISION,
            "policy_training_started": False,
            "closed_loop_evaluation_started": False,
            "test_opened": False,
            "final_pool": False,
            "arm_selection_performed": False,
            "source_preflight_relpath": str(Path(source_preflight_dir)),
            "source_preflight_output_manifest_sha256": file_sha256(Path(source_preflight_dir) / "output_manifest.sha256"),
            "registry_sha256": REGISTRY_SHA256,
            "counts": {name: len(rows) for name, rows in built.items()} | {"evaluation_jobs": len(jobs)},
            "panel_semantics": {
                PANEL_REPRESENTATIVE: "distributional_development_estimate",
                PANEL_SKILL_F: "mechanism_enriched_not_representative",
                PANEL_SKILL_S: "mechanism_enriched_not_representative",
                "d25_witness_training": "mechanism_enriched_non_held_out_training_cases",
                "l4_disjoint_recoverability": "unavailable_no_recovered_case_outside_witness_l4" if not built["recoverability"] else "mechanism_enriched_l4_disjoint_diagnostic",
            },
            "selection_rule": {
                "domain_hex": DOMAIN.hex(),
                "development_size": 288,
                "per_panel": 96,
                "skill_panel_quota": "24 skill_F/raceline1 per map; 12 skill_S each raceline0/raceline2 per map; L4-first hash ordering",
                "representative_quota": "per map: other=12, skill_F=4, skill_S=8 with L4-first hash ordering",
                "panels_are_l2_disjoint": True,
                "training_is_remaining_non_test_l2": True,
                "outcome_used_only_for": ["d25_witness_training", "l4_disjoint_recoverability"],
            },
            "snapshots": list(SNAPSHOT_IDS),
            "pilot_seeds": list(PILOT_SEEDS),
            "development_manifest_sha256": dev_sha,
            "ordered_l2_sha256": _ordered_l2_sha(built["development"]),
            "l4_disjoint_recoverability_available": bool(built["recoverability"]),
        }
        (partial / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (partial / "validation.json").write_text(json.dumps({
            "schema": "bplus-v2.2-task8-manifest-validation-1",
            "passed": True,
            "violations": [],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_output_manifest(partial)
        (partial / "COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    validation = validate_manifest_release(output, root)
    if not validation["passed"]:
        raise AssertionError(f"created invalid Task-8 manifest: {validation}")
    return validation | {"output_manifest_sha256": file_sha256(output / "output_manifest.sha256")}


def validate_manifest_release(release_dir: str | Path, repo_root: str | Path = ".") -> dict:
    release = Path(release_dir)
    violations: list[str] = []
    counts: dict[str, int] = {}
    try:
        if not (release / "COMPLETE").is_file():
            raise ValueError("Task-8 manifest lacks COMPLETE")
        entries = {}
        for line in (release / "output_manifest.sha256").read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            entries[relative] = digest
        observed = {
            path.relative_to(release).as_posix() for path in release.rglob("*")
            if path.is_file() and path.name not in {"output_manifest.sha256", "COMPLETE"}
        }
        if set(entries) != observed or any(file_sha256(release / name) != digest for name, digest in entries.items()):
            raise ValueError("Task-8 output manifest mismatch")
        config = json.loads((release / "config.json").read_text(encoding="utf-8"))
        if (
            config["schema"] != "bplus-v2.2-task8-manifest-release-1"
            or config["owner_decision"] != OWNER_DECISION
            or any(config[name] is not False for name in (
                "policy_training_started", "closed_loop_evaluation_started", "test_opened", "final_pool", "arm_selection_performed"
            ))
            or config["registry_sha256"] != REGISTRY_SHA256
        ):
            raise ValueError("Task-8 authority/scope mismatch")
        built = build_manifests(repo_root)
        specs = {
            "development": ("development_scenarios.tsv", SCENARIO_FIELDS),
            "training": ("training_scenarios.tsv", TRAIN_FIELDS),
            "witness": ("d25_witness_training.tsv", WITNESS_FIELDS),
            "recoverability": ("l4_disjoint_recoverability.tsv", RECOVERABILITY_FIELDS),
            "smoke": ("no_learning_smoke.tsv", SMOKE_FIELDS),
        }
        for name, (relative, fields) in specs.items():
            with (release / relative).open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if tuple(reader.fieldnames or ()) != fields:
                    raise ValueError(f"Task-8 {name} header drift")
                rows = list(reader)
            if rows != built[name]:
                raise ValueError(f"Task-8 {name} recomputation mismatch")
            counts[name] = len(rows)
        jobs = _read_tsv(release / "evaluation_jobs.tsv")
        expected_jobs = _jobs(built["development"], file_sha256(release / "development_scenarios.tsv"))
        if jobs != expected_jobs or tuple(jobs[0]) != JOB_FIELDS:
            raise ValueError("Task-8 evaluation Cartesian product mismatch")
        counts["evaluation_jobs"] = len(jobs)
        if config["counts"] != counts:
            raise ValueError("Task-8 config counts mismatch")
        if config["development_manifest_sha256"] != file_sha256(release / "development_scenarios.tsv"):
            raise ValueError("Task-8 development hash mismatch")
        if config["ordered_l2_sha256"] != _ordered_l2_sha(built["development"]):
            raise ValueError("Task-8 ordered scenario hash mismatch")
    except Exception as error:
        violations.append(f"{type(error).__name__}: {error}")
    return {
        "schema": "bplus-v2.2-task8-manifest-validation-1",
        "passed": not violations,
        "counts": counts,
        "violations": violations,
    }
