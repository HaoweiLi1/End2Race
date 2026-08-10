import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ppo.policy import EvaluatorCompatibleJointDistribution
from utils import atomic_write_json, load_positions_and_speeds_from_params, save_numeric_npz


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-dir", type=Path, default=Path("eval_results/bc_collision_only_anchor_overlap_v2/gate_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("post-trained/panels/collision_bc_anchor_v1"))
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latent_steering_mean(actions):
    means = torch.as_tensor(actions, dtype=torch.float32)
    log_std = torch.log(torch.tensor([0.03, 0.15], dtype=torch.float32))
    distribution = EvaluatorCompatibleJointDistribution().proba_distribution(means, log_std)
    if distribution.latent_steer_mean is None:
        raise RuntimeError("Action distribution did not expose latent steering means")
    return distribution.latent_steer_mean.detach().cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    args = parse_arguments()
    gate_dir = args.gate_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    report_path = gate_dir / "gate_b_report.json"
    plan_path = gate_dir / "gate_b_plan.json"
    branch0_result_path = gate_dir / "branch0" / "results.json"
    full_result_path = gate_dir / "full_bc" / "results.json"
    required = (report_path, plan_path, branch0_result_path, full_result_path)
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Z7 Gate V1 inputs are incomplete")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Anchor dataset output directory must be empty: {output_dir}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    branch0 = json.loads(branch0_result_path.read_text(encoding="utf-8"))
    full = json.loads(full_result_path.read_text(encoding="utf-8"))
    if report.get("experiment_id") != "bc_collision_only_anchor_overlap_v2" or report.get("gate") != "V1":
        raise RuntimeError("Anchor input is not the frozen Z7 collision-only Gate")
    rescued = report["full_bc_admission"]["collision"]["rescued_overtake_scenario_keys"]
    if len(rescued) != 18 or len(set(rescued)) != 18:
        raise RuntimeError("Anchor dataset must contain exactly 18 unique rescued-overtake sources")
    tasks = {task["episode_key"]: task for task in plan["tasks"]}
    if any(key not in tasks for key in rescued):
        raise RuntimeError("A rescued source is missing from the frozen branch plan")

    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_dir = output_dir / "sequences"
    sequence_dir.mkdir()
    rows = []
    for key in rescued:
        task = tasks[key]
        window = task["window"]
        if task["role"] != "cohort" or task["stratum"] != "collision" or window["planned_action_steps"] != 150:
            raise RuntimeError(f"Invalid rescued source contract: {key}")
        if branch0["episodes"][key]["outcome"] not in ("ego-opp", "ego-wall") or full["episodes"][key]["outcome"] != "overtake":
            raise RuntimeError(f"Rescued source outcomes changed: {key}")
        trace_path = gate_dir / "full_bc" / "traces" / f"{key}.npz"
        if not trace_path.is_file():
            raise FileNotFoundError(f"Missing full-BC trace: {trace_path}")
        with np.load(trace_path, allow_pickle=False) as payload:
            lidar = np.asarray(payload["ego_lidar_360"], dtype=np.float32)
            measured_speed = np.asarray(payload["ego_measured_speed_mps"], dtype=np.float32)
            bc_raw_action = np.asarray(payload["bc_raw_action"], dtype=np.float32)
            intervention_active = np.asarray(payload["intervention_active"], dtype=bool)
            action_source_code = np.asarray(payload["action_source_code"], dtype=np.int8)
            action_applied = np.asarray(payload["action_applied"], dtype=bool)
        start = int(window["start_index"])
        end = int(window["end_index_exclusive"])
        if end - start != 150 or end > int(action_applied.sum()) or not bool(action_applied[:end].all()):
            raise RuntimeError(f"Anchor window is not 150 applied steps: {key}")
        expected_active = np.zeros(len(intervention_active), dtype=bool)
        expected_active[start:end] = True
        if not np.array_equal(intervention_active, expected_active) or not bool(np.all(action_source_code[start:end] == 3)):
            raise RuntimeError(f"Anchor intervention markers changed: {key}")

        scenario = task["scenario"]
        params = {"ego_raceline": "raceline1", "opp_raceline": scenario["opp_raceline"], "ego_idx": scenario["ego_idx"], "opp_idx": scenario["opp_idx"]}
        _positions, initial_speeds = load_positions_and_speeds_from_params(params, "Austin")
        previous_speed = np.empty(end, dtype=np.float32)
        previous_speed[0] = np.float32(initial_speeds[0] * 0.9)
        previous_speed[1:] = measured_speed[: end - 1]
        observations = np.concatenate((lidar[:end], previous_speed[:, None]), axis=1).astype(np.float32)
        teacher_action = bc_raw_action[:end]
        teacher_latent = latent_steering_mean(teacher_action)
        anchor_mask = np.zeros(end, dtype=bool)
        anchor_mask[start:end] = True
        arrays = {
            "observations": observations,
            "teacher_latent_steering_mean": teacher_latent,
            "teacher_physical_speed_mean": teacher_action[:, 1].astype(np.float32),
            "anchor_mask": anchor_mask,
        }
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError(f"Anchor sequence contains non-finite data: {key}")
        sequence_path = sequence_dir / f"{key}.npz"
        save_numeric_npz(sequence_path, arrays)
        rows.append({
            "episode_key": key,
            "scenario": scenario,
            "sequence_file": str(sequence_path.relative_to(output_dir)),
            "sequence_sha256": sha256_file(sequence_path),
            "sequence_steps": end,
            "anchor_start_index": start,
            "anchor_end_index_exclusive": end,
            "anchor_steps": int(anchor_mask.sum()),
            "branch0_outcome": branch0["episodes"][key]["outcome"],
            "full_bc_outcome": full["episodes"][key]["outcome"],
        })

    manifest = {
        "schema_version": 1,
        "dataset_id": "collision_bc_anchor_v1",
        "method": "collision-only hindsight-selected counterfactual BC functional regularization",
        "source_experiment": "bc_collision_only_anchor_overlap_v2",
        "source_gate": "V1",
        "source_gate_verdict": report["verdict"],
        "selection_rule": "Z7 collision source, branch0 ego collision, full-BC no ego collision and final overtake, all 150 intervention actions executed",
        "episode_count": len(rows),
        "anchor_steps_per_episode": 150,
        "teacher": "canonical BC closed-loop mean action on the full-BC counterfactual sequence",
        "observation": "360D LiDAR plus previous measured ego speed, reconstructed with the evaluator first-frame speed contract",
        "steering_target": "latent Gaussian mean from the existing evaluator-compatible action distribution",
        "speed_target": "physical Gaussian mean",
        "source_inputs": {
            "gate_b_report_sha256": sha256_file(report_path),
            "gate_b_plan_sha256": sha256_file(plan_path),
            "branch0_results_sha256": sha256_file(branch0_result_path),
            "full_bc_results_sha256": sha256_file(full_result_path),
        },
        "episodes": rows,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"dataset": str(output_dir), "episode_count": len(rows), "anchor_step_count": sum(row["anchor_steps"] for row in rows)}, indent=2))
