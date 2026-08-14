import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from latticeplanner.utils import load_config
from model import End2Race
from ppo.env import EXTERNAL_RESET_OPTION, make_environment
from ppo.scenarios import ScenarioSpec, expanded_scenarios
from utils import atomic_write_json, save_numeric_npz

CONFIG = load_config("ppo/ppo_config.yaml")
CANONICAL_BC = PROJECT_ROOT / "pretrained/end2race.pth"
COLLISION_CACHE = PROJECT_ROOT / "post-trained/collision-cache/pretrained_end2race_austin_collision_pool_479/collision_scenarios.json"
COLLISION_CANDIDATE_OUTCOMES = COLLISION_CACHE.parent / "candidate_outcomes.jsonl"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Build canonical-BC Austin first-action preference data")
    parser.add_argument("--output_dir", default="post-trained/panels/bc_first_action_preference_v1")
    parser.add_argument("--target_source_count", type=int, default=64)
    parser.add_argument("--control_source_count", type=int, default=64)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def actor_action(actor, observation, hidden):
    observation = np.asarray(observation, dtype=np.float32)
    lidar = torch.as_tensor(observation[:360]).reshape(1, 1, 360)
    speed = torch.as_tensor(observation[360:361]).reshape(1, 1, 1)
    with torch.no_grad():
        actions, next_hidden = actor(lidar, speed, hidden)
    action = actions[0, 0].detach().cpu().numpy().astype(np.float32)
    return np.clip(action, (-CONFIG.steering_bound, -8.0), (CONFIG.steering_bound, 8.0)).astype(np.float32), next_hidden


def terminal_score(outcome):
    if outcome == "ego_collision":
        return 0, 0
    if outcome == "follow":
        return 1, 0
    if outcome == "overtake":
        return 1, 1
    raise RuntimeError(f"Unsupported terminal outcome: {outcome!r}")


def strictly_better(left, right):
    left_score = terminal_score(left)
    right_score = terminal_score(right)
    return all(a >= b for a, b in zip(left_score, right_score)) and any(a > b for a, b in zip(left_score, right_score))


def residual_family(residual):
    steering, speed = residual
    if steering and speed:
        return "coordinated"
    return "steering" if steering else "speed"


def baseline_episode(environment, actor, scenario, role):
    observation, _info = environment.reset(options={EXTERNAL_RESET_OPTION: scenario.to_reset_spec(role)})
    start_snapshot = environment.capture_runtime_snapshot()
    observations = []
    actions = []
    hidden_before = []
    clearances = []
    hidden = None
    maximum_steps = int(np.ceil(CONFIG.episode_horizon / CONFIG.simulator_timestep)) + 1
    for _step in range(maximum_steps):
        observations.append(np.asarray(observation, dtype=np.float32).copy())
        hidden_before.append(None if hidden is None else hidden.detach().cpu().numpy().copy())
        action, hidden = actor_action(actor, observation, hidden)
        actions.append(action)
        observation, _reward, terminated, truncated, info = environment.step(action)
        clearances.append(float(environment.current_obb_clearance_m))
        if terminated or truncated:
            return {
                "start_snapshot": start_snapshot,
                "observations": observations,
                "actions": actions,
                "hidden_before": hidden_before,
                "clearances": clearances,
                "outcome": str(info["episode_outcome"]),
            }
    raise RuntimeError("Canonical BC baseline did not reach the true episode terminal")


def prefix_snapshots(environment, baseline, event_step):
    replay_to_lead = {
        event_step - int(lead): int(lead)
        for lead in CONFIG.first_action_preference_lead_steps
        if event_step >= int(lead)
    }
    environment.restore_runtime_snapshot(baseline["start_snapshot"])
    snapshots = {}
    if 0 in replay_to_lead:
        snapshots[replay_to_lead[0]] = environment.capture_runtime_snapshot()
    for replay_count, action in enumerate(baseline["actions"], start=1):
        _observation, _reward, terminated, truncated, _info = environment.step(action)
        if replay_count in replay_to_lead:
            snapshots[replay_to_lead[replay_count]] = environment.capture_runtime_snapshot()
        if len(snapshots) == len(replay_to_lead):
            break
        if terminated or truncated:
            raise RuntimeError("Canonical BC replay terminated before a requested decision prefix")
    if set(snapshots) != set(replay_to_lead.values()):
        raise RuntimeError("Canonical BC replay did not produce every requested prefix")
    return snapshots


def branch_to_terminal(environment, actor, snapshot, first_action, hidden_after):
    observation = environment.restore_runtime_snapshot(snapshot)
    action = np.asarray(first_action, dtype=np.float32)
    hidden = hidden_after.detach().clone()
    maximum_steps = int(np.ceil(CONFIG.episode_horizon / CONFIG.simulator_timestep)) + 1
    for branch_length in range(1, maximum_steps + 1):
        observation, _reward, terminated, truncated, info = environment.step(action)
        if terminated or truncated:
            outcome = str(info["episode_outcome"])
            terminal_score(outcome)
            return outcome, branch_length
        action, hidden = actor_action(actor, observation, hidden)
    raise RuntimeError("Canonical BC branch did not reach the true episode terminal")


def label_episode(environment, actor, scenario, role, baseline, event_step):
    snapshots = prefix_snapshots(environment, baseline, event_step)
    states = []
    branch_count = 0
    simulator_steps = 0
    outcome_counts = {"ego_collision": 0, "overtake": 0, "follow": 0}
    for lead in CONFIG.first_action_preference_lead_steps:
        lead = int(lead)
        if lead not in snapshots:
            continue
        decision_index = event_step - lead
        hidden_array = baseline["hidden_before"][decision_index]
        hidden = None if hidden_array is None else torch.as_tensor(hidden_array, dtype=torch.float32)
        noop, hidden_after = actor_action(actor, baseline["observations"][decision_index], hidden)
        if not np.array_equal(noop, baseline["actions"][decision_index]):
            raise RuntimeError("Canonical BC decision action changed during prefix reconstruction")
        actions = [noop]
        residuals = [None]
        for raw_residual in CONFIG.first_action_preference_action_residuals:
            residual = np.asarray(raw_residual, dtype=np.float32)
            candidate = np.clip(noop + residual, (-CONFIG.steering_bound, -8.0), (CONFIG.steering_bound, 8.0)).astype(np.float32)
            if any(np.array_equal(candidate, existing) for existing in actions):
                continue
            actions.append(candidate)
            residuals.append(tuple(float(value) for value in residual))
        outcomes = []
        for action in actions:
            outcome, branch_length = branch_to_terminal(environment, actor, snapshots[lead], action, hidden_after)
            outcomes.append(outcome)
            branch_count += 1
            simulator_steps += branch_length
            outcome_counts[outcome] += 1
        noop_outcome = outcomes[0]
        pairs = []
        for candidate, residual, candidate_outcome in zip(actions[1:], residuals[1:], outcomes[1:]):
            if strictly_better(candidate_outcome, noop_outcome):
                good, bad, direction = candidate, noop, "candidate_preferred"
            elif strictly_better(noop_outcome, candidate_outcome):
                good, bad, direction = noop, candidate, "noop_preferred"
            else:
                continue
            pairs.append({
                "family": residual_family(residual),
                "direction": direction,
                "good_action": good.tolist(),
                "bad_action": bad.tolist(),
                "noop_outcome": noop_outcome,
                "candidate_outcome": candidate_outcome,
            })
        if pairs:
            states.append({"decision_index": decision_index, "lead_steps": lead, "pairs": pairs})
    return states, branch_count, simulator_steps, outcome_counts


def source_scenarios(target_count, control_count):
    collision_rows = json.loads(COLLISION_CACHE.read_text(encoding="utf-8"))
    targets = tuple(ScenarioSpec(**row) for row in collision_rows)
    candidates = expanded_scenarios("Austin", CONFIG)
    cached_outcomes = [json.loads(line) for line in COLLISION_CANDIDATE_OUTCOMES.read_text(encoding="utf-8").splitlines()]
    if len(cached_outcomes) != len(candidates):
        raise RuntimeError("Canonical-BC collision-candidate outcome cache is incomplete")
    if any(outcome.get("candidate_index") != index or outcome.get("scenario_id") != scenario.scenario_id for index, (scenario, outcome) in enumerate(zip(candidates, cached_outcomes))):
        raise RuntimeError("Canonical-BC collision-candidate outcome identities changed")
    controls = [
        scenario
        for scenario, outcome in zip(candidates, cached_outcomes)
        if outcome["scenario_id"] == scenario.scenario_id and outcome["outcome"] == "other"
    ]
    controls.sort(key=lambda scenario: (-scenario.opp_speedscale, scenario.interval_idx, scenario.startpoint_ordinal, scenario.opp_raceline))
    if len(targets) < target_count or control_count <= 0:
        raise ValueError("Requested canonical-BC source counts are unavailable")
    return targets, tuple(controls)


def build_dataset(output_dir, target_count, control_count):
    if target_count <= 0 or control_count <= 0:
        raise ValueError("Canonical-BC target and control source counts must be positive")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite first-action preference data: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    episodes_dir = temporary / "episodes"
    episodes_dir.mkdir()
    actor = End2Race(hidden_scale=4)
    actor.load_state_dict(torch.load(CANONICAL_BC, map_location="cpu", weights_only=True), strict=True)
    actor.eval()
    environment = make_environment(42, "Austin", privileged=False, reward_gamma=0.999)()
    target_candidates, control_candidates = source_scenarios(target_count, control_count)
    rows = []
    telemetry = {"branch_count": 0, "simulator_steps": 0, "outcome_counts": {"ego_collision": 0, "overtake": 0, "follow": 0}}

    def process_source(role, scenario, baseline, source_index):
        event_step = len(baseline["actions"]) if role == "target" else int(np.argmin(baseline["clearances"])) + 1
        states, branch_count, simulator_steps, outcome_counts = label_episode(environment, actor, scenario, role, baseline, event_step)
        telemetry["branch_count"] += branch_count
        telemetry["simulator_steps"] += simulator_steps
        for outcome, count in outcome_counts.items():
            telemetry["outcome_counts"][outcome] += count
        if not states:
            print(f"[{source_index}/{target_count + control_count}] {role} {scenario.scenario_id}: no strict-Pareto pair", flush=True)
            return
        maximum_index = max(state["decision_index"] for state in states)
        sequence_path = episodes_dir / f"{scenario.scenario_id}.npz"
        observations = np.asarray(baseline["observations"][: maximum_index + 1], dtype=np.float32)
        save_numeric_npz(sequence_path, {"observations": observations, "episode_starts": np.asarray([True] + [False] * maximum_index, dtype=bool)})
        rows.append({
            "episode_key": scenario.scenario_id,
            "role": role,
            "stratum": "canonical_bc_collision" if role == "target" else "canonical_bc_safe_overtake",
            "ego_idx": scenario.ego_idx,
            "opp_raceline": scenario.opp_raceline,
            "opp_speedscale": scenario.opp_speedscale,
            "sequence_file": str(sequence_path.relative_to(temporary)),
            "sequence_sha256": sha256_file(sequence_path),
            "sequence_steps": len(observations),
            "states": states,
        })
        print(f"[{source_index}/{target_count + control_count}] {role} {scenario.scenario_id}: {sum(len(state['pairs']) for state in states)} pairs", flush=True)

    try:
        selected_target_count = 0
        for scenario in target_candidates:
            baseline = baseline_episode(environment, actor, scenario, "collision")
            if baseline["outcome"] == "ego_collision":
                selected_target_count += 1
                process_source("target", scenario, baseline, selected_target_count)
                if selected_target_count == target_count:
                    break
        if selected_target_count != target_count:
            raise RuntimeError("Canonical BC collision training pool did not provide the requested collision sources")
        selected_control_count = 0
        for scenario in control_candidates:
            baseline = baseline_episode(environment, actor, scenario, "collision")
            if baseline["outcome"] == "overtake":
                selected_control_count += 1
                process_source("control", scenario, baseline, target_count + selected_control_count)
                if selected_control_count == control_count:
                    break
        if selected_control_count != control_count:
            raise RuntimeError("Canonical BC difficult candidate pool did not provide the requested safe controls")
    finally:
        environment.close()
    target_rows = [row for row in rows if row["role"] == "target"]
    control_rows = [row for row in rows if row["role"] == "control"]
    if not target_rows or not control_rows:
        shutil.rmtree(temporary)
        raise RuntimeError("Canonical-BC preference data requires at least one labeled target and control episode")
    gate = {
        "schema_version": 1,
        "source_actor": "canonical_bc",
        "target_source_count": target_count,
        "control_source_count": control_count,
        "target_labeled_episode_count": len(target_rows),
        "control_labeled_episode_count": len(control_rows),
        "target_pair_count": sum(len(state["pairs"]) for row in target_rows for state in row["states"]),
        "control_pair_count": sum(len(state["pairs"]) for row in control_rows for state in row["states"]),
        **telemetry,
        "verdict": "pass",
    }
    gate_path = temporary / "gate_report.json"
    atomic_write_json(gate_path, gate)
    manifest = {
        "schema_version": 1,
        "status": "ready_for_formal",
        "source_contract": {
            "actor": str(CANONICAL_BC.resolve()),
            "actor_sha256": sha256_file(CANONICAL_BC),
            "map": "Austin",
            "collision_pool": str(COLLISION_CACHE.resolve()),
            "collision_pool_sha256": sha256_file(COLLISION_CACHE),
            "uses_previous_ppo_actor_or_trace": False,
            "uses_evaluation_panel": False,
            "lead_steps": list(CONFIG.first_action_preference_lead_steps),
            "action_residuals": list(CONFIG.first_action_preference_action_residuals),
            "continuation": "canonical BC deterministic to true terminal",
        },
        "gate_report_sha256": sha256_file(gate_path),
        "loss_contract": {
            "terminal_vector": ["no_ego_collision", "overtake"],
            "pair_loss": "softplus(-(logp_good-logp_bad))",
            "state_weight": "mean pairs",
            "episode_weight": "mean labeled states",
            "role_weight": {"target": 0.5, "control": 0.5},
        },
        "episodes": rows,
    }
    atomic_write_json(temporary / "manifest.json", manifest)
    temporary.replace(output)
    return gate


if __name__ == "__main__":
    arguments = parse_arguments()
    torch.manual_seed(42)
    np.random.seed(42)
    print(json.dumps(build_dataset(arguments.output_dir, arguments.target_source_count, arguments.control_source_count), indent=2, sort_keys=True))
