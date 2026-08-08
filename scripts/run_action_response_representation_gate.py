import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import End2Race
from utils import atomic_write_json


OUTCOME_TO_CLASS = {
    "ego-opp": 0,
    "ego-wall": 0,
    "follow": 1,
    "overtake": 2,
}
CLASS_NAMES = ("collision", "follow", "overtake")
TARGET_STRATA = ("inherited_collision", "created_collision", "lost_overtake")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--branch0-results", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--hidden-root", type=Path, required=True)
    parser.add_argument("--u44-trace-root", type=Path, required=True)
    parser.add_argument("--u44-model-path", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--nested-operating-point", action="store_true")
    parser.add_argument("--nested-outer-z4-seeds", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_exact_p(left_only, right_only):
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    smaller = min(left_only, right_only)
    tail = sum(math.comb(discordant, value) for value in range(smaller + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def selected_record(task, action_name, branch0_results, candidate_results):
    if action_name == "noop":
        return branch0_results[task["episode_key"]]
    return candidate_results[f"{task['episode_key']}::late::{action_name}"]


def validate_inputs(plan, branch0, candidates, args):
    tasks = plan["tasks"]
    task_keys = {task["episode_key"] for task in tasks}
    candidate_names = [candidate["name"] for candidate in plan["candidate_contract"]["candidates"]]
    expected_candidate_keys = {f"{key}::{prefix}::{name}" for key in task_keys for prefix in ("early", "late") for name in candidate_names}
    if len(tasks) != 456 or len(task_keys) != 456:
        raise RuntimeError("Round Z2 task identity contract failed")
    if set(branch0["episodes"]) != task_keys or branch0["summary"] != {"episode_count": 456, "error_count": 0}:
        raise RuntimeError("branch0 result contract failed")
    if set(candidates["episodes"]) != expected_candidate_keys or candidates["summary"] != {"episode_count": 10944, "error_count": 0}:
        raise RuntimeError("candidate result contract failed")
    if sha256_file(args.u44_model_path) != plan["inputs"]["u44_model_sha256"]:
        raise RuntimeError("U44 model identity changed")
    if Path(plan["inputs"]["u44_model_path"]) != args.u44_model_path:
        raise RuntimeError("U44 model path does not match the frozen plan")
    fold_startpoints = {}
    stratum_counts = {}
    for task in tasks:
        fold = int(task["fold"])
        ego_idx = int(task["scenario"]["ego_idx"])
        fold_startpoints.setdefault(fold, set()).add(ego_idx)
        stratum_counts[task["stratum"]] = stratum_counts.get(task["stratum"], 0) + 1
        expected_fold = int(hashlib.sha256(f"counterfactual-action-fold-v1|{ego_idx}".encode("utf-8")).hexdigest(), 16) % 5
        if fold != expected_fold:
            raise RuntimeError(f"fold assignment changed for {task['episode_key']}")
        if int(task["prefixes"]["late"]["start_index"]) < 50:
            raise RuntimeError(f"late prefix lacks 50-step actor history: {task['episode_key']}")
    for left in range(5):
        for right in range(left + 1, 5):
            if fold_startpoints.get(left, set()) & fold_startpoints.get(right, set()):
                raise RuntimeError("ego startpoint leakage across folds")
    expected_strata = {"inherited_collision": 109, "created_collision": 46, "lost_overtake": 13, "inherited_follow": 63, "safe_control": 225}
    if stratum_counts != expected_strata:
        raise RuntimeError(f"stratum counts changed: {stratum_counts}")
    for record in list(branch0["episodes"].values()) + list(candidates["episodes"].values()):
        if record["outcome"] not in OUTCOME_TO_CLASS:
            raise RuntimeError(f"unsupported outcome {record['outcome']}")
    return candidate_names, stratum_counts, {str(fold): len(fold_startpoints[fold]) for fold in sorted(fold_startpoints)}


def load_actor_features(plan, args, device):
    actor = End2Race(mask_prob=0.0, hidden_scale=4).to(device)
    actor.load_state_dict(torch.load(args.u44_model_path, map_location=device, weights_only=True), strict=True)
    actor.eval()
    features = []
    hidden = []
    input_rows = 0
    for task in plan["tasks"]:
        key = task["episode_key"]
        start_index = int(task["prefixes"]["late"]["start_index"])
        indices = np.arange(start_index - 49, start_index + 1, dtype=np.int64)
        trace_path = args.u44_trace_root / f"{key}.npz"
        hidden_path = args.hidden_root / f"{key}.npz"
        if not trace_path.is_file() or not hidden_path.is_file():
            raise RuntimeError(f"missing frozen input for {key}")
        with np.load(trace_path, allow_pickle=False) as trace:
            if "ego_lidar_360" not in trace or "ego_measured_speed_mps" not in trace:
                raise RuntimeError(f"actor-visible trace fields missing for {key}")
            lidar = np.asarray(trace["ego_lidar_360"][indices], dtype=np.float32)
            previous_speed = np.asarray(trace["ego_measured_speed_mps"][indices - 1], dtype=np.float32).reshape(50, 1)
        if lidar.shape != (50, 360) or previous_speed.shape != (50, 1) or not np.isfinite(lidar).all() or not np.isfinite(previous_speed).all():
            raise RuntimeError(f"invalid actor-visible history for {key}")
        with torch.no_grad():
            lidar_tensor = torch.tensor(lidar, dtype=torch.float32, device=device)
            speed_tensor = torch.tensor(previous_speed, dtype=torch.float32, device=device)
            processed_lidar = (-1 / (1 + torch.exp(-actor.k * lidar_tensor)) + 1) * 2
            speed_embedding = actor.speed_mlp(speed_tensor)
            feature = torch.cat((processed_lidar, speed_embedding), dim=1).cpu().numpy()
        with np.load(hidden_path, allow_pickle=False) as payload:
            hidden_vector = np.asarray(payload["late"], dtype=np.float32)
        if feature.shape != (50, 420) or hidden_vector.shape != (1680,) or not np.isfinite(feature).all() or not np.isfinite(hidden_vector).all():
            raise RuntimeError(f"invalid transformed actor input for {key}")
        features.append(feature)
        hidden.append(hidden_vector)
        input_rows += len(indices)
    return np.stack(features).astype(np.float32), np.stack(hidden).astype(np.float32), input_rows


def build_labels(plan, branch0_results, candidate_results, candidate_names):
    names = ["noop"] + candidate_names
    labels = np.empty((len(plan["tasks"]), len(names)), dtype=np.int64)
    for task_index, task in enumerate(plan["tasks"]):
        for action_index, action_name in enumerate(names):
            labels[task_index, action_index] = OUTCOME_TO_CLASS[selected_record(task, action_name, branch0_results, candidate_results)["outcome"]]
    return names, labels


def action_tensor(plan, device):
    actions = [[0.0, 0.0]]
    for candidate in plan["candidate_contract"]["candidates"]:
        actions.append([float(candidate["steering_delta_rad"]) / 0.02, float(candidate["speed_delta_mps"]) / 0.5])
    return torch.tensor(actions, dtype=torch.float32, device=device)


class FrozenHiddenOutcomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Sequential(nn.Linear(1680, 192), nn.ReLU())
        self.outcome_layer = nn.Sequential(nn.Linear(194, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, hidden, actions):
        hidden_encoded = self.hidden_layer(hidden)
        hidden_batch = hidden_encoded.unsqueeze(1).expand(-1, actions.shape[0], -1)
        action_batch = actions.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        return self.outcome_layer(torch.cat((hidden_batch, action_batch), dim=2))


class HistoryOutcomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Sequential(nn.Linear(1680, 128), nn.ReLU())
        self.history_input = nn.Sequential(nn.Linear(420, 64), nn.ReLU())
        self.history_gru = nn.GRU(64, 64, batch_first=True)
        self.outcome_layer = nn.Sequential(nn.Linear(194, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, hidden, history, actions):
        hidden_encoded = self.hidden_layer(hidden)
        history_encoded = self.history_input(history)
        _, history_hidden = self.history_gru(history_encoded)
        state = torch.cat((hidden_encoded, history_hidden[-1]), dim=1)
        state_batch = state.unsqueeze(1).expand(-1, actions.shape[0], -1)
        action_batch = actions.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        return self.outcome_layer(torch.cat((state_batch, action_batch), dim=2))


def train_model(model, train_hidden, train_history, train_labels, actions, class_weights, seed, shuffle_seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model.apply(reset_parameters)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(shuffle_seed)
    final_loss = None
    for _ in range(100):
        order = torch.randperm(len(train_hidden), generator=generator)
        losses = []
        for start in range(0, len(order), 64):
            batch = order[start:start + 64].to(train_hidden.device)
            if train_history is None:
                logits = model(train_hidden[batch], actions)
            else:
                logits = model(train_hidden[batch], train_history[batch], actions)
            loss = nn.functional.cross_entropy(logits.reshape(-1, 3), train_labels[batch].reshape(-1), weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses))
    return final_loss


def reset_parameters(module):
    if isinstance(module, (nn.Linear, nn.GRU)):
        module.reset_parameters()


def choose_actions(probabilities):
    scores = probabilities[:, :, 2] - 5.0 * probabilities[:, :, 0]
    return scores.argmax(axis=1)


def fixed_action_for_fold(plan, names, train_indices, branch0_results, candidate_results):
    scores = []
    for action_index, action_name in enumerate(names):
        success = 0
        harm = 0
        for index in train_indices:
            task = plan["tasks"][index]
            outcome = selected_record(task, action_name, branch0_results, candidate_results)["outcome"]
            if task["stratum"] == "safe_control":
                harm += int(outcome != "overtake")
            else:
                success += int(outcome == "overtake")
        scores.append((success - 5 * harm, -action_index, action_index))
    return max(scores)[2]


def summarize(plan, names, selected, branch0_results, candidate_results):
    successes = {name: 0 for name in ("inherited_collision", "created_collision", "lost_overtake", "inherited_follow")}
    selected_action_by_episode = {}
    selected_outcome_by_episode = {}
    action_counts = {}
    control_new_collision = 0
    control_overtake_loss = 0
    collision_removed = 0
    collision_created = 0
    overtake_lost = 0
    overtake_gained = 0
    for index, task in enumerate(plan["tasks"]):
        action_name = names[int(selected[index])]
        record = selected_record(task, action_name, branch0_results, candidate_results)
        outcome = record["outcome"]
        selected_action_by_episode[task["episode_key"]] = action_name
        selected_outcome_by_episode[task["episode_key"]] = outcome
        action_counts[action_name] = action_counts.get(action_name, 0) + 1
        source_collision = task["source_outcome"] in ("ego-opp", "ego-wall")
        selected_collision = outcome in ("ego-opp", "ego-wall")
        source_overtake = task["source_outcome"] == "overtake"
        selected_overtake = outcome == "overtake"
        collision_removed += int(source_collision and not selected_collision)
        collision_created += int(not source_collision and selected_collision)
        overtake_lost += int(source_overtake and not selected_overtake)
        overtake_gained += int(not source_overtake and selected_overtake)
        if task["stratum"] == "safe_control":
            control_new_collision += int(selected_collision)
            control_overtake_loss += int(not selected_overtake)
        else:
            successes[task["stratum"]] += int(selected_overtake)
    return {
        "success_counts": successes,
        "target_success_total": sum(successes[name] for name in TARGET_STRATA),
        "safe_control_new_collision_count": control_new_collision,
        "safe_control_overtake_loss_count": control_overtake_loss,
        "selected_action_counts": dict(sorted(action_counts.items())),
        "selected_action_by_episode": selected_action_by_episode,
        "selected_outcome_by_episode": selected_outcome_by_episode,
        "paired_vs_noop": {
            "collision_removed": collision_removed,
            "collision_created": collision_created,
            "collision_exact_p": paired_exact_p(collision_removed, collision_created),
            "overtake_lost": overtake_lost,
            "overtake_gained": overtake_gained,
            "overtake_exact_p": paired_exact_p(overtake_lost, overtake_gained),
        },
    }


LAMBDA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 32.0)


def inner_fold_for_startpoint(ego_idx):
    digest = hashlib.sha256(f"action-response-inner-v1|{int(ego_idx)}".encode("utf-8")).hexdigest()
    return int(digest, 16) % 4


def fit_predict_outcomes(model_name, hidden, histories, labels, train_indices, test_indices, actions, device, seed, shuffle_seed):
    hidden_mean = hidden[train_indices].mean(axis=0)
    hidden_std = hidden[train_indices].std(axis=0)
    hidden_std[hidden_std < 1e-6] = 1.0
    train_hidden = torch.tensor((hidden[train_indices] - hidden_mean) / hidden_std, dtype=torch.float32, device=device)
    test_hidden = torch.tensor((hidden[test_indices] - hidden_mean) / hidden_std, dtype=torch.float32, device=device)
    if model_name == "frozen_hidden":
        model = FrozenHiddenOutcomeModel().to(device)
        train_history = None
        test_history = None
    elif model_name == "history":
        feature_mean = histories[train_indices].reshape(-1, 420).mean(axis=0)
        feature_std = histories[train_indices].reshape(-1, 420).std(axis=0)
        feature_std[feature_std < 1e-6] = 1.0
        train_history = torch.tensor((histories[train_indices] - feature_mean) / feature_std, dtype=torch.float32, device=device)
        test_history = torch.tensor((histories[test_indices] - feature_mean) / feature_std, dtype=torch.float32, device=device)
        model = HistoryOutcomeModel().to(device)
    else:
        raise ValueError(f"unknown operating-point model {model_name}")
    train_labels = torch.tensor(labels[train_indices], dtype=torch.long, device=device)
    class_counts = np.bincount(labels[train_indices].reshape(-1), minlength=3)
    if bool(np.any(class_counts == 0)):
        raise RuntimeError(f"{model_name} training split has an empty outcome class")
    class_weights = torch.tensor(len(train_indices) * labels.shape[1] / (3.0 * class_counts), dtype=torch.float32, device=device)
    final_loss = train_model(model, train_hidden, train_history, train_labels, actions, class_weights, seed, shuffle_seed)
    with torch.no_grad():
        if model_name == "frozen_hidden":
            probabilities = torch.softmax(model(test_hidden, actions), dim=2).cpu().numpy()
        else:
            probabilities = torch.softmax(model(test_hidden, test_history, actions), dim=2).cpu().numpy()
    del model, train_hidden, test_hidden, train_history, test_history, train_labels
    torch.cuda.empty_cache()
    return probabilities, final_loss, class_counts


def operating_point_predictions(probabilities, lambda_cost, tau):
    scores = probabilities[:, :, 2] - float(lambda_cost) * probabilities[:, :, 0]
    best_nonnoop = scores[:, 1:].argmax(axis=1) + 1
    rows = np.arange(len(probabilities))
    margins = scores[rows, best_nonnoop] - scores[:, 0]
    selected = np.where(margins >= float(tau), best_nonnoop, 0)
    return selected.astype(np.int64), margins


def report_threshold(tau):
    return float(tau) if np.isfinite(tau) else "infinity"


def subset_metrics(plan, names, indices, selected, branch0_results, candidate_results):
    target_success = 0
    control_count = 0
    control_new_collision = 0
    control_overtake_loss = 0
    intervention_count = 0
    success_counts = {name: 0 for name in TARGET_STRATA}
    for local_index, task_index in enumerate(indices):
        task = plan["tasks"][int(task_index)]
        action_name = names[int(selected[local_index])]
        outcome = selected_record(task, action_name, branch0_results, candidate_results)["outcome"]
        intervention_count += int(action_name != "noop")
        if task["stratum"] == "safe_control":
            control_count += 1
            control_new_collision += int(outcome in ("ego-opp", "ego-wall"))
            control_overtake_loss += int(outcome != "overtake")
        elif task["stratum"] in TARGET_STRATA:
            success = int(outcome == "overtake")
            success_counts[task["stratum"]] += success
            target_success += success
    return {
        "target_success_total": target_success,
        "success_counts": success_counts,
        "safe_control_count": control_count,
        "safe_control_new_collision_count": control_new_collision,
        "safe_control_overtake_loss_count": control_overtake_loss,
        "intervention_count": intervention_count,
    }


def choose_inner_operating_point(plan, names, train_indices, probabilities, branch0_results, candidate_results):
    candidates = []
    for lambda_index, lambda_cost in enumerate(LAMBDA_GRID):
        _, margins = operating_point_predictions(probabilities, lambda_cost, float("inf"))
        thresholds = [float(value) for value in np.unique(margins)] + [float("inf")]
        for tau in thresholds:
            selected, _ = operating_point_predictions(probabilities, lambda_cost, tau)
            metrics = subset_metrics(plan, names, train_indices, selected, branch0_results, candidate_results)
            controls = metrics["safe_control_count"]
            feasible = metrics["safe_control_new_collision_count"] * 225 <= 5 * controls and metrics["safe_control_overtake_loss_count"] * 225 <= 5 * controls
            if not feasible:
                continue
            key = (
                metrics["target_success_total"],
                -metrics["safe_control_new_collision_count"],
                -metrics["safe_control_overtake_loss_count"],
                -metrics["intervention_count"],
                float(lambda_cost),
                float(tau),
                -lambda_index,
            )
            candidates.append((key, float(lambda_cost), float(tau), metrics))
    if not candidates:
        raise RuntimeError("all-noop operating point must make the inner calibration feasible")
    _, lambda_cost, tau, metrics = max(candidates, key=lambda item: item[0])
    return lambda_cost, tau, metrics, len(candidates)


def empirical_frontier(plan, names, probabilities, branch0_results, candidate_results):
    indices = np.arange(len(plan["tasks"]), dtype=np.int64)
    best_by_harm = {}
    for lambda_cost in LAMBDA_GRID:
        _, margins = operating_point_predictions(probabilities, lambda_cost, float("inf"))
        thresholds = [float(value) for value in np.unique(margins)] + [float("inf")]
        for tau in thresholds:
            selected, _ = operating_point_predictions(probabilities, lambda_cost, tau)
            metrics = subset_metrics(plan, names, indices, selected, branch0_results, candidate_results)
            harm = (metrics["safe_control_new_collision_count"], metrics["safe_control_overtake_loss_count"])
            record = {"lambda": float(lambda_cost), "tau": float(tau), **metrics}
            previous = best_by_harm.get(harm)
            if previous is None or (record["target_success_total"], -record["intervention_count"], record["lambda"], record["tau"]) > (previous["target_success_total"], -previous["intervention_count"], previous["lambda"], previous["tau"]):
                best_by_harm[harm] = record
    records = list(best_by_harm.values())
    frontier = []
    for record in records:
        dominated = any(
            other["safe_control_new_collision_count"] <= record["safe_control_new_collision_count"]
            and other["safe_control_overtake_loss_count"] <= record["safe_control_overtake_loss_count"]
            and other["target_success_total"] >= record["target_success_total"]
            and (
                other["safe_control_new_collision_count"] < record["safe_control_new_collision_count"]
                or other["safe_control_overtake_loss_count"] < record["safe_control_overtake_loss_count"]
                or other["target_success_total"] > record["target_success_total"]
            )
            for other in records
        )
        if not dominated:
            frontier.append(record)
    frontier.sort(key=lambda item: (item["safe_control_new_collision_count"], item["safe_control_overtake_loss_count"], item["target_success_total"], item["intervention_count"]))
    matched = [record for record in records if record["safe_control_new_collision_count"] <= 5 and record["safe_control_overtake_loss_count"] <= 5]
    matched_best = max(matched, key=lambda item: (item["target_success_total"], -item["safe_control_new_collision_count"], -item["safe_control_overtake_loss_count"], -item["intervention_count"], item["lambda"], item["tau"]))
    frontier = [{**record, "tau": report_threshold(record["tau"])} for record in frontier]
    matched_best = {**matched_best, "tau": report_threshold(matched_best["tau"])}
    return frontier, matched_best


def paired_selector_comparison(plan, left, right, branch0_results, candidate_results, names):
    target_left_only = 0
    target_right_only = 0
    control_collision_left_only = 0
    control_collision_right_only = 0
    control_loss_left_only = 0
    control_loss_right_only = 0
    for index, task in enumerate(plan["tasks"]):
        left_outcome = selected_record(task, names[int(left[index])], branch0_results, candidate_results)["outcome"]
        right_outcome = selected_record(task, names[int(right[index])], branch0_results, candidate_results)["outcome"]
        if task["stratum"] in TARGET_STRATA:
            target_left_only += int(left_outcome == "overtake" and right_outcome != "overtake")
            target_right_only += int(right_outcome == "overtake" and left_outcome != "overtake")
        if task["stratum"] == "safe_control":
            left_collision = left_outcome in ("ego-opp", "ego-wall")
            right_collision = right_outcome in ("ego-opp", "ego-wall")
            control_collision_left_only += int(left_collision and not right_collision)
            control_collision_right_only += int(right_collision and not left_collision)
            control_loss_left_only += int(left_outcome != "overtake" and right_outcome == "overtake")
            control_loss_right_only += int(right_outcome != "overtake" and left_outcome == "overtake")
    return {
        "target_left_only": target_left_only,
        "target_right_only": target_right_only,
        "target_exact_p": paired_exact_p(target_left_only, target_right_only),
        "control_collision_left_only": control_collision_left_only,
        "control_collision_right_only": control_collision_right_only,
        "control_collision_exact_p": paired_exact_p(control_collision_left_only, control_collision_right_only),
        "control_overtake_loss_left_only": control_loss_left_only,
        "control_overtake_loss_right_only": control_loss_right_only,
        "control_overtake_loss_exact_p": paired_exact_p(control_loss_left_only, control_loss_right_only),
    }


def run_nested_operating_point(plan, names, labels, histories, hidden, actions, branch0_results, candidate_results, args, device):
    fixed_predictions = np.full(len(plan["tasks"]), -1, dtype=np.int64)
    model_results = {}
    for model_name in ("frozen_hidden", "history"):
        outer_probabilities = np.full((len(plan["tasks"]), len(names), 3), np.nan, dtype=np.float32)
        outer_predictions = np.full(len(plan["tasks"]), -1, dtype=np.int64)
        fold_records = []
        for outer_fold in range(5):
            outer_train = np.asarray([index for index, task in enumerate(plan["tasks"]) if int(task["fold"]) != outer_fold], dtype=np.int64)
            outer_test = np.asarray([index for index, task in enumerate(plan["tasks"]) if int(task["fold"]) == outer_fold], dtype=np.int64)
            inner_probabilities = np.full((len(outer_train), len(names), 3), np.nan, dtype=np.float32)
            inner_folds = np.asarray([inner_fold_for_startpoint(plan["tasks"][int(index)]["scenario"]["ego_idx"]) for index in outer_train], dtype=np.int64)
            inner_records = []
            for inner_fold in range(4):
                inner_train = outer_train[inner_folds != inner_fold]
                inner_test_positions = np.flatnonzero(inner_folds == inner_fold)
                inner_test = outer_train[inner_test_positions]
                if len(inner_train) == 0 or len(inner_test) == 0:
                    raise RuntimeError(f"empty nested split outer={outer_fold} inner={inner_fold}")
                probabilities, final_loss, class_counts = fit_predict_outcomes(model_name, hidden, histories, labels, inner_train, inner_test, actions, device, 6200 + 10 * outer_fold + inner_fold, 6300 + 10 * outer_fold + inner_fold)
                inner_probabilities[inner_test_positions] = probabilities
                inner_records.append({
                    "inner_fold": inner_fold,
                    "train_scenario_count": int(len(inner_train)),
                    "test_scenario_count": int(len(inner_test)),
                    "train_startpoint_count": len({int(plan["tasks"][int(index)]["scenario"]["ego_idx"]) for index in inner_train}),
                    "test_startpoint_count": len({int(plan["tasks"][int(index)]["scenario"]["ego_idx"]) for index in inner_test}),
                    "class_counts": {CLASS_NAMES[index]: int(class_counts[index]) for index in range(3)},
                    "final_train_loss": final_loss,
                })
            if not np.isfinite(inner_probabilities).all():
                raise RuntimeError(f"incomplete inner-OOF probabilities for outer fold {outer_fold}")
            lambda_cost, tau, inner_metrics, feasible_count = choose_inner_operating_point(plan, names, outer_train, inner_probabilities, branch0_results, candidate_results)
            outer_model_seed = (5200 if args.nested_outer_z4_seeds else 6400) + outer_fold
            outer_shuffle_seed = (5300 if args.nested_outer_z4_seeds else 6500) + outer_fold
            probabilities, final_loss, class_counts = fit_predict_outcomes(model_name, hidden, histories, labels, outer_train, outer_test, actions, device, outer_model_seed, outer_shuffle_seed)
            selected, margins = operating_point_predictions(probabilities, lambda_cost, tau)
            outer_probabilities[outer_test] = probabilities
            outer_predictions[outer_test] = selected
            outer_metrics = subset_metrics(plan, names, outer_test, selected, branch0_results, candidate_results)
            fold_records.append({
                "outer_fold": outer_fold,
                "train_scenario_count": int(len(outer_train)),
                "test_scenario_count": int(len(outer_test)),
                "selected_lambda": lambda_cost,
                "selected_tau": report_threshold(tau),
                "inner_feasible_operating_point_count": feasible_count,
                "inner_selected_metrics": inner_metrics,
                "outer_test_metrics": outer_metrics,
                "outer_margin_minimum": float(np.min(margins)),
                "outer_margin_maximum": float(np.max(margins)),
                "outer_model_class_counts": {CLASS_NAMES[index]: int(class_counts[index]) for index in range(3)},
                "outer_model_final_train_loss": final_loss,
                "inner_folds": inner_records,
            })
            if model_name == "frozen_hidden":
                fixed_index = fixed_action_for_fold(plan, names, outer_train, branch0_results, candidate_results)
                fixed_predictions[outer_test] = fixed_index
        if not np.isfinite(outer_probabilities).all() or bool(np.any(outer_predictions < 0)):
            raise RuntimeError(f"incomplete outer predictions for {model_name}")
        summary = summarize(plan, names, outer_predictions, branch0_results, candidate_results)
        frontier, matched_best = empirical_frontier(plan, names, outer_probabilities, branch0_results, candidate_results)
        model_results[model_name] = {
            "nested_summary": summary,
            "outer_folds": fold_records,
            "diagnostic_outer_oof_frontier": frontier,
            "diagnostic_outer_oof_best_at_harm_5_5": matched_best,
            "selected_indices": outer_predictions,
        }
    if bool(np.any(fixed_predictions < 0)):
        raise RuntimeError("incomplete grouped fixed baseline predictions")
    fixed_summary = summarize(plan, names, fixed_predictions, branch0_results, candidate_results)
    if fixed_summary["target_success_total"] != 79 or fixed_summary["safe_control_new_collision_count"] != 5 or fixed_summary["safe_control_overtake_loss_count"] != 5:
        raise RuntimeError("grouped fixed baseline identity changed")
    frozen_summary = model_results["frozen_hidden"]["nested_summary"]
    history_summary = model_results["history"]["nested_summary"]
    criteria = {
        "frozen_hidden_safe_control_new_collision_matched": frozen_summary["safe_control_new_collision_count"] <= 5,
        "frozen_hidden_safe_control_overtake_loss_matched": frozen_summary["safe_control_overtake_loss_count"] <= 5,
        "frozen_hidden_target_strictly_above_fixed": frozen_summary["target_success_total"] > 79,
    }
    frozen_predictions = model_results["frozen_hidden"].pop("selected_indices")
    history_predictions = model_results["history"].pop("selected_indices")
    return {
        "schema_version": 1,
        "experiment_id": "harm_matched_frozen_hidden_operating_point",
        "gate": "round_z5_nested_grouped_cv",
        "verdict": "reopen_frozen_hidden_conditional_value" if all(criteria.values()) else "close_tested_harm_matched_outcome_selector",
        "frozen_hidden_conditional_value_reopened": all(criteria.values()),
        "model_contract": {
            "lambda_grid": list(LAMBDA_GRID),
            "tau_candidates": "all unique inner-OOF best-nonnoop minus noop margins plus infinity",
            "outer_fold": "SHA256(counterfactual-action-fold-v1|ego_idx) mod 5",
            "inner_fold": "SHA256(action-response-inner-v1|ego_idx) mod 4",
            "control_collision_rate_maximum": "5/225",
            "control_overtake_loss_rate_maximum": "5/225",
            "inner_model_seed": "6200+10*outer_fold+inner_fold",
            "inner_shuffle_seed": "6300+10*outer_fold+inner_fold",
            "outer_model_seed": "5200+outer_fold" if args.nested_outer_z4_seeds else "6400+outer_fold",
            "outer_shuffle_seed": "5300+outer_fold" if args.nested_outer_z4_seeds else "6500+outer_fold",
            "outer_seed_contract": "exact Z4-A sensitivity" if args.nested_outer_z4_seeds else "independent Round Z5",
            "selection_tie_break": "target desc, collision harm asc, overtake loss asc, interventions asc, lambda desc, tau desc",
        },
        "frozen_hidden": model_results["frozen_hidden"],
        "history": model_results["history"],
        "grouped_fixed_baseline": fixed_summary,
        "criteria": criteria,
        "paired_frozen_vs_fixed": paired_selector_comparison(plan, frozen_predictions, fixed_predictions, branch0_results, candidate_results, names),
        "paired_history_vs_frozen": paired_selector_comparison(plan, history_predictions, frozen_predictions, branch0_results, candidate_results, names),
        "execution": {
            "device": args.device,
            "u44_model_sha256": sha256_file(args.u44_model_path),
            "command": " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv]),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip(),
            "worktree_status": subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.splitlines(),
        },
    }


if __name__ == "__main__":
    args = parse_arguments()
    if args.nested_outer_z4_seeds and not args.nested_operating_point:
        raise ValueError("--nested-outer-z4-seeds requires --nested-operating-point")
    if args.output_report.exists():
        raise FileExistsError(f"refusing to overwrite completed report: {args.output_report}")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Round Z4 requires CUDA")
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    branch0 = json.loads(args.branch0_results.read_text(encoding="utf-8"))
    candidates = json.loads(args.candidate_results.read_text(encoding="utf-8"))
    candidate_names, stratum_counts, fold_startpoint_counts = validate_inputs(plan, branch0, candidates, args)
    histories, hidden, input_rows = load_actor_features(plan, args, device)
    names, labels = build_labels(plan, branch0["episodes"], candidates["episodes"], candidate_names)
    actions = action_tensor(plan, device)

    if args.nested_operating_point:
        report = run_nested_operating_point(plan, names, labels, histories, hidden, actions, branch0["episodes"], candidates["episodes"], args, device)
        report["quality_validation"] = {
            "scenario_count": len(plan["tasks"]),
            "action_count": len(names),
            "labeled_scenario_action_count": int(labels.size),
            "actor_history_row_count": input_rows,
            "hidden_shape": list(hidden.shape),
            "history_shape": list(histories.shape),
            "all_inputs_finite": bool(np.isfinite(hidden).all() and np.isfinite(histories).all()),
            "fold_startpoint_counts": fold_startpoint_counts,
            "outer_fold_startpoints_disjoint": True,
            "stratum_counts": stratum_counts,
        }
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.output_report, report)
        print(json.dumps({
            "verdict": report["verdict"],
            "criteria": report["criteria"],
            "frozen_hidden": {key: report["frozen_hidden"]["nested_summary"][key] for key in ("success_counts", "target_success_total", "safe_control_new_collision_count", "safe_control_overtake_loss_count")},
            "history": {key: report["history"]["nested_summary"][key] for key in ("success_counts", "target_success_total", "safe_control_new_collision_count", "safe_control_overtake_loss_count")},
            "grouped_fixed_baseline": {key: report["grouped_fixed_baseline"][key] for key in ("success_counts", "target_success_total", "safe_control_new_collision_count", "safe_control_overtake_loss_count")},
            "paired_frozen_vs_fixed": report["paired_frozen_vs_fixed"],
        }, indent=2, sort_keys=True))
        sys.exit(0)

    control_predictions = np.full(len(plan["tasks"]), -1, dtype=np.int64)
    treatment_predictions = np.full(len(plan["tasks"]), -1, dtype=np.int64)
    fixed_predictions = np.full(len(plan["tasks"]), -1, dtype=np.int64)
    control_class_predictions = np.full(labels.shape, -1, dtype=np.int64)
    treatment_class_predictions = np.full(labels.shape, -1, dtype=np.int64)
    folds = []
    for fold in range(5):
        train_indices = np.asarray([index for index, task in enumerate(plan["tasks"]) if int(task["fold"]) != fold], dtype=np.int64)
        test_indices = np.asarray([index for index, task in enumerate(plan["tasks"]) if int(task["fold"]) == fold], dtype=np.int64)
        if len(train_indices) == 0 or len(test_indices) == 0:
            raise RuntimeError(f"empty grouped fold {fold}")
        hidden_mean = hidden[train_indices].mean(axis=0)
        hidden_std = hidden[train_indices].std(axis=0)
        hidden_std[hidden_std < 1e-6] = 1.0
        feature_mean = histories[train_indices].reshape(-1, 420).mean(axis=0)
        feature_std = histories[train_indices].reshape(-1, 420).std(axis=0)
        feature_std[feature_std < 1e-6] = 1.0
        train_hidden = torch.tensor((hidden[train_indices] - hidden_mean) / hidden_std, dtype=torch.float32, device=device)
        test_hidden = torch.tensor((hidden[test_indices] - hidden_mean) / hidden_std, dtype=torch.float32, device=device)
        train_history = torch.tensor((histories[train_indices] - feature_mean) / feature_std, dtype=torch.float32, device=device)
        test_history = torch.tensor((histories[test_indices] - feature_mean) / feature_std, dtype=torch.float32, device=device)
        train_labels = torch.tensor(labels[train_indices], dtype=torch.long, device=device)
        class_counts = np.bincount(labels[train_indices].reshape(-1), minlength=3)
        if bool(np.any(class_counts == 0)):
            raise RuntimeError(f"fold {fold} has an empty outcome class")
        class_weights = torch.tensor(len(train_indices) * len(names) / (3.0 * class_counts), dtype=torch.float32, device=device)

        control = FrozenHiddenOutcomeModel().to(device)
        control_loss = train_model(control, train_hidden, None, train_labels, actions, class_weights, 5200 + fold, 5300 + fold)
        with torch.no_grad():
            control_probabilities = torch.softmax(control(test_hidden, actions), dim=2).cpu().numpy()
        control_predictions[test_indices] = choose_actions(control_probabilities)
        control_class_predictions[test_indices] = control_probabilities.argmax(axis=2)

        treatment = HistoryOutcomeModel().to(device)
        treatment_loss = train_model(treatment, train_hidden, train_history, train_labels, actions, class_weights, 5200 + fold, 5300 + fold)
        with torch.no_grad():
            treatment_probabilities = torch.softmax(treatment(test_hidden, test_history, actions), dim=2).cpu().numpy()
        treatment_predictions[test_indices] = choose_actions(treatment_probabilities)
        treatment_class_predictions[test_indices] = treatment_probabilities.argmax(axis=2)

        fixed_index = fixed_action_for_fold(plan, names, train_indices, branch0["episodes"], candidates["episodes"])
        fixed_predictions[test_indices] = fixed_index
        folds.append({
            "fold": fold,
            "train_scenario_count": int(len(train_indices)),
            "test_scenario_count": int(len(test_indices)),
            "train_startpoint_count": len({int(plan["tasks"][index]["scenario"]["ego_idx"]) for index in train_indices}),
            "test_startpoint_count": len({int(plan["tasks"][index]["scenario"]["ego_idx"]) for index in test_indices}),
            "class_counts": {CLASS_NAMES[index]: int(class_counts[index]) for index in range(3)},
            "class_weights": {CLASS_NAMES[index]: float(class_weights[index].cpu()) for index in range(3)},
            "fixed_baseline_action": names[fixed_index],
            "control_final_train_loss": control_loss,
            "treatment_final_train_loss": treatment_loss,
        })
        del control, treatment, train_hidden, test_hidden, train_history, test_history, train_labels
        torch.cuda.empty_cache()

    if bool(np.any(control_predictions < 0)) or bool(np.any(treatment_predictions < 0)) or bool(np.any(fixed_predictions < 0)):
        raise RuntimeError("out-of-fold action predictions are incomplete")
    if bool(np.any(control_class_predictions < 0)) or bool(np.any(treatment_class_predictions < 0)):
        raise RuntimeError("out-of-fold outcome predictions are incomplete")

    control_summary = summarize(plan, names, control_predictions, branch0["episodes"], candidates["episodes"])
    treatment_summary = summarize(plan, names, treatment_predictions, branch0["episodes"], candidates["episodes"])
    fixed_summary = summarize(plan, names, fixed_predictions, branch0["episodes"], candidates["episodes"])
    criteria = {
        "inherited_collision_overtake_rescue": treatment_summary["success_counts"]["inherited_collision"] >= 11,
        "created_collision_overtake_rescue": treatment_summary["success_counts"]["created_collision"] >= 7,
        "lost_overtake_restore": treatment_summary["success_counts"]["lost_overtake"] >= 4,
        "safe_control_new_collision": treatment_summary["safe_control_new_collision_count"] <= 4,
        "safe_control_overtake_loss": treatment_summary["safe_control_overtake_loss_count"] <= 11,
        "target_margin_over_fixed": treatment_summary["target_success_total"] - fixed_summary["target_success_total"] >= 9,
        "target_margin_over_frozen_hidden": treatment_summary["target_success_total"] - control_summary["target_success_total"] >= 9,
    }
    label_counts = np.bincount(labels.reshape(-1), minlength=3)
    report = {
        "schema_version": 1,
        "experiment_id": "action_response_representation_gate",
        "gate": "round_z4_a_development_grouped_oof",
        "verdict": "pass_to_independent_validation" if all(criteria.values()) else "fail_close_tested_representation_instance",
        "quality_validation": {
            "scenario_count": len(plan["tasks"]),
            "action_count": len(names),
            "labeled_scenario_action_count": int(labels.size),
            "actor_history_steps_per_scenario": 50,
            "actor_feature_count_per_step": 420,
            "actor_history_row_count": input_rows,
            "hidden_shape": list(hidden.shape),
            "history_shape": list(histories.shape),
            "all_inputs_finite": bool(np.isfinite(hidden).all() and np.isfinite(histories).all()),
            "fold_startpoint_counts": fold_startpoint_counts,
            "fold_startpoints_disjoint": True,
            "input_fields": ["ego_lidar_360", "previous_ego_measured_speed_mps"],
            "forbidden_fields_used": [],
            "stratum_counts": stratum_counts,
            "outcome_label_counts": {CLASS_NAMES[index]: int(label_counts[index]) for index in range(3)},
        },
        "model_contract": {
            "control": "hidden1680-linear192-relu-action2-linear64-relu-class3",
            "treatment": "hidden1680-linear128-relu-plus-history50x420-linear64-relu-gru64-action2-linear64-relu-class3",
            "selection_score": "P(overtake)-5*P(collision)",
            "epochs": 100,
            "batch_size_scenarios": 64,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "fold_seed": "5200+fold",
            "shuffle_seed": "5300+fold",
            "normalization": "training-fold mean/std; std below 1e-6 replaced by 1",
        },
        "out_of_fold_outcome_accuracy": {
            "frozen_hidden_control": float(np.mean(control_class_predictions == labels)),
            "history_treatment": float(np.mean(treatment_class_predictions == labels)),
        },
        "frozen_hidden_control": control_summary,
        "history_treatment": treatment_summary,
        "grouped_fixed_baseline": fixed_summary,
        "folds": folds,
        "criteria": criteria,
        "execution": {
            "device": args.device,
            "u44_model_sha256": sha256_file(args.u44_model_path),
            "command": " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv]),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip(),
            "worktree_status": subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.splitlines(),
        },
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_report, report)
    print(json.dumps({
        "verdict": report["verdict"],
        "criteria": criteria,
        "out_of_fold_outcome_accuracy": report["out_of_fold_outcome_accuracy"],
        "frozen_hidden_control": {key: control_summary[key] for key in ("success_counts", "target_success_total", "safe_control_new_collision_count", "safe_control_overtake_loss_count")},
        "history_treatment": {key: treatment_summary[key] for key in ("success_counts", "target_success_total", "safe_control_new_collision_count", "safe_control_overtake_loss_count")},
        "grouped_fixed_baseline": {key: fixed_summary[key] for key in ("success_counts", "target_success_total", "safe_control_new_collision_count", "safe_control_overtake_loss_count")},
    }, indent=2, sort_keys=True))
