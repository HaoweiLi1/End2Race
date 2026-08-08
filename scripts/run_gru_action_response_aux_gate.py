import argparse
import copy
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
from scripts.run_action_response_representation_gate import TARGET_STRATA, paired_selector_comparison, selected_record, subset_metrics, summarize, validate_inputs
from utils import atomic_write_json, load_positions_and_speeds_from_params

SEED_BASES = (7100, 8100)
LAMBDA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


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
    parser.add_argument("--epochs", type=int, default=10)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_fold(test_fold):
    return (int(test_fold) + 1) % 5


def load_recurrent_inputs(plan, args, actor, device):
    tasks = plan["tasks"]
    scenario_count = len(tasks)
    start_indices = np.asarray([int(task["prefixes"]["late"]["start_index"]) for task in tasks], dtype=np.int64)
    burnin_ends = start_indices - 50
    maximum_burnin_end = int(burnin_ends.max())
    window_lidar = np.empty((scenario_count, 50, 360), dtype=np.float32)
    window_speed = np.empty((scenario_count, 50, 1), dtype=np.float32)
    initial_hidden = np.empty((scenario_count, actor.gru.hidden_size), dtype=np.float32)
    frozen_hidden = np.empty((scenario_count, actor.gru.hidden_size), dtype=np.float32)
    source_action = np.empty((scenario_count, 2), dtype=np.float32)
    source_trace_hashes = {}
    for index, task in enumerate(tasks):
        key = task["episode_key"]
        trace_path = args.u44_trace_root / f"{key}.npz"
        hidden_path = args.hidden_root / f"{key}.npz"
        if not trace_path.is_file() or not hidden_path.is_file():
            raise RuntimeError(f"missing recurrent input for {key}")
        with np.load(trace_path, allow_pickle=False) as payload:
            lidar = np.asarray(payload["ego_lidar_360"], dtype=np.float32)
            measured_speed = np.asarray(payload["ego_measured_speed_mps"], dtype=np.float32)
            raw_action = np.asarray(payload["ego_raw_action"], dtype=np.float32)
        start_index = int(start_indices[index])
        burnin_end = int(burnin_ends[index])
        if lidar.ndim != 2 or lidar.shape[1] != 360 or len(measured_speed) != len(lidar) or raw_action.shape != (len(lidar), 2) or start_index >= len(lidar):
            raise RuntimeError(f"invalid source trace arrays for {key}")
        params = {"ego_raceline": "raceline1", "opp_raceline": task["scenario"]["opp_raceline"], "ego_idx": task["scenario"]["ego_idx"], "opp_idx": task["scenario"]["opp_idx"]}
        _, initial_speeds = load_positions_and_speeds_from_params(params, "Austin")
        previous_speed = np.empty(start_index + 1, dtype=np.float32)
        previous_speed[0] = float(initial_speeds[0] * 0.9)
        previous_speed[1:] = measured_speed[:start_index]
        window_lidar[index] = lidar[burnin_end + 1:start_index + 1]
        window_speed[index, :, 0] = previous_speed[burnin_end + 1:start_index + 1]
        source_action[index] = raw_action[start_index]
        with np.load(hidden_path, allow_pickle=False) as payload:
            frozen_hidden[index] = np.asarray(payload["late"], dtype=np.float32)
        source_trace_hashes[key] = sha256_file(trace_path)
        with torch.no_grad():
            hidden = torch.zeros((1, 1, actor.gru.hidden_size), dtype=torch.float32, device=device)
            lidar_tensor = torch.tensor(lidar[:burnin_end + 1], dtype=torch.float32, device=device).unsqueeze(0)
            speed_tensor = torch.tensor(previous_speed[:burnin_end + 1], dtype=torch.float32, device=device).reshape(1, burnin_end + 1, 1)
            for step in range(burnin_end + 1):
                _, hidden = actor(lidar_tensor[:, step:step + 1], speed_tensor[:, step:step + 1], hidden)
            initial_hidden[index] = hidden[0, 0].cpu().numpy()
    if not bool(np.isfinite(initial_hidden).all() and np.isfinite(window_lidar).all() and np.isfinite(window_speed).all() and np.isfinite(frozen_hidden).all() and np.isfinite(source_action).all()):
        raise RuntimeError("non-finite recurrent inputs")

    with torch.no_grad():
        feature_batches = []
        for start in range(0, scenario_count, 64):
            lidar_tensor = torch.tensor(window_lidar[start:start + 64], dtype=torch.float32, device=device)
            speed_tensor = torch.tensor(window_speed[start:start + 64], dtype=torch.float32, device=device)
            processed_lidar = (-1 / (1 + torch.exp(-actor.k * lidar_tensor)) + 1) * 2
            speed_embedding = actor.speed_mlp(speed_tensor)
            feature_batches.append(torch.cat((processed_lidar, speed_embedding), dim=2).cpu().numpy())
        window_features = np.concatenate(feature_batches, axis=0).astype(np.float32)
        feature_tensor = torch.tensor(window_features, dtype=torch.float32, device=device)
        reconstructed_hidden = np.empty_like(frozen_hidden)
        reconstructed_action = np.empty_like(source_action)
        for index in range(scenario_count):
            replay_hidden = torch.tensor(initial_hidden[index], dtype=torch.float32, device=device).reshape(1, 1, -1)
            for step in range(50):
                _, replay_hidden = actor.gru(feature_tensor[index:index + 1, step:step + 1], replay_hidden)
            reconstructed_hidden[index] = replay_hidden[0, 0].cpu().numpy()
            reconstructed_action[index] = actor.output_layer(replay_hidden[0, 0]).cpu().numpy()
    hidden_max_error = float(np.max(np.abs(reconstructed_hidden.astype(np.float64) - frozen_hidden.astype(np.float64))))
    action_max_error = float(np.max(np.abs(reconstructed_action.astype(np.float64) - source_action.astype(np.float64))))
    if hidden_max_error > 1e-5 or action_max_error > 1e-5:
        raise RuntimeError(f"50-step recurrent reconstruction failed: hidden={hidden_max_error}, action={action_max_error}")
    return window_features, initial_hidden, frozen_hidden, source_action, {
        "scenario_count": scenario_count,
        "history_steps": 50,
        "feature_shape": list(window_features.shape),
        "initial_hidden_shape": list(initial_hidden.shape),
        "frozen_hidden_shape": list(frozen_hidden.shape),
        "maximum_burnin_step": maximum_burnin_end,
        "reconstructed_hidden_max_abs_error": hidden_max_error,
        "reconstructed_action_max_abs_error": action_max_error,
        "source_trace_count": len(source_trace_hashes),
        "all_inputs_finite": True,
    }


def build_targets(plan, branch0_results, candidate_results):
    names = ["noop"] + [candidate["name"] for candidate in plan["candidate_contract"]["candidates"]]
    collision = np.empty((len(plan["tasks"]), len(names)), dtype=np.float32)
    progress_delta = np.empty((len(plan["tasks"]), len(names)), dtype=np.float32)
    for task_index, task in enumerate(plan["tasks"]):
        noop_progress = float(branch0_results[task["episode_key"]]["final_relative_position_m"])
        for action_index, action_name in enumerate(names):
            record = selected_record(task, action_name, branch0_results, candidate_results)
            collision[task_index, action_index] = float(record["outcome"] in ("ego-opp", "ego-wall"))
            progress_delta[task_index, action_index] = float(record["final_relative_position_m"] - noop_progress)
    if not bool(np.isfinite(collision).all() and np.isfinite(progress_delta).all()):
        raise RuntimeError("non-finite auxiliary targets")
    return names, collision, progress_delta


def action_values(plan):
    actions = [[0.0, 0.0]]
    for candidate in plan["candidate_contract"]["candidates"]:
        actions.append([float(candidate["steering_delta_rad"]) / 0.02, float(candidate["speed_delta_mps"]) / 0.5])
    return np.asarray(actions, dtype=np.float32)


class ResponseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Sequential(nn.Linear(1680, 192), nn.ReLU())
        self.response_layer = nn.Sequential(nn.Linear(194, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, hidden, actions):
        encoded = self.hidden_layer(hidden)
        hidden_batch = encoded.unsqueeze(1).expand(-1, actions.shape[0], -1)
        action_batch = actions.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        return self.response_layer(torch.cat((hidden_batch, action_batch), dim=2))


def reset_parameters(module):
    if isinstance(module, nn.Linear):
        module.reset_parameters()


def model_predictions(model_type, actor_gru, actor_output_layer, features, initial_hidden, frozen_hidden, collision_targets, progress_targets, train_indices, predict_indices, actions, device, epochs, seed):
    hidden_mean = frozen_hidden[train_indices].mean(axis=0)
    hidden_std = frozen_hidden[train_indices].std(axis=0)
    hidden_std[hidden_std < 1e-6] = 1.0
    progress_mean = float(progress_targets[train_indices].mean())
    progress_std = float(progress_targets[train_indices].std())
    if progress_std < 1e-6:
        progress_std = 1.0
    positives = float(collision_targets[train_indices].sum())
    negatives = float(collision_targets[train_indices].size - positives)
    if positives <= 0 or negatives <= 0:
        raise RuntimeError("auxiliary collision target has an empty class")
    pos_weight = negatives / positives

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    head = ResponseHead().to(device)
    head.apply(reset_parameters)
    initial_head_state = copy.deepcopy(head.state_dict())
    if model_type == "trainable_gru":
        gru = copy.deepcopy(actor_gru).to(device)
        gru.train()
        optimizer = optim.Adam([{"params": gru.parameters(), "lr": 3e-6}, {"params": head.parameters(), "lr": 3e-4}], weight_decay=1e-4)
    elif model_type == "frozen_hidden":
        gru = None
        optimizer = optim.Adam(head.parameters(), lr=3e-4, weight_decay=1e-4)
    else:
        raise ValueError(f"unknown model type {model_type}")
    head.load_state_dict(initial_head_state)

    feature_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    initial_tensor = torch.tensor(initial_hidden, dtype=torch.float32, device=device)
    frozen_tensor = torch.tensor(frozen_hidden, dtype=torch.float32, device=device)
    collision_tensor = torch.tensor(collision_targets, dtype=torch.float32, device=device)
    progress_tensor = torch.tensor((progress_targets - progress_mean) / progress_std, dtype=torch.float32, device=device)
    action_tensor = torch.tensor(actions, dtype=torch.float32, device=device)
    mean_tensor = torch.tensor(hidden_mean, dtype=torch.float32, device=device)
    std_tensor = torch.tensor(hidden_std, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 100)
    train_indices_tensor = torch.tensor(train_indices, dtype=torch.long)
    final_loss = None
    for _ in range(epochs):
        order = train_indices_tensor[torch.randperm(len(train_indices_tensor), generator=generator)]
        losses = []
        for start in range(0, len(order), 64):
            batch = order[start:start + 64].to(device)
            if gru is None:
                hidden = frozen_tensor[batch]
            else:
                _, next_hidden = gru(feature_tensor[batch], initial_tensor[batch].unsqueeze(0))
                hidden = next_hidden[0]
            response = head((hidden - mean_tensor) / std_tensor, action_tensor)
            collision_loss = nn.functional.binary_cross_entropy_with_logits(response[:, :, 0], collision_tensor[batch], pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
            progress_loss = nn.functional.smooth_l1_loss(response[:, :, 1], progress_tensor[batch])
            loss = 0.5 * collision_loss + 0.5 * progress_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(head.parameters()) + ([] if gru is None else list(gru.parameters())), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses))

    predict_tensor = torch.tensor(predict_indices, dtype=torch.long, device=device)
    with torch.no_grad():
        if gru is None:
            predicted_hidden = frozen_tensor[predict_tensor]
        else:
            _, next_hidden = gru(feature_tensor[predict_tensor], initial_tensor[predict_tensor].unsqueeze(0))
            predicted_hidden = next_hidden[0]
        response = head((predicted_hidden - mean_tensor) / std_tensor, action_tensor)
        collision_probability = torch.sigmoid(response[:, :, 0]).cpu().numpy()
        progress_prediction = (response[:, :, 1] * progress_std + progress_mean).cpu().numpy()
        hidden_reference = frozen_tensor[predict_tensor]
        hidden_relative_l2 = torch.linalg.vector_norm(predicted_hidden - hidden_reference, dim=1) / torch.clamp(torch.linalg.vector_norm(hidden_reference, dim=1), min=1e-12)
        actor_action = actor_output_layer(predicted_hidden)
        reference_action = actor_output_layer(hidden_reference)
        action_abs_difference = torch.abs(actor_action - reference_action)
    parameter_relative_l2 = 0.0
    if gru is not None:
        numerator = sum(float(torch.sum((parameter.detach() - reference.detach()) ** 2).cpu()) for parameter, reference in zip(gru.parameters(), actor_gru.parameters()))
        denominator = sum(float(torch.sum(reference.detach() ** 2).cpu()) for reference in actor_gru.parameters())
        parameter_relative_l2 = math.sqrt(numerator / max(denominator, 1e-24))
    diagnostics = {
        "final_train_loss": final_loss,
        "progress_target_train_mean_m": progress_mean,
        "progress_target_train_std_m": progress_std,
        "collision_positive_weight": pos_weight,
        "gru_parameter_relative_l2": parameter_relative_l2,
        "test_hidden_relative_l2_mean": float(hidden_relative_l2.mean().cpu()),
        "test_hidden_relative_l2_max": float(hidden_relative_l2.max().cpu()),
        "test_actor_steering_abs_change_mean": float(action_abs_difference[:, 0].mean().cpu()),
        "test_actor_steering_abs_change_max": float(action_abs_difference[:, 0].max().cpu()),
        "test_actor_speed_abs_change_mean": float(action_abs_difference[:, 1].mean().cpu()),
        "test_actor_speed_abs_change_max": float(action_abs_difference[:, 1].max().cpu()),
    }
    del head, gru, feature_tensor, initial_tensor, frozen_tensor, collision_tensor, progress_tensor, action_tensor
    torch.cuda.empty_cache()
    return collision_probability, progress_prediction, diagnostics


def operating_point_predictions(collision_probability, progress_prediction, lambda_cost, tau):
    scores = progress_prediction - float(lambda_cost) * collision_probability
    best_nonnoop = scores[:, 1:].argmax(axis=1) + 1
    rows = np.arange(len(scores))
    margins = scores[rows, best_nonnoop] - scores[:, 0]
    selected = np.where(margins >= float(tau), best_nonnoop, 0)
    return selected.astype(np.int64), margins


def choose_operating_point(plan, names, indices, collision_probability, progress_prediction, branch0_results, candidate_results):
    candidates = []
    for lambda_cost in LAMBDA_GRID:
        _, margins = operating_point_predictions(collision_probability, progress_prediction, lambda_cost, -float("inf"))
        thresholds = sorted(set(float(value) for value in margins)) + [float("inf")]
        for tau in thresholds:
            selected, _ = operating_point_predictions(collision_probability, progress_prediction, lambda_cost, tau)
            metrics = subset_metrics(plan, names, indices, selected, branch0_results, candidate_results)
            collision_limit = math.floor(0.05 * metrics["safe_control_count"])
            overtake_limit = math.floor(0.05 * metrics["safe_control_count"])
            if metrics["safe_control_new_collision_count"] <= collision_limit and metrics["safe_control_overtake_loss_count"] <= overtake_limit:
                candidates.append((metrics["target_success_total"], -metrics["safe_control_new_collision_count"], -metrics["safe_control_overtake_loss_count"], -metrics["intervention_count"], lambda_cost, tau, metrics))
    if not candidates:
        raise RuntimeError("noop operating point was not feasible")
    chosen = max(candidates, key=lambda row: row[:6])
    return float(chosen[4]), float(chosen[5]), chosen[6], len(candidates)


def run_seed(seed_base, plan, names, actor, features, initial_hidden, frozen_hidden, collision_targets, progress_targets, actions, branch0_results, candidate_results, args, device):
    model_predictions_by_name = {}
    model_fold_records = {}
    for model_name in ("frozen_hidden", "trainable_gru"):
        selected_all = np.full(len(plan["tasks"]), -1, dtype=np.int64)
        folds = []
        for test_fold in range(5):
            calibration = calibration_fold(test_fold)
            train_indices = np.asarray([index for index, task in enumerate(plan["tasks"]) if int(task["fold"]) not in (test_fold, calibration)], dtype=np.int64)
            calibration_indices = np.asarray([index for index, task in enumerate(plan["tasks"]) if int(task["fold"]) == calibration], dtype=np.int64)
            test_indices = np.asarray([index for index, task in enumerate(plan["tasks"]) if int(task["fold"]) == test_fold], dtype=np.int64)
            prediction_indices = np.concatenate((calibration_indices, test_indices))
            collision_probability, progress_prediction, diagnostics = model_predictions(model_name, actor.gru, actor.output_layer, features, initial_hidden, frozen_hidden, collision_targets, progress_targets, train_indices, prediction_indices, actions, device, args.epochs, seed_base + test_fold)
            calibration_count = len(calibration_indices)
            lambda_cost, tau, calibration_metrics, feasible_count = choose_operating_point(plan, names, calibration_indices, collision_probability[:calibration_count], progress_prediction[:calibration_count], branch0_results, candidate_results)
            selected_test, margins = operating_point_predictions(collision_probability[calibration_count:], progress_prediction[calibration_count:], lambda_cost, tau)
            selected_all[test_indices] = selected_test
            folds.append({
                "test_fold": test_fold,
                "calibration_fold": calibration,
                "train_scenario_count": len(train_indices),
                "calibration_scenario_count": len(calibration_indices),
                "test_scenario_count": len(test_indices),
                "train_startpoint_count": len({int(plan["tasks"][index]["scenario"]["ego_idx"]) for index in train_indices}),
                "calibration_startpoint_count": len({int(plan["tasks"][index]["scenario"]["ego_idx"]) for index in calibration_indices}),
                "test_startpoint_count": len({int(plan["tasks"][index]["scenario"]["ego_idx"]) for index in test_indices}),
                "selected_lambda": lambda_cost,
                "selected_tau": tau if math.isfinite(tau) else "infinity",
                "feasible_calibration_operating_point_count": feasible_count,
                "calibration_metrics": calibration_metrics,
                "test_margin_minimum": float(margins.min()),
                "test_margin_maximum": float(margins.max()),
                "training_diagnostics": diagnostics,
            })
        if bool(np.any(selected_all < 0)):
            raise RuntimeError(f"incomplete OOF predictions for {model_name}")
        model_predictions_by_name[model_name] = selected_all
        model_fold_records[model_name] = folds
    frozen_summary = summarize(plan, names, model_predictions_by_name["frozen_hidden"], branch0_results, candidate_results)
    treatment_summary = summarize(plan, names, model_predictions_by_name["trainable_gru"], branch0_results, candidate_results)
    changed = all(fold["training_diagnostics"]["gru_parameter_relative_l2"] >= 1e-7 and fold["training_diagnostics"]["test_hidden_relative_l2_mean"] >= 1e-5 for fold in model_fold_records["trainable_gru"])
    criteria = {
        "trainable_gru_changed_representation_all_folds": changed,
        "treatment_safe_control_new_collision_at_most_5": treatment_summary["safe_control_new_collision_count"] <= 5,
        "treatment_safe_control_overtake_loss_at_most_5": treatment_summary["safe_control_overtake_loss_count"] <= 5,
        "treatment_target_at_least_88": treatment_summary["target_success_total"] >= 88,
        "treatment_target_margin_over_frozen_at_least_9": treatment_summary["target_success_total"] - frozen_summary["target_success_total"] >= 9,
        "treatment_inherited_collision_at_least_11": treatment_summary["success_counts"]["inherited_collision"] >= 11,
        "treatment_created_collision_at_least_7": treatment_summary["success_counts"]["created_collision"] >= 7,
        "treatment_lost_overtake_at_least_4": treatment_summary["success_counts"]["lost_overtake"] >= 4,
    }
    return {
        "seed_base": seed_base,
        "verdict": "pass_to_independent_validation" if all(criteria.values()) else "fail_close_tested_representation_only_instance",
        "criteria": criteria,
        "frozen_hidden": {"summary": frozen_summary, "folds": model_fold_records["frozen_hidden"]},
        "trainable_gru": {"summary": treatment_summary, "folds": model_fold_records["trainable_gru"]},
        "paired_trainable_gru_vs_frozen": paired_selector_comparison(plan, model_predictions_by_name["trainable_gru"], model_predictions_by_name["frozen_hidden"], branch0_results, candidate_results, names),
    }


if __name__ == "__main__":
    args = parse_arguments()
    if args.output_report.exists():
        raise FileExistsError(f"refusing to overwrite completed report: {args.output_report}")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("representation-changing Gate requires CUDA")
    if args.epochs != 10:
        raise ValueError("the preregistered Gate requires exactly 10 epochs")
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    branch0 = json.loads(args.branch0_results.read_text(encoding="utf-8"))
    candidates = json.loads(args.candidate_results.read_text(encoding="utf-8"))
    candidate_names, stratum_counts, fold_startpoint_counts = validate_inputs(plan, branch0, candidates, args)
    actor = End2Race(mask_prob=0.0, hidden_scale=4).to(device)
    actor.load_state_dict(torch.load(args.u44_model_path, map_location=device, weights_only=True), strict=True)
    actor.eval()
    features, initial_hidden, frozen_hidden, source_action, recurrent_quality = load_recurrent_inputs(plan, args, actor, device)
    names, collision_targets, progress_targets = build_targets(plan, branch0["episodes"], candidates["episodes"])
    if names != ["noop"] + candidate_names:
        raise RuntimeError("action order changed")
    actions = action_values(plan)

    seed_reports = []
    for seed_base in SEED_BASES:
        seed_reports.append(run_seed(seed_base, plan, names, actor, features, initial_hidden, frozen_hidden, collision_targets, progress_targets, actions, branch0["episodes"], candidates["episodes"], args, device))
    all_seeds_pass = all(report["verdict"] == "pass_to_independent_validation" for report in seed_reports)
    report = {
        "schema_version": 1,
        "experiment_id": "gru_action_response_auxiliary_representation_gate",
        "gate": "round_z8_rotating_grouped_train_calibration_test",
        "verdict": "pass_to_independent_validation" if all_seeds_pass else "fail_close_tested_representation_only_instance",
        "representation_only_2b_class_refuted": False,
        "tested_instance_pass": all_seeds_pass,
        "quality_validation": {
            "scenario_count": len(plan["tasks"]),
            "action_count": len(names),
            "labeled_scenario_action_count": int(collision_targets.size),
            "stratum_counts": stratum_counts,
            "fold_startpoint_counts": fold_startpoint_counts,
            "fold_startpoints_disjoint": True,
            "recurrent_input": recurrent_quality,
            "collision_label_count": int(collision_targets.sum()),
            "progress_delta_minimum_m": float(progress_targets.min()),
            "progress_delta_maximum_m": float(progress_targets.max()),
            "progress_noop_max_abs_m": float(np.max(np.abs(progress_targets[:, 0]))),
        },
        "method_contract": {
            "auxiliary_targets": ["ego_collision_indicator", "final_relative_progress_delta_vs_noop_m"],
            "auxiliary_loss": "0.5*class-balanced BCE + 0.5*SmoothL1 on train-split standardized progress delta",
            "actor_visible_sequence": "50 actual U44 GRU input rows ending at late intervention start; exact stepwise U44 burn-in state detached",
            "treatment": "U44-initialized trainable GRU plus response head; k, speed_mlp and actor output_layer frozen",
            "control": "same response head on frozen U44 hidden",
            "gru_learning_rate": 3e-6,
            "head_learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "gradient_norm_clip": 1.0,
            "epochs": args.epochs,
            "batch_size_scenarios": 64,
            "seed_bases": list(SEED_BASES),
            "split": "for test fold f, calibration=(f+1)%5 and remaining three folds train; each startpoint is test exactly once",
            "lambda_grid": list(LAMBDA_GRID),
            "tau_grid": "all calibration best-nonnoop minus noop margins plus infinity",
            "calibration_harm_budget": "floor(0.05*N_safe_controls) separately for collision and overtake loss",
            "selection_tie_break": "target desc, collision harm asc, overtake loss asc, interventions asc, lambda desc, tau desc",
            "head_deleted_after_training": True,
        },
        "seed_reports": seed_reports,
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
        "recurrent_quality": recurrent_quality,
        "seeds": [{"seed_base": item["seed_base"], "verdict": item["verdict"], "criteria": item["criteria"], "frozen_hidden": {key: item["frozen_hidden"]["summary"][key] for key in ("success_counts", "target_success_total", "safe_control_new_collision_count", "safe_control_overtake_loss_count")}, "trainable_gru": {key: item["trainable_gru"]["summary"][key] for key in ("success_counts", "target_success_total", "safe_control_new_collision_count", "safe_control_overtake_loss_count")}, "paired": item["paired_trainable_gru_vs_frozen"]} for item in seed_reports],
    }, indent=2, sort_keys=True))
