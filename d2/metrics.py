"""Frame- and episode-level metrics for D2 probes."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _probability(value, name: str = "probability") -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must lie in [0, 1]")
    return array


def _boolean(value, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.ndim != 1 or (length is not None and len(array) != length):
        raise ValueError(f"{name} has invalid shape")
    return array


def average_precision(target, probability) -> float:
    y = _boolean(target, "target")
    p = _probability(probability)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("target/probability length mismatch")
    positives = int(np.count_nonzero(y))
    if positives == 0:
        return 0.0
    order = np.argsort(-p, kind="mergesort")
    sorted_p = p[order]
    sorted_y = y[order].astype(np.int64)
    cumulative_true = np.cumsum(sorted_y)
    cumulative_total = np.arange(1, len(y) + 1)
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_p) != 0), len(y) - 1]
    true_at = cumulative_true[group_ends]
    precision = true_at / cumulative_total[group_ends]
    recall = true_at / positives
    recall_delta = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_delta * precision))


def binary_metrics(
    target,
    probability,
    prevalence_reference: float,
    bins: int = 10,
) -> dict:
    y = _boolean(target, "target")
    p = _probability(probability)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("target/probability length mismatch")
    prevalence = float(prevalence_reference)
    if not math.isfinite(prevalence) or not 0.0 <= prevalence <= 1.0:
        raise ValueError("prevalence reference must lie in [0, 1]")
    if bins <= 0:
        raise ValueError("bins must be positive")
    numeric_y = y.astype(np.float64)
    brier = float(np.mean((p - numeric_y) ** 2))
    reference_brier = float(np.mean((prevalence - numeric_y) ** 2))
    brier_skill = float(1.0 - brier / reference_brier) if reference_brier > 0.0 else None

    bin_index = np.minimum((p * bins).astype(int), bins - 1)
    reliability = []
    ece = 0.0
    for index in range(bins):
        selected = bin_index == index
        count = int(np.count_nonzero(selected))
        mean_probability = float(np.mean(p[selected])) if count else None
        observed = float(np.mean(numeric_y[selected])) if count else None
        if count:
            ece += count / len(y) * abs(mean_probability - observed)
        reliability.append(
            {
                "bin": index,
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": count,
                "mean_probability": mean_probability,
                "observed_rate": observed,
            }
        )
    return {
        "count": len(y),
        "positive_count": int(np.count_nonzero(y)),
        "prevalence": float(np.mean(numeric_y)),
        "aucpr": average_precision(y, p),
        "brier": brier,
        "reference_brier": reference_brier,
        "brier_skill": brier_skill,
        "ece": float(ece),
        "reliability": reliability,
    }


def _alarm_inputs(
    probability,
    valid,
    positive_window,
    episode_index,
    episode_any_collision,
    episode_ego_collision,
    time,
    final_time,
):
    p = _probability(probability)
    n = len(p)
    valid = _boolean(valid, "valid", n)
    positive = _boolean(positive_window, "positive_window", n)
    episode_index = np.asarray(episode_index, dtype=np.int64)
    time = np.asarray(time, dtype=np.float64)
    if episode_index.shape != (n,) or time.shape != (n,) or not np.all(np.isfinite(time)):
        raise ValueError("frame episode/time arrays have invalid shapes")
    any_collision = _boolean(episode_any_collision, "episode_any_collision")
    ego_collision = _boolean(episode_ego_collision, "episode_ego_collision", len(any_collision))
    final_time = np.asarray(final_time, dtype=np.float64)
    if final_time.shape != (len(any_collision),) or not np.all(np.isfinite(final_time)):
        raise ValueError("episode final_time has invalid shape")
    if n and (np.min(episode_index) < 0 or np.max(episode_index) >= len(any_collision)):
        raise ValueError("frame episode index is out of range")
    return p, valid, positive, episode_index, any_collision, ego_collision, time, final_time


def _episode_maxima(probability, mask, episode_index, episode_count) -> np.ndarray:
    maxima = np.full(episode_count, -np.inf, dtype=np.float64)
    selected = np.flatnonzero(mask)
    if len(selected):
        np.maximum.at(maxima, episode_index[selected], probability[selected])
    return maxima


def evaluate_alarm_threshold(
    probability,
    valid,
    positive_window,
    episode_index,
    episode_any_collision,
    episode_ego_collision,
    time,
    final_time,
    threshold: float,
) -> dict:
    (
        p,
        valid,
        positive,
        episode_index,
        any_collision,
        ego_collision,
        time,
        final_time,
    ) = _alarm_inputs(
        probability,
        valid,
        positive_window,
        episode_index,
        episode_any_collision,
        episode_ego_collision,
        time,
        final_time,
    )
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("alarm threshold must be finite")
    episode_count = len(any_collision)
    safe_ids = np.flatnonzero(~any_collision)
    event_ids = np.flatnonzero(ego_collision)
    if len(safe_ids) == 0 or len(event_ids) == 0:
        raise ValueError("alarm metrics require safe and ego-collision episodes")
    all_max = _episode_maxima(p, valid, episode_index, episode_count)
    event_max = _episode_maxima(p, valid & positive, episode_index, episode_count)
    safe_alarm = all_max[safe_ids] >= threshold
    event_alarm = event_max[event_ids] >= threshold

    frame_alarm = valid & (p >= threshold)
    preterminal_alarm = frame_alarm & (time < final_time[episode_index])
    earliest = np.full(episode_count, np.inf, dtype=np.float64)
    selected = np.flatnonzero(preterminal_alarm)
    if len(selected):
        np.minimum.at(earliest, episode_index[selected], time[selected])
    leads = [
        float(final_time[episode] - earliest[episode]) if math.isfinite(earliest[episode]) else float("nan")
        for episode in event_ids
    ]
    finite_leads = [value for value in leads if math.isfinite(value)]
    event_count = len(event_ids)
    result = {
        "threshold": threshold,
        "safe_episode_count": len(safe_ids),
        "safe_episode_alarm_count": int(np.count_nonzero(safe_alarm)),
        "safe_episode_false_alarm_rate": float(np.mean(safe_alarm)),
        "event_episode_count": event_count,
        "event_episode_alarm_count": int(np.count_nonzero(event_alarm)),
        "event_recall": float(np.mean(event_alarm)),
        "earliest_lead_seconds": finite_leads,
    }
    for lead in (0.5, 1.0, 2.0):
        token = str(lead).rstrip("0").rstrip(".").replace(".", "p")
        result[f"warned_at_least_{token}s"] = float(
            sum(math.isfinite(value) and value + 1e-9 >= lead for value in leads) / event_count
        )
    return result


def select_alarm_threshold(
    probability,
    valid,
    positive_window,
    episode_index,
    episode_any_collision,
    episode_ego_collision,
    time,
    final_time,
    false_alarm_limit: float = 0.10,
) -> dict:
    inputs = _alarm_inputs(
        probability,
        valid,
        positive_window,
        episode_index,
        episode_any_collision,
        episode_ego_collision,
        time,
        final_time,
    )
    p, valid, positive, episode_index, any_collision, ego_collision, time, final_time = inputs
    limit = float(false_alarm_limit)
    if not math.isfinite(limit) or not 0.0 <= limit <= 1.0:
        raise ValueError("false alarm limit must lie in [0, 1]")
    episode_count = len(any_collision)
    safe_ids = np.flatnonzero(~any_collision)
    event_ids = np.flatnonzero(ego_collision)
    if len(safe_ids) == 0 or len(event_ids) == 0:
        raise ValueError("threshold selection requires safe and ego-collision episodes")
    all_max = _episode_maxima(p, valid, episode_index, episode_count)
    event_max = _episode_maxima(p, valid & positive, episode_index, episode_count)
    critical = np.unique(
        np.concatenate(
            [
                all_max[safe_ids][np.isfinite(all_max[safe_ids])],
                event_max[event_ids][np.isfinite(event_max[event_ids])],
            ]
        )
    )
    if len(critical) == 0:
        raise ValueError("no finite episode maxima for threshold selection")
    candidates = np.unique(
        np.concatenate(
            [
                critical,
                np.nextafter(critical, np.inf),
                np.array([0.0, np.nextafter(float(np.max(critical)), np.inf)]),
            ]
        )
    )
    best = None
    for threshold in candidates:
        safe_rate = float(np.mean(all_max[safe_ids] >= threshold))
        if safe_rate > limit + 1e-15:
            continue
        recall = float(np.mean(event_max[event_ids] >= threshold))
        candidate = (recall, float(threshold), safe_rate)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise AssertionError("no threshold satisfies the false-alarm constraint")
    result = evaluate_alarm_threshold(
        p,
        valid,
        positive,
        episode_index,
        any_collision,
        ego_collision,
        time,
        final_time,
        threshold=best[1],
    )
    result["false_alarm_limit"] = limit
    return result


def ttc_mae(target_ttc, predicted_ttc, cutoff_s: float = 2.0) -> dict:
    target = np.asarray(target_ttc, dtype=np.float64)
    predicted = np.asarray(predicted_ttc, dtype=np.float64)
    if target.ndim != 1 or predicted.shape != target.shape:
        raise ValueError("TTC arrays have invalid shapes")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(predicted)):
        raise ValueError("TTC arrays contain non-finite values")
    selected = target < float(cutoff_s)
    count = int(np.count_nonzero(selected))
    if count == 0:
        return {"count": 0, "mae": None, "cutoff_s": float(cutoff_s)}
    return {
        "count": count,
        "mae": float(np.mean(np.abs(predicted[selected] - target[selected]))),
        "cutoff_s": float(cutoff_s),
    }
