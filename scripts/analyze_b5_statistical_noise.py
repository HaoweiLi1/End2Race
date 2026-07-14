#!/usr/bin/env python3
"""Qualify B5-A paired changes on the opened Austin development panel.

This is a read-only post-hoc analysis.  It preserves the historical B5-A
verdict while quantifying occurrence-level churn, startpoint clustering, and
the extra optimism introduced by looking at three correlated snapshots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "end2race-b5-statistical-qualification-1"
VARIANTS = ("seed1_iter10", "seed1_iter20", "seed1_iter30")
BOOTSTRAP_DOMAIN = b"end2race:b5:opened-development:cluster-bootstrap:v1\0"
BOOTSTRAP_REPLICATES = 200_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty B5 statistical table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("B5 statistical table field order drift")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_bool(value: str) -> bool:
    if value not in ("True", "False"):
        raise ValueError(f"invalid paired boolean: {value!r}")
    return value == "True"


def collision(row: Mapping[str, str]) -> bool:
    value = parse_bool(row["ego_collision"]) or parse_bool(row["opp_collision"])
    if (row["outcome"] == "collision") != value:
        raise ValueError("paired collision/outcome contract drift")
    return value


def overtake(row: Mapping[str, str]) -> bool:
    return row["outcome"] == "overtaking"


def exact_binomial_tail(successes: int, trials: int) -> float:
    if not 0 <= successes <= trials:
        raise ValueError("invalid binomial tail arguments")
    return math.fsum(math.comb(trials, value) for value in range(successes, trials + 1)) / (
        2**trials
    )


def exact_mcnemar_two_sided(positive: int, negative: int) -> float:
    trials = positive + negative
    if trials == 0:
        return 1.0
    lower = min(positive, negative)
    probability = math.fsum(math.comb(trials, value) for value in range(lower + 1)) / (
        2**trials
    )
    return min(1.0, 2.0 * probability)


def sign_flip_distribution(vectors: Sequence[Sequence[int]]) -> Counter[tuple[int, ...]]:
    if not vectors:
        raise ValueError("empty startpoint effect inventory")
    width = len(vectors[0])
    distribution: Counter[tuple[int, ...]] = Counter({(0,) * width: 1})
    for vector in vectors:
        if len(vector) != width:
            raise ValueError("startpoint effect width drift")
        updated: Counter[tuple[int, ...]] = Counter()
        for state, count in distribution.items():
            updated[tuple(state[index] + vector[index] for index in range(width))] += count
            updated[tuple(state[index] - vector[index] for index in range(width))] += count
        distribution = updated
    if sum(distribution.values()) != 2 ** len(vectors):
        raise AssertionError("sign-flip state multiplicity drift")
    return distribution


def marginal_sign_flip_p(
    distribution: Mapping[tuple[int, ...], int],
    index: int,
    observed: int,
) -> tuple[float, float]:
    denominator = sum(distribution.values())
    one_sided = sum(count for state, count in distribution.items() if state[index] >= observed)
    two_sided = sum(
        count for state, count in distribution.items() if abs(state[index]) >= abs(observed)
    )
    return one_sided / denominator, two_sided / denominator


def cluster_bootstrap_interval(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, float | int]:
    if values.shape != (50,):
        raise ValueError("B5 cluster bootstrap requires exactly 50 startpoint effects")
    if replicates <= 0:
        raise ValueError("bootstrap replicate count must be positive")
    generator = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.int16)
    chunk = 10_000
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        indices = generator.integers(0, len(values), size=(stop - start, len(values)))
        samples[start:stop] = values[indices].sum(axis=1)
    return {
        "replicates": replicates,
        "seed": seed,
        "ci90_low": float(np.quantile(samples, 0.05)),
        "ci90_high": float(np.quantile(samples, 0.95)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def validate_and_index(
    rows: Sequence[Mapping[str, str]],
) -> tuple[dict[tuple[str, str], Mapping[str, str]], dict[int, tuple[str, ...]]]:
    expected_variants = ("BC", *VARIANTS)
    counts = Counter(row["variant"] for row in rows)
    if counts != Counter({variant: 600 for variant in expected_variants}):
        raise ValueError(f"paired variant inventory drift: {counts}")
    index: dict[tuple[str, str], Mapping[str, str]] = {}
    cases_by_startpoint: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        key = (row["case_id"], row["variant"])
        if key in index:
            raise ValueError("duplicate paired case/variant")
        index[key] = row
        startpoint = int(row["startpoint_ordinal"])
        if not 0 <= startpoint < 50:
            raise ValueError("paired startpoint ordinal drift")
        cases_by_startpoint[startpoint].add(row["case_id"])
        collision(row)
        if int(row["deterministic_speed_projection_count"]) != 0:
            raise ValueError("B5 statistical input contains deterministic speed projection")
    if set(cases_by_startpoint) != set(range(50)) or any(
        len(cases) != 12 for cases in cases_by_startpoint.values()
    ):
        raise ValueError("paired startpoint block inventory drift")
    cases = {case for case, variant in index if variant == "BC"}
    for variant in expected_variants:
        if {case for case, current in index if current == variant} != cases:
            raise ValueError("paired case inventory differs by variant")
    return index, {
        startpoint: tuple(sorted(cases)) for startpoint, cases in cases_by_startpoint.items()
    }


def metric_effect(
    baseline: Mapping[str, str], candidate: Mapping[str, str], metric: str
) -> int:
    if metric == "collision":
        return int(collision(baseline)) - int(collision(candidate))
    if metric == "overtake":
        return int(overtake(candidate)) - int(overtake(baseline))
    raise ValueError(f"unknown paired metric: {metric}")


def occurrence_counts(
    index: Mapping[tuple[str, str], Mapping[str, str]],
    cases: Iterable[str],
    variant: str,
    metric: str,
) -> tuple[int, int]:
    positive = negative = 0
    for case in cases:
        value = metric_effect(index[(case, "BC")], index[(case, variant)], metric)
        positive += value == 1
        negative += value == -1
    return positive, negative


def main() -> None:
    args = parse_args()
    rows = read_rows(args.paired_rows)
    index, cases_by_startpoint = validate_and_index(rows)
    all_cases = tuple(sorted(case for case, variant in index if variant == "BC"))
    paired_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    metric_summaries: dict[str, Any] = {}

    for metric in ("collision", "overtake"):
        vectors: list[tuple[int, ...]] = []
        for startpoint in range(50):
            vector = tuple(
                sum(
                    metric_effect(index[(case, "BC")], index[(case, variant)], metric)
                    for case in cases_by_startpoint[startpoint]
                )
                for variant in VARIANTS
            )
            vectors.append(vector)
            block_rows.append(
                {
                    "metric": metric,
                    "startpoint_ordinal": startpoint,
                    **{variant: vector[position] for position, variant in enumerate(VARIANTS)},
                }
            )

        distribution = sign_flip_distribution(vectors)
        observed = tuple(sum(vector[index] for vector in vectors) for index in range(len(VARIANTS)))
        denominator = sum(distribution.values())
        observed_max = max(observed)
        selection_max_p = (
            sum(count for state, count in distribution.items() if max(state) >= observed_max)
            / denominator
        )
        metric_summaries[metric] = {
            "observed_net_effect_by_variant": dict(zip(VARIANTS, observed)),
            "startpoint_count": len(vectors),
            "joint_sign_flip_state_count": len(distribution),
            "selection_aware_one_sided_max_net_effect_p": selection_max_p,
            "selection_statistic": "maximum net effect across iter10/iter20/iter30",
        }

        for position, variant in enumerate(VARIANTS):
            positive, negative = occurrence_counts(index, all_cases, variant, metric)
            if positive - negative != observed[position]:
                raise AssertionError("occurrence and block effect disagree")
            cluster_one, cluster_two = marginal_sign_flip_p(
                distribution, position, observed[position]
            )
            seed_digest = hashlib.sha256(BOOTSTRAP_DOMAIN)
            seed_digest.update(f"{metric}:{variant}".encode("ascii"))
            seed = int.from_bytes(seed_digest.digest()[:8], "big")
            bootstrap = cluster_bootstrap_interval(
                np.asarray([vector[position] for vector in vectors], dtype=np.int16),
                seed=seed,
                replicates=args.bootstrap_replicates,
            )
            paired_rows.append(
                {
                    "metric": metric,
                    "variant": variant,
                    "positive_change": positive,
                    "negative_change": negative,
                    "discordant_occurrences": positive + negative,
                    "net_effect": positive - negative,
                    "occurrence_exact_mcnemar_two_sided_p": exact_mcnemar_two_sided(
                        positive, negative
                    ),
                    "occurrence_exact_directional_one_sided_p": exact_binomial_tail(
                        positive, positive + negative
                    )
                    if positive + negative
                    else 1.0,
                    "cluster_sign_flip_one_sided_p": cluster_one,
                    "cluster_sign_flip_two_sided_p": cluster_two,
                    "cluster_bootstrap_ci90_low": bootstrap["ci90_low"],
                    "cluster_bootstrap_ci90_high": bootstrap["ci90_high"],
                    "cluster_bootstrap_ci95_low": bootstrap["ci95_low"],
                    "cluster_bootstrap_ci95_high": bootstrap["ci95_high"],
                    "cluster_bootstrap_replicates": bootstrap["replicates"],
                    "cluster_bootstrap_seed": bootstrap["seed"],
                }
            )

    summary = {
        "schema": SCHEMA,
        "analysis_status": "read-only post-hoc qualification of an opened-development panel",
        "input": {
            "paired_rows": str(args.paired_rows),
            "paired_rows_sha256": sha256_file(args.paired_rows),
            "rows": len(rows),
            "variants": ["BC", *VARIANTS],
            "startpoints": 50,
            "conditions_per_startpoint": 12,
        },
        "metrics": metric_summaries,
        "inference_contract": {
            "occurrence_exact_mcnemar": (
                "transparent case-level description; it ignores within-startpoint dependence"
            ),
            "cluster_sign_flip": (
                "exact enumeration conditional on the 50 observed block-effect vectors, but "
                "its inferential validity requires startpoint-level sign symmetry/exchangeability"
            ),
            "cluster_bootstrap": (
                "percentile interval from resampling the 50 startpoint blocks with replacement"
            ),
            "selection_adjustment": (
                "joint same-sign enumeration across the three correlated snapshots; the max-net-"
                "effect event is conservative because it does not condition on the overtake gate"
            ),
        },
        "claim_boundary": {
            "historical_verdict_preserved": "OPENED_DEVELOPMENT_SURVIVOR",
            "qualification": (
                "feasibility gate passed, paired safety effect statistically inconclusive, "
                "checkpoint not promoted"
            ),
            "fresh_or_final_confirmation": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir / "paired_inference.tsv", paired_rows)
    write_tsv(args.output_dir / "startpoint_effects.tsv", block_rows)
    json_dump(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
