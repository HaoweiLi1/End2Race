#!/usr/bin/env python3
"""Run the one allowed 128-to-256 episode P1 extension for inconclusive pools."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time

import torch

try:
    import audit_rl_gradient_direction as p1
    from audit_rl_direction_common import (
        EXPERIMENT_DIR,
        ROOT,
        RUN_DIR,
        assert_frozen_contract,
        make_env,
        read_json,
        sha256_file,
        write_json_atomic,
        FixedScenarioProvider,
    )
except ModuleNotFoundError:
    from scripts import audit_rl_gradient_direction as p1
    from scripts.audit_rl_direction_common import (
        EXPERIMENT_DIR,
        ROOT,
        RUN_DIR,
        assert_frozen_contract,
        make_env,
        read_json,
        sha256_file,
        write_json_atomic,
        FixedScenarioProvider,
    )


TARGET_POOLS = ("H0_CURRENT_DET", "H2_STOCH_CORE")


def main() -> None:
    started = time.monotonic()
    frozen_hashes = assert_frozen_contract()
    extension_preregistration = read_json(EXPERIMENT_DIR / "P1_EXTENSION_PREREGISTRATION.json")
    initial_result_path = EXPERIMENT_DIR / "P1_GRADIENT_DIRECTION.json"
    if sha256_file(initial_result_path) != extension_preregistration["initial_p1_result_sha256"]:
        raise RuntimeError("Initial P1 result drifted after extension preregistration")
    initial_result = read_json(initial_result_path)
    for pool_name in TARGET_POOLS:
        if initial_result["pool_verdicts"][pool_name] != "INCONCLUSIVE":
            raise RuntimeError(f"Pool is not eligible for the one extension: {pool_name}")

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("P1 extension requires CUDA")
    policy = p1._policy(device)
    initial_actor_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in policy.end2race_actor.state_dict().items()
    }
    extension_results = {}
    for pool_name in TARGET_POOLS:
        shard_records = [
            read_json(RUN_DIR / "p1" / pool_name / f"shard_{index}" / "shard_result.json")
            for index in range(4)
        ]
        seeds = extension_preregistration["seeds"][pool_name]
        for offset, seed in enumerate(seeds, start=4):
            shard_dir = RUN_DIR / "p1" / pool_name / f"shard_{offset}"
            shard_result_path = shard_dir / "shard_result.json"
            if shard_result_path.is_file():
                shard_record = read_json(shard_result_path)
                if len(shard_record["episodes"]) != 32 or int(shard_record["seed"]) != int(seed):
                    raise RuntimeError(f"Invalid extension shard record: {shard_result_path}")
                gradient_path = ROOT / shard_record["gradient"]["gradient_file"]
                if sha256_file(gradient_path) != shard_record["gradient"]["gradient_file_sha256"]:
                    raise RuntimeError(f"Extension gradient hash mismatch: {gradient_path}")
                shard_records.append(shard_record)
                print(f"P1_EXTENSION_SHARD_RESUME pool={pool_name} shard={offset}", flush=True)
                continue
            print(f"P1_EXTENSION_SHARD_START pool={pool_name} shard={offset} seed={seed}", flush=True)
            provider = FixedScenarioProvider()
            env = make_env(provider, int(seed))
            episodes = []
            try:
                for episode_index, (scenario, branch) in enumerate(p1._shard_scenarios(pool_name, int(seed))):
                    destination = shard_dir / "episodes" / f"episode_{episode_index:02d}_{scenario.scenario_id}.npz"
                    row = p1._collect_episode(
                        env,
                        provider,
                        policy,
                        scenario,
                        branch=branch,
                        pool_name=pool_name,
                        seed=int(seed),
                        episode_index=episode_index,
                        device=device,
                        destination=destination,
                    )
                    episodes.append(row)
                    print(
                        f"P1_EXTENSION_COLLECT pool={pool_name} shard={offset} "
                        f"episode={episode_index + 1}/32 outcome={row['outcome']}",
                        flush=True,
                    )
            finally:
                env.close()
            collection = p1._collection_summary(episodes)
            gradient = p1._gradient_shard(policy, episodes, shard_dir / "gradients.pt", device)
            shard_record = {
                "shard_index": offset,
                "seed": int(seed),
                "episodes": episodes,
                "collection": collection,
                "gradient": gradient,
            }
            write_json_atomic(shard_result_path, shard_record)
            shard_records.append(shard_record)
            current_actor_state = policy.end2race_actor.state_dict()
            if any(
                not torch.equal(current_actor_state[name].detach().cpu(), reference)
                for name, reference in initial_actor_state.items()
            ):
                raise RuntimeError("Actor parameters changed during P1 extension")
            print(
                f"P1_EXTENSION_SHARD_COMPLETE pool={pool_name} shard={offset} "
                f"collisions={collection['actual_ego_collisions']}",
                flush=True,
            )
        if len(shard_records) != 8:
            raise RuntimeError(f"P1 extension requires eight total shards for {pool_name}")
        extension_results[pool_name] = p1._aggregate_pool(pool_name, shard_records, device)
        write_json_atomic(RUN_DIR / "p1" / pool_name / "pool_result_256.json", extension_results[pool_name])
        print(
            f"P1_EXTENSION_POOL_COMPLETE pool={pool_name} "
            f"verdict={extension_results[pool_name]['verdict']} "
            f"median_cos={extension_results[pool_name]['pairwise_combined_median']:.6f}",
            flush=True,
        )

    extension_record = {
        "schema_version": 1,
        "record": "P1_ALLOWED_256_EPISODE_EXTENSION",
        "status": "COMPLETED_NO_FURTHER_EXTENSION_ALLOWED",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_head": initial_result["source_head"],
        "device": "cuda",
        "optimizer_steps": 0,
        "actor_parameters_bitwise_unchanged": True,
        "frozen_hashes": frozen_hashes,
        "pools": extension_results,
        "pool_verdicts": {name: record["verdict"] for name, record in extension_results.items()},
        "elapsed_seconds": float(time.monotonic() - started),
    }
    extension_path = EXPERIMENT_DIR / "P1_GRADIENT_DIRECTION_EXTENSION.json"
    write_json_atomic(extension_path, extension_record)

    final_verdicts = dict(initial_result["pool_verdicts"])
    final_verdicts.update(extension_record["pool_verdicts"])
    initial_result["status"] = "COMPLETED_AFTER_ALLOWED_EXTENSION"
    initial_result["initial_128_pool_verdicts"] = dict(initial_result["pool_verdicts"])
    initial_result["pool_verdicts"] = final_verdicts
    initial_result["allowed_extension"] = {
        "path": str(extension_path.relative_to(ROOT)),
        "sha256": sha256_file(extension_path),
        "pools": list(TARGET_POOLS),
        "additional_episodes_per_pool": 128,
        "final_episodes_per_extended_pool": 256,
        "second_extension_allowed": False,
    }
    initial_result["final_pool_summaries"] = {
        name: {
            "episodes": (
                extension_results[name]["complete_episodes"]
                if name in extension_results
                else initial_result["pools"][name]["complete_episodes"]
            ),
            "verdict": final_verdicts[name],
            "pairwise_combined_median": (
                extension_results[name]["pairwise_combined_median"]
                if name in extension_results
                else initial_result["pools"][name]["pairwise_combined_median"]
            ),
            "collision_action_delta_sign_agreement": (
                extension_results[name]["probe"]["collision_action_delta_sign_agreement"]
                if name in extension_results
                else initial_result["pools"][name]["probe"]["collision_action_delta_sign_agreement"]
            ),
        }
        for name in p1.POOL_NAMES
    }
    write_json_atomic(initial_result_path, initial_result)
    print(
        f"P1_EXTENSION_COMPLETE verdicts={json.dumps(final_verdicts, sort_keys=True)} "
        f"elapsed_seconds={extension_record['elapsed_seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
