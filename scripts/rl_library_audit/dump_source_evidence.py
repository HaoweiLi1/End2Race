#!/usr/bin/env python3
"""Dump exact installed source requested by the audit using inspect.getsource()."""

from __future__ import annotations

import inspect
from pathlib import Path

from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer, create_sequencers, pad
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.ppo_recurrent import RecurrentPPO
from tianshou.data import Batch, Collector, ReplayBuffer
from tianshou.policy import PGPolicy, PPOPolicy
from tianshou.utils.net.common import Recurrent as TianshouRecurrent


TARGETS = [
    Collector.collect,
    Collector._reset_state,
    PGPolicy.forward,
    PPOPolicy.process_fn,
    PPOPolicy.learn,
    ReplayBuffer,
    ReplayBuffer.add,
    ReplayBuffer.get,
    Batch.split,
    TianshouRecurrent.forward,
    RecurrentPPO._setup_model,
    RecurrentPPO.collect_rollouts,
    RecurrentPPO.train,
    RecurrentRolloutBuffer,
    create_sequencers,
    pad,
    RecurrentActorCriticPolicy,
]


def main() -> None:
    chunks = []
    for target in TARGETS:
        lines, start = inspect.getsourcelines(target)
        path = Path(inspect.getsourcefile(target) or "").resolve()
        chunks.append(
            f"===== {target.__module__}.{target.__qualname__} {path}:{start}-{start + len(lines) - 1} =====\n"
            + inspect.getsource(target)
        )
    output = Path(__file__).with_name("source_evidence.txt")
    output.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
