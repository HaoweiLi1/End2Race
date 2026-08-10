import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CollisionBCAnchor:

    def __init__(self, path, policy, device):
        self.root = Path(path).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Collision BC anchor manifest does not exist: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("dataset_id") != "collision_bc_anchor_v1" or self.manifest.get("episode_count") != 18 or self.manifest.get("anchor_steps_per_episode") != 150:
            raise RuntimeError("Collision BC anchor manifest contract changed")
        keys = [row["episode_key"] for row in self.manifest["episodes"]]
        if len(keys) != 18 or len(set(keys)) != 18:
            raise RuntimeError("Collision BC anchor episode keys must be 18 unique values")
        self.policy = policy
        self.device = device
        self.episodes = []
        for row in self.manifest["episodes"]:
            sequence_path = self.root / row["sequence_file"]
            if not sequence_path.is_file() or sha256_file(sequence_path) != row["sequence_sha256"]:
                raise RuntimeError(f"Collision BC anchor sequence identity changed: {row['episode_key']}")
            with np.load(sequence_path, allow_pickle=False) as payload:
                observations = np.asarray(payload["observations"], dtype=np.float32)
                teacher_latent = np.asarray(payload["teacher_latent_steering_mean"], dtype=np.float32)
                teacher_speed = np.asarray(payload["teacher_physical_speed_mean"], dtype=np.float32)
                mask = np.asarray(payload["anchor_mask"], dtype=bool)
            length = int(row["sequence_steps"])
            start = int(row["anchor_start_index"])
            end = int(row["anchor_end_index_exclusive"])
            if observations.shape != (length, 361) or teacher_latent.shape != (length,) or teacher_speed.shape != (length,) or mask.shape != (length,):
                raise RuntimeError(f"Collision BC anchor sequence shapes changed: {row['episode_key']}")
            if end - start != 150 or int(mask.sum()) != 150 or not bool(mask[start:end].all()) or bool(mask[:start].any()) or bool(mask[end:].any()):
                raise RuntimeError(f"Collision BC anchor mask changed: {row['episode_key']}")
            if any(not np.isfinite(value).all() for value in (observations, teacher_latent, teacher_speed)):
                raise RuntimeError(f"Collision BC anchor sequence is non-finite: {row['episode_key']}")
            self.episodes.append({
                "episode_key": row["episode_key"],
                "observations": torch.as_tensor(observations, dtype=torch.float32, device=device),
                "teacher_latent": torch.as_tensor(teacher_latent, dtype=torch.float32, device=device),
                "teacher_speed": torch.as_tensor(teacher_speed, dtype=torch.float32, device=device),
                "mask": torch.as_tensor(mask, dtype=torch.bool, device=device),
            })

    def loss(self):
        total_losses = []
        steering_losses = []
        speed_losses = []
        for episode in self.episodes:
            observations = episode["observations"]
            hidden = torch.zeros((1, 1, self.policy.end2race_actor.gru.hidden_size), dtype=torch.float32, device=self.device)
            means = []
            for observation in observations:
                action, hidden = self.policy.end2race_actor(observation[:360].reshape(1, 1, 360), observation[360:].reshape(1, 1, 1), hidden)
                means.append(action[0, 0])
            mean_actions = torch.stack(means)
            distribution = self.policy._distribution(mean_actions)
            if distribution.latent_steer_mean is None:
                raise RuntimeError("Collision BC anchor distribution has no latent steering mean")
            mask = episode["mask"]
            steering = 0.5 * ((distribution.latent_steer_mean[mask] - episode["teacher_latent"][mask]) / 0.03).square().mean()
            speed = 0.5 * ((mean_actions[mask, 1] - episode["teacher_speed"][mask]) / 0.15).square().mean()
            steering_losses.append(steering)
            speed_losses.append(speed)
            total_losses.append(steering + speed)
        return torch.stack(total_losses).mean(), torch.stack(steering_losses).mean(), torch.stack(speed_losses).mean()

    def maximum_action_error(self):
        maximum = 0.0
        with torch.no_grad():
            for episode in self.episodes:
                observations = episode["observations"]
                hidden = torch.zeros((1, 1, self.policy.end2race_actor.gru.hidden_size), dtype=torch.float32, device=self.device)
                means = []
                for observation in observations:
                    action, hidden = self.policy.end2race_actor(observation[:360].reshape(1, 1, 360), observation[360:].reshape(1, 1, 1), hidden)
                    means.append(action[0, 0])
                mean_actions = torch.stack(means)
                distribution = self.policy._distribution(mean_actions)
                mask = episode["mask"]
                maximum = max(maximum, float(torch.max(torch.abs(distribution.latent_steer_mean[mask] - episode["teacher_latent"][mask])).item()))
                maximum = max(maximum, float(torch.max(torch.abs(mean_actions[mask, 1] - episode["teacher_speed"][mask])).item()))
        return maximum
