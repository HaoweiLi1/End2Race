#!/usr/bin/env python3
"""Probe whether frozen BC recurrent features predict clearance risks.

This is a diagnostic for the "reward vs. representation" fork:

- If frozen BC GRU features predict front/side/rear risk well, the deployable
  representation already contains the information and reward shaping/gating is
  the likely bottleneck.
- If they do not, frozen features may be the bottleneck and unfreezing or an
  auxiliary deployable risk head becomes a stronger candidate.

The script does not train PPO and does not modify checkpoints.
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train_ppo import End2RacePPOEnv
from ppo_utils import RewardWeights, clearance_risk, load_frozen_bc, obs_to_tensors, relative_geometry, zero_hidden
from utils import wrap_rel_s


TARGETS = ("front_risk", "side_risk", "rear_risk", "clearance_risk")


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen BC feature -> clearance-risk probe.")
    parser.add_argument("--bc_model_path", type=str, default="pretrained/end2race.pth")
    parser.add_argument("--map_name", type=str, default="Austin")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--hidden_scale", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lateral_offset_prob", type=float, default=0.0)
    parser.add_argument("--opp_speedscale_min", type=float, default=0.45)
    parser.add_argument("--opp_speedscale_max", type=float, default=0.85)
    parser.add_argument("--interval_min", type=int, default=6)
    parser.add_argument("--interval_max", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="")
    return parser.parse_args()


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def current_risk_labels(env, obs):
    geom = relative_geometry(env._raw_obs, env.ref)
    rel_s = wrap_rel_s(geom["ego_s_raw"] - geom["opp_s_raw"], env.ref.track_length)
    risk = clearance_risk(rel_s, geom["lat_gap"], geom["ego_v_s"], geom["opp_v_s"], env.reward_weights)
    return np.array([risk[k] for k in TARGETS], dtype=np.float32)


def collect_features(args, device):
    rw = RewardWeights()
    env = End2RacePPOEnv(
        map_name=args.map_name,
        seed=args.train_seed,
        reward_weights=rw,
        lateral_offset_prob=args.lateral_offset_prob,
        speedscale_range=(args.opp_speedscale_min, args.opp_speedscale_max),
        interval_range=(args.interval_min, args.interval_max),
    )
    bc = load_frozen_bc(args.bc_model_path, device, args.hidden_scale)
    hidden = zero_hidden(bc.gru.hidden_size, device)
    obs = env.reset()

    xs, ys = [], []
    try:
        for _ in range(args.steps):
            lidar_t, speed_t = obs_to_tensors(obs, device)
            with torch.no_grad():
                feat, next_hidden = bc.forward_features(lidar_t, speed_t, hidden)
                action_t = bc.output_layer(feat)
            xs.append(feat.view(-1).detach().cpu().numpy().astype(np.float32))
            ys.append(current_risk_labels(env, obs))

            action = action_t.view(-1).detach().cpu().numpy().astype(np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs = env.reset()
                hidden = zero_hidden(bc.gru.hidden_size, device)
            else:
                hidden = next_hidden.detach()
    finally:
        env.close()

    return np.stack(xs), np.stack(ys)


class ProbeNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def fit_probe(x, y, args, device):
    rng = np.random.default_rng(args.train_seed)
    idx = rng.permutation(len(x))
    split = int(0.8 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]

    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std

    x_train = torch.as_tensor(x[train_idx], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y[train_idx], dtype=torch.float32, device=device)
    x_test = torch.as_tensor(x[test_idx], dtype=torch.float32, device=device)
    y_test = torch.as_tensor(y[test_idx], dtype=torch.float32, device=device)

    model = ProbeNet(x.shape[1], y.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for _ in range(args.epochs):
        perm = torch.randperm(x_train.shape[0], device=device)
        for start in range(0, x_train.shape[0], args.batch_size):
            b = perm[start:start + args.batch_size]
            pred = model(x_train[b])
            loss = loss_fn(pred, y_train[b])
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        pred = model(x_test).cpu().numpy()
    y_true = y_test.cpu().numpy()
    return y_true, pred


def metrics(y_true, pred):
    rows = []
    for i, name in enumerate(TARGETS):
        yt = y_true[:, i]
        yp = pred[:, i]
        mse = float(np.mean((yt - yp) ** 2))
        var = float(np.var(yt))
        r2 = float(1.0 - mse / max(var, 1e-8))
        if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(yt, yp)[0, 1])
        pos = float(np.mean(yt > 0.1))
        rows.append((name, mse, r2, corr, pos, float(np.mean(yt)), float(np.mean(yp))))
    return rows


def main():
    args = parse_args()
    device = choose_device(args.device)
    x, y = collect_features(args, device)
    y_true, pred = fit_probe(x, y, args, device)
    rows = metrics(y_true, pred)

    if args.out:
        out = Path(args.out)
    else:
        out = Path("logs") / f"probe_side_rear_risk_seed{args.train_seed}" / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        f.write("# Frozen BC Feature Risk Probe\n\n")
        f.write(f"- steps: `{args.steps}`\n")
        f.write(f"- seed: `{args.train_seed}`\n")
        f.write(f"- device: `{device}`\n")
        f.write(f"- lateral_offset_prob: `{args.lateral_offset_prob}`\n")
        f.write(f"- speedscale_range: `[{args.opp_speedscale_min}, {args.opp_speedscale_max}]`\n")
        f.write(f"- interval_range: `[{args.interval_min}, {args.interval_max})`\n\n")
        f.write("| target | mse | r2 | corr | frac > 0.1 | true mean | pred mean |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for name, mse, r2, corr, pos, ym, pm in rows:
            corr_s = "nan" if math.isnan(corr) else f"{corr:.3f}"
            f.write(f"| {name} | {mse:.5f} | {r2:.3f} | {corr_s} | {pos:.3f} | {ym:.4f} | {pm:.4f} |\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
