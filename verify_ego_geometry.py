"""
Empirically verify two claims:
  (1) LiDAR originates from the ego's geometric center.
  (2) The car's physical half-length along its heading axis is 0.29 m
      (= length/2 with length=0.58), NOT 0.165 m (= (lf+lr)/2).

Method — "ruler-car" geometry test:
  Spawn ego at Austin raceline idx=0. Spawn a second agent ("opp") at a
  precisely known offset D along each cardinal direction (forward, rear,
  left, right) relative to the ego, with the same yaw.

  Then read the scan beam that points toward the opp:

      expected_reading = D  -  half_dim_of_opp_along_that_axis

  where half_dim = half_length (if forward/rear) or half_width (if left/right).

  Comparing expected vs observed for several D proves or disproves each claim:
    • If observed = D − 0.29 (forward/rear)  → claim (1) and (2) both hold.
    • If observed = D − 0.165 (forward/rear) → length assumption (2) is wrong.
    • If observed − expected ≠ 0 consistently → LiDAR is offset from center.
"""

import os
import numpy as np
import gym  # noqa
import f110_gym  # noqa

MAP_DIR   = "f1tenth_racetracks/Austin"
MAP_PATH  = os.path.join(MAP_DIR, "Austin_map")
RACELINE  = os.path.join(MAP_DIR, "raceline1.csv")

LENGTH = 0.58
WIDTH  = 0.31
HALF_L_PHYS = LENGTH / 2    # 0.29  (from get_vertices, full car body)
HALF_L_WB   = 0.165         # (lf+lr)/2 — the value currently in validator
HALF_W_PHYS = WIDTH / 2     # 0.155

wp = np.loadtxt(RACELINE, delimiter=";", skiprows=1)
x0, y0, yaw0 = wp[0, 1], wp[0, 2], wp[0, 3]

# Unit vectors in world frame
c, s = np.cos(yaw0), np.sin(yaw0)
FORWARD = np.array([ c,  s])   # vehicle +x
REAR    = np.array([-c, -s])
LEFT    = np.array([-s,  c])   # vehicle +y
RIGHT   = np.array([ s, -c])

# Beam indices (1440-beam raw scan). beam i has angle -π + i*(2π/1439)
# → beam 720 ≈ 0 rad (forward)   beam 0 ≈ -π (rear)
#   beam 360 ≈ -π/2 (right)      beam 1080 ≈ +π/2 (left)
BEAM = {"forward": 720, "rear": 0, "right": 360, "left": 1080}

# Which half-dimension the beam should probe
EXPECTED_HALF = {
    "forward": HALF_L_PHYS,
    "rear":    HALF_L_PHYS,
    "left":    HALF_W_PHYS,
    "right":   HALF_W_PHYS,
}

# How far ahead / aside to place the opp. Must (a) exceed ego+opp combined
# half-extent so they don't overlap, and (b) be closer than the track wall
# in that direction so the opp dominates the reading.
D_VALUES = {
    "forward": [0.80, 1.00, 1.50, 1.80],  # wall ~2.23 m ahead
    "rear":    [0.80, 1.00, 1.50, 1.80],  # wall ~2.15 m behind
    "left":    [0.50, 0.70, 0.85],        # wall ~1.03 m on the left
    "right":   [0.50, 0.70, 1.00],        # wall ~1.22 m on the right
}

UNIT = {"forward": FORWARD, "rear": REAR, "left": LEFT, "right": RIGHT}

# ─── Run experiment ──────────────────────────────────────────────────────
env = gym.make("f110-v0", map=MAP_PATH, map_ext=".png", timestep=0.01, num_agents=2)

print(f"Ego pose (raceline idx=0):  x={x0:.3f}  y={y0:.3f}  yaw={np.degrees(yaw0):+.2f}°\n")
header = f"{'dir':>8s} {'D':>6s} | {'obs':>8s} {'D-0.290':>9s} {'Δphys':>8s} | {'D-0.165':>9s} {'Δwb':>8s}"
print(header)
print("-" * len(header))

results = {k: [] for k in BEAM}

for direction in ["forward", "rear", "right", "left"]:
    unit = UNIT[direction]
    beam_idx = BEAM[direction]
    half_dim = EXPECTED_HALF[direction]

    for D in D_VALUES[direction]:
        opp_xy = np.array([x0, y0]) + D * unit
        poses = np.array([[x0, y0, yaw0],
                          [opp_xy[0], opp_xy[1], yaw0]])
        obs, *_ = env.reset(poses=poses)
        observed = float(np.asarray(obs["scans"][0])[beam_idx])

        # Expected readings under each hypothesis
        exp_phys = D - half_dim               # half_l = 0.29  (physical body)
        exp_wb   = D - (HALF_L_WB if direction in ("forward", "rear")
                         else HALF_W_PHYS)    # half_l = 0.165 (wheelbase)

        d_phys = observed - exp_phys
        d_wb   = observed - exp_wb

        print(f"{direction:>8s} {D:>6.2f} | "
              f"{observed:>8.4f} {exp_phys:>9.4f} {d_phys:>+8.4f} | "
              f"{exp_wb:>9.4f} {d_wb:>+8.4f}")
        results[direction].append((D, observed, exp_phys, exp_wb))

# ─── Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Interpretation")
print("=" * 60)
print("""
  Δphys = observed − (D − 0.290)   [hypothesis: full car body, LiDAR at center]
  Δwb   = observed − (D − 0.165)   [hypothesis: wheelbase half, LiDAR at center]

  • If Δphys ≈ 0 across all rows  →  half_l = 0.29 is physically correct,
                                     AND LiDAR is indeed at ego center.
  • If Δwb ≈ 0 across all rows    →  f1tenth treats car as 0.33 m long
                                     (contradicts get_vertices in source).
  • A constant non-zero offset in Δphys (e.g. +0.10 m for forward AND rear
    with equal sign) would indicate LiDAR is displaced along the heading.
  • Ray discretization / voxel artefacts are ~0.01 m, so |Δ| < 0.02 counts
    as an exact match.
""")
