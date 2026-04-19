"""
Verify LiDAR beam-to-angle mapping and car-body geometry assumptions used in
episode_validator.py.

Procedure:
  1. Place ego at Austin raceline waypoint idx=0 (known pose).
  2. Capture the raw 1440-beam scan from f1tenth_gym.
  3. Downsample to 360 beams (matches demonstration.py / training data).
  4. Check three claims:
       (A) beam 180 in the 360-beam scan corresponds to vehicle FORWARD
           (because f1tenth uses angle_i = -π + i * (fov/(N-1)) with fov=2π).
       (B) half_l used in episode_validator.py (0.165 m) underestimates car
           length — actual car length is 0.58 m, so half_l should be 0.29 m.
           Consequence: computed `surface_dist = scan - d_edge` can go negative
           for front/rear beams when using 0.165, which is physically impossible
           (LiDAR cannot see the inside of its own car body).
       (C) current sector labels in check_proximity are rotated 180° — beam 0
           is actually the REAR, not the front.
"""

import os
import numpy as np
import gym  # noqa: F401  -- required to register f110-v0
import f110_gym  # noqa: F401


# ── Setup ────────────────────────────────────────────────────────────────
MAP_DIR   = "f1tenth_racetracks/Austin"
MAP_PATH  = os.path.join(MAP_DIR, "Austin_map")
RACELINE  = os.path.join(MAP_DIR, "raceline1.csv")

HALF_W              = 0.31 / 2          # 0.155 m  (correct)
HALF_L_CURRENT      = 0.165             # episode_validator.py current value
HALF_L_CORRECTED    = 0.58 / 2          # 0.29 m   (actual length/2)

wp = np.loadtxt(RACELINE, delimiter=";", skiprows=1)
x0, y0, yaw0 = wp[0, 1], wp[0, 2], wp[0, 3]
print(f"Ego pose at idx=0: x={x0:.4f}  y={y0:.4f}  yaw={yaw0:.4f} rad "
      f"({np.degrees(yaw0):+.2f}°)")

env = gym.make("f110-v0", map=MAP_PATH, map_ext=".png", timestep=0.01, num_agents=1)
obs, *_ = env.reset(poses=np.array([[x0, y0, yaw0]]))

scan_1440 = np.asarray(obs["scans"][0])
assert scan_1440.shape == (1440,), f"Unexpected scan shape {scan_1440.shape}"


# ── Angle mapping ───────────────────────────────────────────────────────
# From base_classes.py line 131: angle_i = -fov/2 + i * (fov/(N-1))
# With fov=2π, N=1440: beam 720 ≈ 0 rad (FORWARD)
FOV = 2 * np.pi
incr_1440 = FOV / (1440 - 1)
angles_1440 = -np.pi + np.arange(1440) * incr_1440

# Downsample (matches latticeplanner/utils.py::downsample_lidar)
scan_360 = scan_1440[::4][:360]
angles_360 = angles_1440[::4][:360]


# ── Claim A: beam 180 = FORWARD in the 360-beam scan ────────────────────
print("\n" + "=" * 60)
print("(A) Beam-to-angle mapping")
print("=" * 60)

key_360 = [
    ("beam 0",   0,   "expected: REAR  (-180°)"),
    ("beam 90",  90,  "expected: RIGHT ( -90°)"),
    ("beam 180", 180, "expected: FRONT (   0°)"),
    ("beam 270", 270, "expected: LEFT  ( +90°)"),
]
for name, idx, hint in key_360:
    print(f"  {name:9s}  angle={np.degrees(angles_360[idx]):+7.2f}°  "
          f"range={scan_360[idx]:7.3f} m   {hint}")

argmax_i = int(scan_360.argmax())
argmin_i = int(scan_360.argmin())
print(f"\n  Farthest reading : beam {argmax_i:3d} "
      f"(angle {np.degrees(angles_360[argmax_i]):+.2f}°) = {scan_360[argmax_i]:.2f} m")
print(f"  Closest reading  : beam {argmin_i:3d} "
      f"(angle {np.degrees(angles_360[argmin_i]):+.2f}°) = {scan_360[argmin_i]:.2f} m")


# ── Claim B: half_l undervalues car length ──────────────────────────────
print("\n" + "=" * 60)
print("(B) Car-body geometry (surface_dist = scan - d_edge)")
print("=" * 60)

def d_edge(angles, half_l, half_w):
    c = np.abs(np.cos(angles)) + 1e-12
    s = np.abs(np.sin(angles)) + 1e-12
    return np.minimum(half_l / c, half_w / s)

for label, hl in [("CURRENT   (half_l=0.165)", HALF_L_CURRENT),
                  ("CORRECTED (half_l=0.290)", HALF_L_CORRECTED)]:
    de = d_edge(angles_360, hl, HALF_W)
    surf = scan_360 - de
    n_neg = int((surf < -1e-4).sum())
    print(f"\n  {label}")
    print(f"    max d_edge             = {de.max():.4f} m")
    print(f"    min(scan - d_edge)     = {surf.min():+.4f} m")
    print(f"    # beams with surf < 0  = {n_neg}  "
          f"(physically impossible if LiDAR is at center)")


# ── Claim C: sector labels off by 180° ──────────────────────────────────
print("\n" + "=" * 60)
print("(C) check_proximity sector labels")
print("=" * 60)

sectors_current = [
    ("front (code)",      list(range(0, 30)) + list(range(330, 360))),
    ("front_right (code)",list(range(30, 60))),
    ("right (code)",      list(range(60, 120))),
    ("rear_right (code)", list(range(120, 150))),
    ("rear (code)",       list(range(150, 210))),
    ("rear_left (code)",  list(range(210, 240))),
    ("left (code)",       list(range(240, 300))),
    ("front_left (code)", list(range(300, 330))),
]
print("  Mean angle per 'code-labeled' sector (what the beams actually are):")
for name, idx in sectors_current:
    mean_ang = np.degrees(np.arctan2(
        np.sin(angles_360[idx]).mean(), np.cos(angles_360[idx]).mean()))
    print(f"    {name:22s}  actual mean angle = {mean_ang:+7.2f}°")
print("\n  → 'front (code)' beams are at ±180° (actually the REAR).")
print("  → 'rear (code)'  beams are at    0° (actually the FRONT).")


# ── Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
front_beam_claim = 180
rear_beam_claim  = 0
d_front = scan_360[front_beam_claim]
d_rear  = scan_360[rear_beam_claim]
print(f"  If beam 180 = FORWARD : reading = {d_front:.2f} m")
print(f"  If beam 0   = REAR    : reading = {d_rear:.2f} m")
print(f"  Yaw of ego = {np.degrees(yaw0):+.2f}°  →  forward in world ≈ this direction")
print(f"  Visually compare with the screenshot of Austin map at idx 0")
print(f"  to confirm which of the two directions has each observed distance.")
