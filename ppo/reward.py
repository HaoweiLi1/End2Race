import numpy as np
from scipy.ndimage import map_coordinates
from latticeplanner.utils import get_vertices


class ClearanceCalculator:

    def __init__(self, distance_field, resolution, origin, vehicle_length, vehicle_width):
        self.distance_field = np.asarray(distance_field, dtype=np.float64)
        self.resolution = float(resolution)
        self.origin = np.asarray(origin, dtype=np.float64)
        self.vehicle_length = float(vehicle_length)
        self.vehicle_width = float(vehicle_width)
        self.map_cosine = float(np.cos(self.origin[2]))
        self.map_sine = float(np.sin(self.origin[2]))
        self.perimeter_spacing = 0.5 * self.resolution

    def _vehicle_clearances(self, ego_vertices, opponent_vertices, reference_heading):
        # Calculate OBB separation in the ego longitudinal and lateral frame.
        # Opposite rectangle edges share an axis, so each body contributes two SAT axes.
        overlap = True
        for rectangle in (ego_vertices, opponent_vertices):
            for index in (0, 1):
                edge = rectangle[(index + 1) % 4] - rectangle[index]
                axis = np.asarray((-edge[1], edge[0]), dtype=np.float64)
                ego_projection = ego_vertices @ axis
                opponent_projection = opponent_vertices @ axis
                if ego_projection.max() < opponent_projection.min() or opponent_projection.max() < ego_projection.min():
                    overlap = False
                    break
            if not overlap:
                break
        if overlap:
            return 0.0, 0.0, 0.0

        separation = np.zeros(2, dtype=np.float64)
        best_distance_sq = np.inf
        for source, target in ((ego_vertices, opponent_vertices), (opponent_vertices, ego_vertices)):
            for point in source:
                for index in range(4):
                    start = target[index]
                    segment = target[(index + 1) % 4] - start
                    fraction = np.clip(np.dot(point - start, segment) / np.dot(segment, segment), 0.0, 1.0)
                    vector = start + fraction * segment - point
                    distance_sq = float(np.dot(vector, vector))
                    if distance_sq < best_distance_sq:
                        separation = vector
                        best_distance_sq = distance_sq

        longitudinal_axis = np.asarray((np.cos(reference_heading), np.sin(reference_heading)))
        lateral_axis = np.asarray((-np.sin(reference_heading), np.cos(reference_heading)))
        return float(np.linalg.norm(separation)), abs(float(np.dot(separation, longitudinal_axis))), abs(float(np.dot(separation, lateral_axis)))

    def _wall_clearance(self, vertices):
        # Sample ego perimeter clearance from the map distance field.
        perimeter = []
        for index in range(4):
            start = vertices[index]
            end = vertices[(index + 1) % 4]
            sample_count = int(np.ceil(np.linalg.norm(end - start) / self.perimeter_spacing)) + 1
            # The next edge contains this edge's endpoint, so keep each corner only once.
            perimeter.append(np.linspace(start, end, sample_count)[:-1])

        points = np.concatenate(perimeter)
        translated = points - self.origin[:2]
        columns = (translated[:, 0] * self.map_cosine + translated[:, 1] * self.map_sine) / self.resolution
        rows = (-translated[:, 0] * self.map_sine + translated[:, 1] * self.map_cosine) / self.resolution
        # Read wall distance at the vehicle perimeter from the simulator's map distance field.
        distances = map_coordinates(self.distance_field, np.vstack((rows, columns)), order=1, mode="constant", cval=0.0)
        return max(0.0, float(distances.min()) - self.perimeter_spacing)

    def calculate(self, ego_pose, opponent_pose):
        # Return total, longitudinal, lateral and wall clearances.
        ego_vertices = get_vertices(ego_pose, self.vehicle_length, self.vehicle_width)
        opponent_vertices = get_vertices(opponent_pose, self.vehicle_length, self.vehicle_width)
        vehicle_clearances = self._vehicle_clearances(ego_vertices, opponent_vertices, ego_pose[2])
        # Total OBB distance, ego-frame longitudinal/lateral distance, then wall distance.
        return (*vehicle_clearances, self._wall_clearance(ego_vertices))


# Convert current clearances into a bounded negative risk potential.
def risk_potential(longitudinal_clearance_m, lateral_clearance_m, wall_clearance_m, *, longitudinal_safe_m, lateral_safe_m, wall_safe_m, maximum_magnitude):
    vehicle_distance = np.hypot(longitudinal_clearance_m / longitudinal_safe_m, lateral_clearance_m / lateral_safe_m)
    wall_distance = wall_clearance_m / wall_safe_m
    shortfall = max(0.0, 1.0 - min(vehicle_distance, wall_distance))
    return float(-maximum_magnitude * shortfall * shortfall)

# Reward ego forward progress.
def progress_reward(ego_delta, weight):
    return float(weight * ego_delta)

# Reward progress relative to a moving opponent.
def relative_reward(ego_delta, opponent_delta, opponent_collision_latched, weight):
    return 0.0 if opponent_collision_latched else float(weight * (ego_delta - opponent_delta))


# Apply the ego collision penalty once collision is observed.
def collision_reward(ego_collision, weight):
    return float(weight) if ego_collision else 0.0

# Shape reward from the change in risk potential.
def risk_reward(previous_potential, next_potential, gamma, terminated, weight):
    next_potential = 0.0 if terminated else next_potential
    return float(weight * (gamma * next_potential - previous_potential)), next_potential

# Measure signed progress across the cyclic track boundary.
def wrapped_progress_delta(current_s, previous_s, track_length):
    return float((current_s - previous_s + 0.5 * track_length) % track_length - 0.5 * track_length)
