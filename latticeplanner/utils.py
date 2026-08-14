import numpy as np
import math
from numba import njit
from scipy.spatial import cKDTree
import yaml
from types import SimpleNamespace as Namespace
import os

@njit(cache=True)
def avgPoint(vertices):
    return np.sum(vertices, axis=0) / vertices.shape[0]


@njit(cache=True)
def indexOfFurthestPoint(vertices, d):
    return np.argmax(vertices.dot(d))


@njit(cache=True)
def support(vertices1, vertices2, d):
    i = indexOfFurthestPoint(vertices1, d)
    j = indexOfFurthestPoint(vertices2, -d)
    return vertices1[i] - vertices2[j]

@njit(cache=True)
def closestPoint2Origin(a, b):
    ab = b-a
    ao = -a
    length = ab.dot(ab)
    if length < 1e-10:
        return a
    frac = ao.dot(ab)/length
    if frac < 0:
        return a
    if frac > 1:
        return b
    return frac*ab+a


@njit(cache=True)
def distance(vertices1, vertices2, direc):

    vertices1 = np.ascontiguousarray(vertices1)
    vertices2 = np.ascontiguousarray(vertices2)
    direc = np.ascontiguousarray(direc)
    a = support(vertices1, vertices2, direc)
    b = support(vertices1, vertices2, -direc)
    d = closestPoint2Origin(a, b)
    dist = np.linalg.norm(d)
    while True:
        if dist < 1e-10:
            return dist
        d = -d
        c = support(vertices1, vertices2, d)
        temp1 = c.dot(d)
        temp2 = a.dot(d)
        # should get bigger along d or you hit the end
        if (temp1 - temp2) < 1e-10:
            return dist
        p1 = closestPoint2Origin(a, c)
        p2 = closestPoint2Origin(c, b)
        dist1 = np.linalg.norm(p1)
        dist2 = np.linalg.norm(p2)
        if dist1 < dist2:
            b = c
            d = p1
            dist = dist1
        else:
            a = c
            d = p2
            dist = dist2

@njit(cache=True)
def get_vertices(pose, length, width):
    """
    Utility function to return vertices of the car body given pose and size
    Args:
        pose (np.ndarray, (3, )): current world coordinate pose of the vehicle
        length (float): car length
        width (float): car width
    Returns:
        vertices (np.ndarray, (4, 2)): corner vertices of the vehicle body
    """
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    x, y = pose[0], pose[1]
    tl_x = -length/2 * c + width/2 * (-s) + x
    tl_y = -length / 2 * s + width / 2 * c + y
    tr_x = length/2 * c + width/2 * (-s) + x
    tr_y = length / 2 * s + width / 2 * c + y
    bl_x = -length/2 * c + (-width/2) * (-s) + x
    bl_y = -length / 2 * s + (-width / 2) * c + y
    br_x = length/2 * c + (-width/2) * (-s) + x
    br_y = length / 2 * s + (-width / 2) * c + y
    vertices = np.asarray([[tl_x, tl_y], [bl_x, bl_y], [br_x, br_y], [tr_x, tr_y]])
    # assert np.linalg.norm(vertices_1-vertices) < 1e-4
    # print(vertices_1, vertices)
    return vertices

# @njit(cache=True)
def obsDict2oppoArray(obs, ego_idx=0):
    agent_num = len(obs['poses_x'])
    res = []
    for i in range(agent_num):
        if i != ego_idx:
            res.append(np.array([obs['poses_x'][i], obs['poses_y'][i], obs['poses_theta'][i]]))
    return np.array(res)


@njit(cache=True)
def _geometric_progress(points):
    progress = np.zeros(len(points))
    for index in range(len(points) - 1):
        progress[index + 1] = progress[index] + np.linalg.norm(points[index + 1] - points[index])
    return progress


@njit(cache=True)
def _interpolate_geometric_progress(progress, points, index, fraction):
    return progress[index] + fraction * np.linalg.norm(points[index + 1] - points[index])


class TrackProjector:

    def __init__(self, points, progress=None):
        self.points = np.asarray(points, dtype=np.float64)
        self.segment_start = self.points[:-1]
        self.segment_vector = self.points[1:] - self.points[:-1]
        self.segment_norm_sq = np.einsum("ij,ij->i", self.segment_vector, self.segment_vector)
        self.segment_length = np.sqrt(self.segment_norm_sq)
        if progress is None:
            self.progress = _geometric_progress(self.points)
            self.wrap_progress = False
        else:
            self.progress = np.asarray(progress, dtype=np.float64)
            self.wrap_progress = True
        self.track_length = float(self.progress[-1])
        self.segment_progress = self.progress[1:] - self.progress[:-1]
        self.tree = cKDTree(self.segment_start)

    @classmethod
    def from_csv(cls, path):
        reference = np.loadtxt(path, delimiter=";", comments="#", dtype=np.float64)
        return cls(reference[:, 1:3], reference[:, 0])

    def nearest(self, point):
        point = np.asarray(point, dtype=np.float64)
        offsets = point - self.segment_start
        fractions = np.clip(
            np.einsum("ij,ij->i", offsets, self.segment_vector) / self.segment_norm_sq,
            0.0,
            1.0,
        )
        projections = self.segment_start + fractions[:, None] * self.segment_vector
        differences = point - projections
        distance_sq = np.einsum("ij,ij->i", differences, differences)
        index = int(np.argmin(distance_sq))
        return projections[index], float(np.sqrt(distance_sq[index])), float(fractions[index]), index

    def progress_at(self, point):
        _projection, _distance, fraction, index = self.nearest(point)
        if not self.wrap_progress:
            return float(_interpolate_geometric_progress(self.progress, self.points, index, fraction))
        return float((self.progress[index] + fraction * self.segment_progress[index]) % self.track_length)

    def frenet(self, point):
        point = np.asarray(point, dtype=np.float64)
        _distance, nearest = self.tree.query(point)
        candidates = np.asarray((nearest, (nearest - 1) % len(self.segment_start)), dtype=np.int64)
        starts = self.segment_start[candidates]
        vectors = self.segment_vector[candidates]
        offsets = point - starts
        fractions = np.clip(
            np.einsum("ij,ij->i", offsets, vectors) / self.segment_norm_sq[candidates],
            0.0,
            1.0,
        )
        projections = starts + fractions[:, None] * vectors
        differences = point - projections
        choice = int(np.argmin(np.einsum("ij,ij->i", differences, differences)))
        segment = int(candidates[choice])
        tangent = self.segment_vector[segment] / self.segment_length[segment]
        normal = np.asarray((-tangent[1], tangent[0]), dtype=np.float64)
        progress = (self.progress[segment] + fractions[choice] * self.segment_progress[segment]) % self.track_length
        lateral = float(np.dot(differences[choice], normal))
        heading = float(np.arctan2(tangent[1], tangent[0]))
        return float(progress), lateral, heading


@njit(cache=True)
def intersect_point(point, radius, trajectory, t=0.0, wrap=False):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    epsilon = 1e-9
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :]
        V = np.ascontiguousarray(end - start)

        a = np.dot(V, V)
        if a <= epsilon:
            continue
        b = 2.0 * np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
        discriminant = b * b - 4 * a * c

        if discriminant < -epsilon:
            continue
        discriminant = np.sqrt(max(discriminant, 0.0))
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)
        minimum_t = start_t - epsilon if i == start_i else -epsilon
        for root in (t1, t2):
            if minimum_t <= root <= 1.0 + epsilon:
                first_t = min(max(root, 0.0), 1.0)
                first_i = i
                first_p = start + first_t * V
                break
        if first_p is not None:
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :]
            V = end - start

            a = np.dot(V, V)
            if a <= epsilon:
                continue
            b = 2.0 * np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
            discriminant = b * b - 4 * a * c

            if discriminant < -epsilon:
                continue
            discriminant = np.sqrt(max(discriminant, 0.0))
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
            for root in (t1, t2):
                if -epsilon <= root <= 1.0 + epsilon:
                    first_t = min(max(root, 0.0), 1.0)
                    first_i = i
                    first_p = start + first_t * V
                    break
            if first_p is not None:
                break

    return first_p, first_i, first_t

"""
Geometry utilities
"""



@njit(cache=True)
def get_rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.ascontiguousarray(np.array([[c, -s], [s, c]]))


@njit(cache=True)
def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi
    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle


@njit(cache=True)
def zero_2_2pi(angle):
    if angle > 2 * math.pi:
        return angle - 2.0 * math.pi
    if angle < 0:
        return angle + 2.0 * math.pi

    return angle

# (x0, y0, theta0, k0, dk, arc_length)
def sample_traj(clothoid, npts, v):
    # traj (m, 5)
    traj = np.empty((npts, 5))
    k0 = clothoid.Parameters[3]
    dk = clothoid.Parameters[4]

    for i in range(npts):
        s = i * (clothoid.length / max(npts - 1, 1))
        traj[i, 0] = clothoid.X(s)
        traj[i, 1] = clothoid.Y(s)
        traj[i, 2] = v
        traj[i, 3] = clothoid.Theta(s)
        traj[i, 4] = np.sqrt(clothoid.XDD(s) ** 2 + clothoid.YDD(s) ** 2)
    return traj


@njit(cache=True)
def xy_2_rc(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution):
    """
    Translate (x, y) coordinate into (r, c) in the matrix
        Args:
            x (float): coordinate in x (m)
            y (float): coordinate in y (m)
            orig_x (float): x coordinate of the map origin (m)
            orig_y (float): y coordinate of the map origin (m)

        Returns:
            r (int): row number in the transform matrix of the given point
            c (int): column number in the transform matrix of the given point
    """
    # translation
    x_trans = x - orig_x
    y_trans = y - orig_y

    # rotation
    x_rot = x_trans * orig_c + y_trans * orig_s
    y_rot = -x_trans * orig_s + y_trans * orig_c

    # clip the state to be a cell
    if x_rot < 0 or x_rot >= width * resolution or y_rot < 0 or y_rot >= height * resolution:
        c = -1
        r = -1
    else:
        c = int(x_rot / resolution)
        r = int(y_rot / resolution)

    return r, c


@njit(cache=True)
def map_collision(points, dt, map_metainfo, eps=0.4):
    """
    Check wheter a point is in collision with the map

    Args:
        points (numpy.ndarray(N, 2)): points to check
        dt (numpy.ndarray(n, m)): the map distance transform
        map_metainfo (tuple (x, y, c, s, h, w, resol)): map metainfo
        eps (float, default=0.1): collision threshold
    Returns:
        collisions (numpy.ndarray (N, )): boolean vector of wheter input points are in collision

    """
    orig_x, orig_y, orig_c, orig_s, height, width, resolution = map_metainfo
    collisions = np.empty((points.shape[0],))
    for i in range(points.shape[0]):
        if dt[xy_2_rc(points[i, 0], points[i, 1], orig_x, orig_y, orig_c, orig_s, height, width, resolution)] <= eps:
            collisions[i] = True
        else:
            collisions[i] = False
    return np.ascontiguousarray(collisions)


@njit(cache=True)
def x2y_distances_argmin(X, Y):
    """
    X: (n, 2)
    Y: (m, 2)

    return (n, 1)
    """
    n = len(X)
    min_idx = np.zeros(n)
    for i in range(n):
        diff = Y - X[i]  # (m, 2)
        # It is because numba does not support 'axis' keyword
        norm2 = diff * diff  # (m, 2)
        norm2 = norm2[:, 0] + norm2[:, 1]
        min_idx[i] = np.argmin(norm2)
    return min_idx

@njit(cache=True)
def tripleProduct(a, b, c):
    """
    Return triple product of three vectors
    Args:
        a, b, c (np.ndarray, (2,)): input vectors
    Returns:
        (np.ndarray, (2,)): triple product
    """
    ac = a.dot(c)
    bc = b.dot(c)
    return b * ac - a * bc


@njit(cache=True)
def perpendicular(pt):
    """
    Return a 2-vector's perpendicular vector
    Args:
        pt (np.ndarray, (2,)): input vector
    Returns:
        pt (np.ndarray, (2,)): perpendicular vector
    """
    temp = pt[0]
    pt[0] = pt[1]
    pt[1] = -1 * temp
    return pt


@njit(cache=True)
def collision(vertices1, vertices2):
    """
    GJK test to see whether two bodies overlap
    Args:
        vertices1 (np.ndarray, (n, 2)): vertices of the first body
        vertices2 (np.ndarray, (n, 2)): vertices of the second body
    Returns:
        overlap (boolean): True if two bodies collide
    """
    index = 0
    simplex = np.empty((3, 2))

    position1 = avgPoint(vertices1)
    position2 = avgPoint(vertices2)

    d = position1 - position2

    if d[0] == 0 and d[1] == 0:
        d[0] = 1.0

    a = support(vertices1, vertices2, d)
    simplex[index, :] = a

    if d.dot(a) <= 0:
        return False

    d = -a

    iter_count = 0
    while iter_count < 1e3:
        a = support(vertices1, vertices2, d)
        index += 1
        simplex[index, :] = a
        if d.dot(a) <= 0:
            return False

        ao = -a

        if index < 2:
            b = simplex[0, :]
            ab = b - a
            d = tripleProduct(ab, ao, ab)
            if np.linalg.norm(d) < 1e-10:
                d = perpendicular(ab)
            continue

        b = simplex[1, :]
        c = simplex[0, :]
        ab = b - a
        ac = c - a

        acperp = tripleProduct(ab, ac, ac)

        if acperp.dot(ao) >= 0:
            d = acperp
        else:
            abperp = tripleProduct(ac, ab, ab)
            if abperp.dot(ao) < 0:
                return True
            simplex[0, :] = simplex[1, :]
            d = abperp

        simplex[1, :] = simplex[2, :]
        index -= 1

        iter_count += 1
    return False

@njit(cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2] - position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    # curvature & steering
    radius = 1 / (2.0 * waypoint_y / lookahead_distance ** 2)
    steering_angle = np.arctan(wheelbase / radius)
    return speed, steering_angle

@njit(cache=True)
def get_actuation_PD(pose_theta, lookahead_point, position, lookahead_distance, wheelbase, prev_error, P, D):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2] - position)
    speed = lookahead_point[2]
    if lookahead_distance < 1e-6:
        return speed, 0., 0.
    error = 2.0 * waypoint_y / lookahead_distance ** 2
    if np.abs(waypoint_y) < 1e-4:
        return speed, 0., error
    steering_angle = P * error + D * (error-prev_error)
    return speed, steering_angle, error

def load_centerline_from_map(map_directory):
    """
    Load centerline from raceline1.csv
    """
    raceline1_path = os.path.join(map_directory, 'raceline1.csv')
    if os.path.exists(raceline1_path):
        wp = np.loadtxt(raceline1_path, delimiter=';', skiprows=1)
        return np.vstack((wp[:, 1], wp[:, 2])).T
    raise FileNotFoundError(f"raceline1.csv not found in {map_directory}")

def random_position(waypoints_xytheta, sampled_number=1, rng=None, xy_noise=0.0, theta_noise=0.0, 
                   ego_idx=100, interval_idx=20):
    """Generate random starting positions along waypoints with optional noise"""
    for i in range(sampled_number):
        starting_idx = (ego_idx + i * interval_idx) % len(waypoints_xytheta)
        x, y, theta = waypoints_xytheta[starting_idx][0], waypoints_xytheta[starting_idx][1], waypoints_xytheta[starting_idx][2]
        x = x + rng.random(size=1)[0] * xy_noise
        y = y + rng.random(size=1)[0] * xy_noise
        theta = zero_2_2pi(theta) + rng.random(size=1)[0] * theta_noise 
        if i == 0:
            res = np.array([[x, y, theta]])
        else:
            res = np.vstack((res, np.array([[x, y, theta]])))
    return res, ego_idx

def downsample_lidar(lidar_data, original_points=1440, target_points=360):
    """Downsample lidar data from original resolution to target resolution"""
    lidar_array = np.array(lidar_data)
    step = original_points // target_points
    downsampled = lidar_array[::step]
    return downsampled[:target_points]

def load_config(config_path='lattice_config.yaml'):
    """Load lattice planner configuration from YAML file"""
    with open(config_path) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    return Namespace(**config_dict)

def get_map_paths(map_name):
    """Generate map-related paths for a given map name"""
    map_directory = os.path.join('f1tenth_racetracks', map_name)
    map_path = os.path.join(map_directory, f'{map_name}_map')
    return map_directory, map_path
