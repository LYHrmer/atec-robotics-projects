"""Task E observation-only geometry. Runtime dependencies: NumPy and SciPy.

Constants are exported from the local Piper USD joint frames, not simulator
state. All quaternions in this module are scalar-first (w, x, y, z). Positions
and FK results are in the Task E world frame; joint angles are radians.
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


TABLE_TOP_Z = 0.6613141183247961 * 1.25
TABLE_HALF_X = 0.6468062441005529 * 1.25 / 2
BASE_POSITION = np.array([1.0 + TABLE_HALF_X, 0.0, TABLE_TOP_Z])
BASE_QUATERNION = np.array([0., 0., 0., 1.])
DEFAULT_JOINT_POS = np.array([0., 1.2, -1.5, 0., 1.2, 0., .035, -.035])
JOINT_LOWER = np.deg2rad([-150.00035095214844, 0., -169.99656677246094,
                         -99.98113250732422, -69.90084838867188, -120.0002670288086])
JOINT_UPPER = np.deg2rad([124.21723937988281, 179.9087371826172, 0.,
                         99.98113250732422, 69.90084838867188, 120.0002670288086])
# Every revolute axis is local +Z. T_parent_child(q)=T_parent_joint Rz(q).
_JOINT_POS = np.array([[0., 0., .12300000339746475], [0., 0., 0.],
                       [.2850300073623657, 0., 0.], [-.021984, -.25075, 0.],
                       [0., 0., 0.], [.000088259, -.091, 0.]])
_JOINT_QUAT = np.array([[1., 0., 0., 0.],
                        [.04800847, -.048013408, -.7054761, -.7054739],
                        [.6239962, 0., 0., -.7814274],
                        [.7071055, .7071081, 0., 0.],
                        [.7071055, -.7071081, 0., 0.],
                        [.7071055, .7071081, 0., 0.]])
_JOINT_ROT = Rotation.from_quat(_JOINT_QUAT[:, [1, 2, 3, 0]]).as_matrix()
VIDEO_CAM_POS = np.array([-.2, 0., TABLE_TOP_Z + .8])
VIDEO_CAM_QUAT_WORLD = np.array([.957, 0., .290, 0.])
# Camera world convention: +X forward, +Y left, +Z up. ROS: +Z forward,
# +X right, +Y down. The following maps ROS vectors to camera-world axes.
_ROS_TO_WORLD_CAMERA = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
VIDEO_FOCAL_LENGTH = 24.0
VIDEO_HORIZONTAL_APERTURE = 20.955


def quat_matrix(quaternion):
    return Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()


def matrix_quat(matrix):
    return Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]


def top_grasp_rotation(jaw_xy):
    """Downward approach with jaw local Y parallel to the supplied XY axis."""
    jaw = np.r_[np.asarray(jaw_xy, dtype=float)[:2], 0.]
    jaw /= np.linalg.norm(jaw)
    approach = np.array([0., 0., -1.])
    return np.column_stack((np.cross(jaw, approach), jaw, approach))


def fk(joints, *, base_position=BASE_POSITION, base_quaternion=BASE_QUATERNION,
       return_jacobian=False):
    """Return 4x4 world gripper_base transform (and optionally 6x6 Jacobian).

    Input accepts six arm joints or all eight joints. The grasping finger
    origins are 0.1358 m along gripper_base local +Z.
    """
    q = np.asarray(joints, dtype=float).reshape(-1)[:6]
    if q.size != 6:
        raise ValueError("fk requires six arm joint positions")
    p = np.array(base_position, dtype=float, copy=True)
    r = quat_matrix(base_quaternion)
    origins, axes = [], []
    for i, angle in enumerate(q):
        p = p + r @ _JOINT_POS[i]
        r = r @ _JOINT_ROT[i]
        if return_jacobian:
            origins.append(p.copy())
            axes.append(r[:, 2].copy())
        c, s = np.cos(angle), np.sin(angle)
        r = r @ np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = r, p
    if not return_jacobian:
        return transform
    axes, origins = np.asarray(axes), np.asarray(origins)
    jacobian = np.concatenate((np.cross(axes, p - origins).T, axes.T), axis=0)
    return transform, jacobian


@dataclass
class IKResult:
    joints: np.ndarray
    position_error: float
    orientation_error: float
    success: bool
    evaluations: int


@dataclass
class GraspIKResult:
    ik: IKResult
    position: np.ndarray
    rotation: np.ndarray
    tilt_degrees: float
    finger_floor_clearance: float


def solve_ik(position, rotation=None, seed=None, *, orientation_weight=.20,
             multi_start=True, max_nfev=90, position_tolerance=.005,
             orientation_tolerance=.08):
    """Bounded pose IK; returns errors so unreachable targets can be rejected.

    rotation accepts a 3x3 matrix or scalar-first quaternion; None is position
    only. Multi-start should be used once when planning a phase. For repeated
    close-loop corrections pass last solution as seed and multi_start=False.
    """
    desired_pos = np.asarray(position, dtype=float)
    desired_rot = None if rotation is None else np.asarray(rotation, dtype=float)
    if desired_rot is not None and desired_rot.shape == (4,):
        desired_rot = quat_matrix(desired_rot)
    lo, hi = JOINT_LOWER + 1e-6, JOINT_UPPER - 1e-6
    initial = np.asarray(DEFAULT_JOINT_POS[:6] if seed is None else seed, dtype=float)[:6]
    initial = np.clip(initial, lo, hi)

    def residual(q):
        pose = fk(q)
        delta = pose[:3, 3] - desired_pos
        if desired_rot is None:
            return delta
        angular = Rotation.from_matrix(pose[:3, :3] @ desired_rot.T).as_rotvec()
        return np.r_[delta, orientation_weight * angular]

    seeds = [initial]
    if multi_start:
        # Deterministic branches, with forearm folded to respect the joint3 sign.
        bearing = np.arctan2(-desired_pos[1], BASE_POSITION[0] - desired_pos[0])
        seeds += [np.array([bearing, j2, j3, j4, j5, 0.])
                  for j2, j3, j4, j5 in [(1.5, -1.8, 0., 1.),
                                        (.5, -1., 0., 1.),
                                        (2.5, -1.5, 0., -.8),
                                        (1.5, -2., 1.4, -.8),
                                        (1.5, -2., -1.4, -.8)]]
    best, count = None, 0
    for guess in seeds:
        fit = least_squares(residual, np.clip(guess, lo, hi), bounds=(lo, hi),
                            max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        count += fit.nfev
        pose = fk(fit.x)
        pe = float(np.linalg.norm(pose[:3, 3] - desired_pos))
        oe = 0. if desired_rot is None else float(
            np.linalg.norm(Rotation.from_matrix(pose[:3, :3] @ desired_rot.T).as_rotvec()))
        score = pe * pe + (orientation_weight * oe) ** 2
        if best is None or score < best[0]:
            best = (score, fit.x.copy(), pe, oe)
        if pe < position_tolerance and oe < orientation_tolerance:
            break
    _, joints, pe, oe = best
    return IKResult(joints, pe, oe,
                    pe < position_tolerance and oe < orientation_tolerance, count)


def differential_ik(joints, position, rotation=None, *, damping=.025,
                    max_joint_delta=.05, orientation_weight=.20):
    """One observation-only DLS update; suitable for low-cost visual servoing."""
    q = np.asarray(joints, dtype=float)[:6]
    pose, jac = fk(q, return_jacobian=True)
    error = np.asarray(position) - pose[:3, 3]
    if rotation is None:
        jac = jac[:3]
    else:
        rot = np.asarray(rotation)
        if rot.shape == (4,):
            rot = quat_matrix(rot)
        angular = Rotation.from_matrix(rot @ pose[:3, :3].T).as_rotvec()
        error = np.r_[error, orientation_weight * angular]
        jac[3:] *= orientation_weight
    dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(len(error)), error)
    dq = np.clip(dq, -max_joint_delta, max_joint_delta)
    return np.clip(q + dq, JOINT_LOWER, JOINT_UPPER)


def finger_floor_clearance(position, rotation, jaw_opening=.07):
    """Conservative lowest finger-box height above the table, in metres.

    Collision meshes in gripper coordinates span X +-0.028, Z 0.0593..0.1358;
    each finger is 0.0265 thick outside its inner contact plane.
    """
    row = np.asarray(rotation)[2]
    lowest = (float(np.asarray(position)[2]) - abs(row[0]) * .028
              - abs(row[1]) * (jaw_opening / 2 + .0265)
              + min(row[2] * .0593, row[2] * .1358))
    return lowest - TABLE_TOP_Z


def solve_grasp_ik(contact_position, jaw_xy, seed=None, *, grasp_depth=.115,
                   tilt_degrees=(0., 5., 10., 15., 20., 25., 30., 40.),
                   minimum_floor_clearance=.002, orientation_weight=.15):
    """Search feasible downward/tilted jaw poses around a desired contact point.

    Tilt points the fingers outward from the robot base, increasing reach while
    keeping the desired finger contact point fixed. Both signs of the jaw axis
    are tested because the finite wrist-roll limit breaks their IK symmetry.
    ``contact_position`` must be a visible object surface/body point above the
    table; a banana contact height around table+0.030 with depth=.115 leaves
    fingertips above the table. The returned clearance is conservative.
    """
    contact = np.asarray(contact_position, dtype=float)
    radial = contact[:2] - BASE_POSITION[:2]
    radial /= max(np.linalg.norm(radial), 1e-9)
    jaw_base = np.r_[np.asarray(jaw_xy, dtype=float)[:2], 0.]
    jaw_base /= np.linalg.norm(jaw_base)
    best = None
    for tilt in tilt_degrees:
        angle = np.deg2rad(tilt)
        approach = np.r_[radial * np.sin(angle), -np.cos(angle)]
        for sign in (1., -1.):
            jaw = jaw_base * sign
            jaw = jaw - approach * np.dot(jaw, approach)
            jaw /= np.linalg.norm(jaw)
            rotation = np.column_stack((np.cross(jaw, approach), jaw, approach))
            position = contact - grasp_depth * approach
            clearance = finger_floor_clearance(position, rotation)
            if clearance < minimum_floor_clearance:
                continue
            ik = solve_ik(position, rotation, seed, orientation_weight=orientation_weight,
                          multi_start=False, position_tolerance=.003,
                          orientation_tolerance=.05)
            candidate = GraspIKResult(ik, position, rotation, float(tilt), clearance)
            score = ik.position_error + orientation_weight * ik.orientation_error
            if best is None or score < best[0]:
                best = score, candidate
            if ik.success:
                return candidate
    if best is None:
        raise ValueError('Every grasp pose would put the fingers below the table; raise contact height')
    return best[1]


def video_camera_parameters(width=640, height=480):
    """Return (3x3 K, 4x4 world_from_ros_camera) for obs video RGB-D."""
    f = width * VIDEO_FOCAL_LENGTH / VIDEO_HORIZONTAL_APERTURE
    k = np.array([[f, 0., width / 2], [0., f, height / 2], [0., 0., 1.]])
    transform = np.eye(4)
    transform[:3, :3] = quat_matrix(VIDEO_CAM_QUAT_WORLD) @ _ROS_TO_WORLD_CAMERA
    transform[:3, 3] = VIDEO_CAM_POS
    return k, transform


def ee_camera_parameters(joints, width=640, height=480,
                         focal_length=15., horizontal_aperture=20.955):
    """Calibrated Task E EE camera transform from FK.

    Defaults match BaseSceneCfg.ee_camera; EE offset uses ROS convention.
    """
    f = width * focal_length / horizontal_aperture
    k = np.array([[f, 0., width / 2], [0., f, height / 2], [0., 0., 1.]])
    offset = np.eye(4)
    offset[:3, :3] = Rotation.from_euler('z', -np.pi / 2).as_matrix()
    offset[:3, 3] = [-.05, 0., .06]
    return k, fk(joints) @ offset


def unproject_depth(depth, *, intrinsics=None, world_from_camera=None,
                    mask=None, stride=1, return_pixels=False):
    """Project metric optical-axis depth into world points.

    IsaacLab's 'depth' equals distance_to_image_plane, not radial range. Returns
    Nx3 points and, if requested, corresponding Nx2 integer (row, column).
    Zero, NaN and infinite depth values are discarded.
    """
    dep = np.asarray(depth).squeeze()
    if dep.ndim != 2:
        raise ValueError("depth must have shape HxW or HxWx1")
    height, width = dep.shape
    default_k, default_pose = video_camera_parameters(width, height)
    k = default_k if intrinsics is None else np.asarray(intrinsics)
    pose = default_pose if world_from_camera is None else np.asarray(world_from_camera)
    rows, cols = np.mgrid[0:height:stride, 0:width:stride]
    values = dep[rows, cols]
    valid = np.isfinite(values) & (values > 0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)[rows, cols]
    rows, cols, values = rows[valid], cols[valid], values[valid]
    camera = np.column_stack(((cols - k[0, 2]) * values / k[0, 0],
                              (rows - k[1, 2]) * values / k[1, 1], values))
    points = camera @ pose[:3, :3].T + pose[:3, 3]
    return (points, np.column_stack((rows, cols))) if return_pixels else points


def project_world(points, *, intrinsics=None, world_from_camera=None,
                  width=640, height=480):
    """Return Nx2 (column,row) pixel coordinates and optical-axis depth."""
    default_k, default_pose = video_camera_parameters(width, height)
    k = default_k if intrinsics is None else np.asarray(intrinsics)
    pose = default_pose if world_from_camera is None else np.asarray(world_from_camera)
    cam = (np.asarray(points) - pose[:3, 3]) @ pose[:3, :3]
    uvw = cam @ k.T
    return uvw[..., :2] / uvw[..., 2:3], cam[..., 2]


def joints_from_proprio(proprio):
    """Recover the absolute eight joints from Task E's relative proprio[:8]."""
    return np.asarray(proprio, dtype=float).reshape(-1)[:8] + DEFAULT_JOINT_POS


def action_from_joint_targets(targets):
    return (np.asarray(targets, dtype=float) - DEFAULT_JOINT_POS) / .5
