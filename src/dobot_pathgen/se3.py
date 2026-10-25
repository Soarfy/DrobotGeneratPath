"""Small SE(3) helpers used by the planner (numpy only)."""

from __future__ import annotations

import math

import numpy as np


def rpy_to_matrix(rpy_rad) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = (float(v) for v in rpy_rad)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def rpy_deg_to_matrix(rpy_deg) -> np.ndarray:
    return rpy_to_matrix(np.deg2rad(np.asarray(rpy_deg, dtype=float)))


def make_transform(rotation: np.ndarray, translation) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(rotation, dtype=float)
    T[:3, 3] = np.asarray(translation, dtype=float)
    return T


def xyz_rpy_to_matrix(xyz_m, rpy_deg=(0.0, 0.0, 0.0)) -> np.ndarray:
    return make_transform(rpy_deg_to_matrix(rpy_deg), np.asarray(xyz_m, dtype=float))


def matrix_to_xyz_rpy_deg(T: np.ndarray):
    xyz = T[:3, 3].copy()
    rpy = matrix_to_rpy_deg(T[:3, :3])
    return xyz, rpy


def matrix_to_rpy_deg(R: np.ndarray):
    """Inverse of rpy_to_matrix, returns degrees."""
    R = np.asarray(R, dtype=float)
    sp = -R[2, 0]
    sp = max(-1.0, min(1.0, sp))
    pitch = math.asin(sp)
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return np.rad2deg([roll, pitch, yaw])


def rotation_about_axis(axis, angle_rad) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    axis = axis / n
    x, y, z = axis
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


def rotvec_from_matrix(R: np.ndarray) -> np.ndarray:
    """Log map of SO(3) -> rotation vector."""
    R = np.asarray(R, dtype=float)
    cos_theta = max(-1.0, min(1.0, (np.trace(R) - 1.0) * 0.5))
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    if abs(math.pi - theta) < 1e-6:
        # near pi: recover axis from the symmetric part
        A = (R + np.eye(3)) * 0.5
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        idx = int(np.argmax(axis))
        if axis[idx] > 1e-9:
            if idx == 0:
                axis[1] = A[0, 1] / axis[0]
                axis[2] = A[0, 2] / axis[0]
            elif idx == 1:
                axis[0] = A[0, 1] / axis[1]
                axis[2] = A[1, 2] / axis[1]
            else:
                axis[0] = A[0, 2] / axis[2]
                axis[1] = A[1, 2] / axis[2]
        n = np.linalg.norm(axis)
        if n < 1e-9:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = axis / n
        return axis * theta
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis * (theta / (2.0 * math.sin(theta)))


def quat_xyzw_from_matrix(R: np.ndarray):
    """Rotation matrix -> quaternion (x, y, z, w)."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return q


def matrix_from_quat_xyzw(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def normalize(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v
