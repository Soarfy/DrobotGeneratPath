"""Minimal URDF kinematic-chain parser and forward kinematics.

Only the subset used by the CR5A is supported: fixed/revolute/prismatic joints,
origin xyz/rpy, axis and limits. Fixed joints contribute a constant transform.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import se3


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    lower: float = 0.0
    upper: float = 0.0
    effort: float = 0.0
    velocity: float = 0.0
    movable: bool = False

    @property
    def is_movable(self) -> bool:
        return self.type in ("revolute", "continuous", "prismatic")


@dataclass
class Chain:
    """Ordered root -> tip chain."""

    joints: List[Joint]
    joint_names: List[str]
    lower: np.ndarray
    upper: np.ndarray
    velocity: np.ndarray

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    def fk(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != self.dof:
            raise ValueError(f"expected {self.dof} joint values, got {q.size}")
        T = np.eye(4)
        k = 0
        for joint in self.joints:
            T = T @ joint.origin
            if joint.is_movable:
                value = q[k]
                k += 1
                if joint.type == "prismatic":
                    T = T @ se3.make_transform(np.eye(3), joint.axis * value)
                else:
                    T = T @ se3.make_transform(se3.rotation_about_axis(joint.axis, value), np.zeros(3))
        return T

    def fk_all(self, q) -> List[np.ndarray]:
        """Return the world transform of every link frame along the chain."""
        q = np.asarray(q, dtype=float).reshape(-1)
        frames = []
        T = np.eye(4)
        k = 0
        for joint in self.joints:
            T = T @ joint.origin
            if joint.is_movable:
                value = q[k]
                k += 1
                if joint.type == "prismatic":
                    T = T @ se3.make_transform(np.eye(3), joint.axis * value)
                else:
                    T = T @ se3.make_transform(se3.rotation_about_axis(joint.axis, value), np.zeros(3))
            frames.append(T.copy())
        return frames

    def joint_frames(self, q):
        """World position/axis of every movable joint, for the geometric Jacobian.

        Returns (points Nx3, axes Nx3, T_tip 4x4).
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        points = []
        axes = []
        T = np.eye(4)
        k = 0
        for joint in self.joints:
            T = T @ joint.origin
            if joint.is_movable:
                value = q[k]
                k += 1
                # Axis and origin are expressed in the frame after the origin offset.
                axis_world = T[:3, :3] @ joint.axis
                norm = np.linalg.norm(axis_world)
                if norm > 1e-12:
                    axis_world = axis_world / norm
                points.append(T[:3, 3].copy())
                axes.append(axis_world)
                if joint.type == "prismatic":
                    T = T @ se3.make_transform(np.eye(3), joint.axis * value)
                else:
                    T = T @ se3.make_transform(
                        se3.rotation_about_axis(joint.axis, value), np.zeros(3)
                    )
        return np.array(points, dtype=float), np.array(axes, dtype=float), T

    def clamp(self, q):
        return np.minimum(np.maximum(np.asarray(q, dtype=float), self.lower), self.upper)

    def limit_violation(self, q) -> float:
        q = np.asarray(q, dtype=float)
        return float(np.max(np.maximum(self.lower - q, q - self.upper)))


def _parse_origin(element: Optional[ET.Element]) -> np.ndarray:
    if element is None:
        return np.eye(4)
    xyz = np.fromstring(element.attrib.get("xyz", "0 0 0"), sep=" ", dtype=float)
    rpy = np.fromstring(element.attrib.get("rpy", "0 0 0"), sep=" ", dtype=float)
    if xyz.size != 3:
        xyz = np.zeros(3)
    if rpy.size != 3:
        rpy = np.zeros(3)
    # URDF stores rpy in radians.
    return se3.make_transform(se3.rpy_to_matrix(rpy), xyz)


def parse_urdf(path) -> Dict[str, Joint]:
    root = ET.parse(str(path)).getroot()
    joints: Dict[str, Joint] = {}
    for elem in root.findall("joint"):
        name = elem.attrib.get("name", "")
        jtype = elem.attrib.get("type", "fixed")
        parent = elem.find("parent")
        child = elem.find("child")
        if parent is None or child is None:
            continue
        axis_elem = elem.find("axis")
        axis = np.array([0.0, 0.0, 1.0])
        if axis_elem is not None and axis_elem.attrib.get("xyz"):
            axis = np.fromstring(axis_elem.attrib["xyz"], sep=" ", dtype=float)
        limit = elem.find("limit")
        lower = upper = effort = velocity = 0.0
        if limit is not None:
            lower = float(limit.attrib.get("lower", 0.0))
            upper = float(limit.attrib.get("upper", 0.0))
            effort = float(limit.attrib.get("effort", 0.0))
            velocity = float(limit.attrib.get("velocity", 0.0))
        joint = Joint(
            name=name,
            type=jtype,
            parent=parent.attrib.get("link", ""),
            child=child.attrib.get("link", ""),
            origin=_parse_origin(elem.find("origin")),
            axis=se3.normalize(axis),
            lower=lower,
            upper=upper,
            effort=effort,
            velocity=velocity,
            movable=jtype in ("revolute", "continuous", "prismatic"),
        )
        joints[name] = joint
    return joints


def apply_position_limits(chain: "Chain", limits) -> "Chain":
    """Override the URDF joint limits in place.

    The shipped CR5A URDF uses +/-6.28 rad as a placeholder for five joints,
    which lets IK return physically impossible windings.
    """
    if limits is None:
        return chain
    lim = np.asarray(limits, dtype=float)
    if lim.shape != (chain.dof, 2):
        raise ValueError(f"position limits must be {chain.dof}x2, got {lim.shape}")
    chain.lower = lim[:, 0].copy()
    chain.upper = lim[:, 1].copy()
    return chain


def build_chain(path, tip_link: str, joint_order: Optional[List[str]] = None) -> Chain:
    """Build the chain from the URDF root link to ``tip_link``.

    ``joint_order`` fixes the joint ordering (important for matching the real
    controller). If omitted, the URDF chain order is used.
    """
    joints = parse_urdf(path)
    by_child = {j.child: j for j in joints.values()}
    chain: List[Joint] = []
    link = tip_link
    visited = set()
    while link in by_child:
        if link in visited:
            raise ValueError(f"cycle detected at link {link}")
        visited.add(link)
        joint = by_child[link]
        chain.append(joint)
        link = joint.parent
    chain.reverse()

    movable = [j for j in chain if j.is_movable]
    names = [j.name for j in movable]
    if joint_order:
        if names != list(joint_order):
            raise ValueError(
                "joint_order must match the kinematic chain order; "
                f"chain={names}, requested={list(joint_order)}"
            )

    names = [j.name for j in movable]
    lower = np.array([j.lower for j in movable], dtype=float)
    upper = np.array([j.upper for j in movable], dtype=float)
    velocity = np.array([j.velocity for j in movable], dtype=float)
    return Chain(joints=chain, joint_names=names, lower=lower, upper=upper, velocity=velocity)
