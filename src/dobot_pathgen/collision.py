"""Geometric clearances between the tool, the robot links and the workpiece.

This is a proximity check, not a physics simulation: it samples the tool body
and the robot link meshes as point clouds and reports minimum distances. It is
enough to catch the failures that actually happen with a long brush mounted at
an angle (rod grazing the upper arm, tip disc hitting a neighbouring tooth).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import se3
from .mesh_io import read_stl
from .urdf_chain import Chain


# URDF link name -> mesh file stem used in assets/robot/cr5a/meshes/cr5/
LINK_MESHES = {
    "base_link": "base_link",
    "Link1": "J1",
    "Link2": "J2",
    "Link3": "J3",
    "Link4": "J4",
    "Link5": "J5",
    "Link6": "J6",
}


@dataclass
class ToolGeometry:
    """Tool body description in the Link6 frame."""

    tcp_offset_m: np.ndarray   # tip position in Link6
    radius_m: float            # rod radius
    head_radius_m: Optional[float] = None
    head_length_m: float = 0.03

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.tcp_offset_m))

    @property
    def axis(self) -> np.ndarray:
        return se3.normalize(self.tcp_offset_m)


def tool_points(geom: ToolGeometry, T_link: np.ndarray, n_around: int = 12) -> np.ndarray:
    """Sample the tool surface in the world frame for a given Link6 pose."""
    off = np.asarray(geom.tcp_offset_m, dtype=float)
    length = geom.length
    axis = geom.axis
    radius = float(geom.radius_m)

    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u = se3.normalize(np.cross(axis, ref))
    v = se3.normalize(np.cross(axis, u))

    angles = np.linspace(0.0, 2.0 * np.pi, n_around, endpoint=False)
    ring = np.empty((n_around, 3))
    for i, a in enumerate(angles):
        ring[i] = np.cos(a) * u + np.sin(a) * v

    R = T_link[:3, :3]
    p = T_link[:3, 3]
    pts: List[np.ndarray] = [(p + R @ np.zeros(3)).reshape(1, 3), (p + R @ off).reshape(1, 3)]

    # Rod side surface.
    for t in np.linspace(0.02, 0.98, 14):
        centre = p + t * length * (R @ axis)
        pts.append((radius * ring) @ R.T + centre)

    # Tool head: a slightly wider disc at the tip, perpendicular to the rod.
    head_r = float(geom.head_radius_m or radius)
    if head_r > radius * 1.01:
        head_start = max(0.0, 1.0 - geometry_fraction(geom))
        for t in np.linspace(head_start, 1.0, 3):
            centre = p + t * length * (R @ axis)
            pts.append((head_r * ring) @ R.T + centre)
        for k in (0.4, 0.7, 1.0):
            disc = np.column_stack(
                [
                    (head_r * k * np.cos(angles)),
                    (head_r * k * np.sin(angles)),
                    np.zeros_like(angles),
                ]
            )
            disc_world = (disc @ np.column_stack([u, v, axis]).T) @ R.T + (p + R @ off)
            pts.append(disc_world)
    return np.vstack(pts)


def geometry_fraction(geom: ToolGeometry) -> float:
    if geom.length < 1e-9:
        return 0.0
    return min(0.4, geom.head_length_m / geom.length)


def load_link_clouds(urdf_path, tool_mesh: Optional[Path] = None, stride: int = 8) -> Dict[str, np.ndarray]:
    """Load subsampled point clouds for each robot link, in link frames."""
    mesh_dir = Path(urdf_path).parent.parent / "meshes" / "cr5"
    clouds: Dict[str, np.ndarray] = {}
    for link, stem in LINK_MESHES.items():
        candidate = mesh_dir / f"{stem}.STL"
        if not candidate.exists():
            candidate = mesh_dir / f"{stem}.stl"
        if not candidate.exists():
            continue
        verts, _, _ = read_stl(candidate)
        clouds[link] = verts[::stride].copy()
    return clouds


def link_world_clouds(chain: Chain, q, link_clouds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Place each link's point cloud in the world using the chain frames."""
    frames = chain.fk_all(q)
    # chain.joints = [fixed dummy, joint1..joint6]; frames[k] is after chain.joints[k]
    order = ["base_link", "Link1", "Link2", "Link3", "Link4", "Link5", "Link6"]
    out: Dict[str, np.ndarray] = {}
    for k, link in enumerate(order):
        if k >= len(frames) or link not in link_clouds:
            continue
        T = frames[k]
        out[link] = link_clouds[link] @ T[:3, :3].T + T[:3, 3]
    return out


def _min_distance(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.spatial import cKDTree

        d, _ = cKDTree(b).query(a)
        return float(np.min(d))
    except ImportError:  # pragma: no cover
        # Chunked fallback so memory stays bounded.
        best = np.inf
        for i in range(0, len(a), 256):
            chunk = a[i:i + 256]
            d = np.linalg.norm(chunk[:, None, :] - b[None, :, :], axis=2)
            best = min(best, float(d.min()))
        return best


def check_clearances(
    chain: Chain,
    trajectory,
    geom: ToolGeometry,
    link_clouds: Dict[str, np.ndarray],
    workpiece_points: Optional[np.ndarray] = None,
    self_margin_m: float = 0.01,
    workpiece_margin_m: float = 0.0,
    exclude_links: Tuple[str, ...] = ("Link5", "Link6"),
) -> Dict:
    """Scan every waypoint for tool-vs-robot and tool-vs-workpiece clearance."""
    self_worst = (np.inf, -1, "")
    work_worst = (np.inf, -1)
    self_violations: List[Dict] = []
    work_violations: List[Dict] = []

    for i, w in enumerate(trajectory.waypoints):
        T_link = chain.fk(w.positions)
        tool = tool_points(geom, T_link)
        clouds = link_world_clouds(chain, w.positions, link_clouds)

        for link, cloud in clouds.items():
            if link in exclude_links:
                continue
            d = _min_distance(tool, cloud)
            if d < self_worst[0]:
                self_worst = (d, i, link)
            if d < self_margin_m:
                self_violations.append(
                    {"index": i, "link": link, "distance_mm": d * 1000.0}
                )

        if workpiece_points is not None and len(workpiece_points):
            d = _min_distance(tool_points(geom, T_link, n_around=16), workpiece_points)
            if d < work_worst[0]:
                work_worst = (d, i)
            if d < workpiece_margin_m:
                work_violations.append({"index": i, "distance_mm": d * 1000.0})

    return {
        "self_min_distance_mm": float(self_worst[0] * 1000.0) if np.isfinite(self_worst[0]) else None,
        "self_min_index": int(self_worst[1]),
        "self_min_link": self_worst[2],
        "self_violations": self_violations[:20],
        "self_violation_count": len(self_violations),
        "workpiece_min_distance_mm": (
            float(work_worst[0] * 1000.0) if np.isfinite(work_worst[0]) else None
        ),
        "workpiece_min_index": int(work_worst[1]),
        "workpiece_violations": work_violations[:20],
        "workpiece_violation_count": len(work_violations),
        "self_margin_mm": self_margin_m * 1000.0,
        "passed": len(self_violations) == 0 and len(work_violations) == 0,
    }
