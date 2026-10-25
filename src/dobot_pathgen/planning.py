"""Task/trajectory data model, IK compilation and validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import se3
from .ik import solve_ik_all_branches, solve_ik_multistart, solve_ik_tracking
from .pathgen import (
    SurfacePath,
    arch_surface_path,
    densify_positions,
    fixed_axis_tool_poses,
    normal_approach_tool_poses,
    fixed_orientation_tool_poses,
    densify_poses,
    limit_tilt,
    path_length,
    raster_surface_path,
    select_region,
    tool_poses_from_surface,
)
from .urdf_chain import Chain, build_chain


@dataclass
class Waypoint:
    time_s: float
    positions: List[float]          # radians, joint order of the chain
    surface_point: Optional[List[float]] = None
    tool_pose: Optional[List[List[float]]] = None


@dataclass
class Trajectory:
    joint_names: List[str]
    waypoints: List[Waypoint]
    meta: Dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.waypoints[-1].time_s if self.waypoints else 0.0

    def positions(self) -> np.ndarray:
        return np.array([w.positions for w in self.waypoints], dtype=float)

    def to_dict(self) -> Dict:
        return {
            "joint_names": self.joint_names,
            "meta": self.meta,
            "waypoints": [
                {
                    "time_s": w.time_s,
                    "positions": w.positions,
                    "surface_point": w.surface_point,
                    "tool_pose": w.tool_pose,
                }
                for w in self.waypoints
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Trajectory":
        return cls(
            joint_names=list(data["joint_names"]),
            meta=dict(data.get("meta", {})),
            waypoints=[
                Waypoint(
                    time_s=float(w["time_s"]),
                    positions=[float(v) for v in w["positions"]],
                    surface_point=w.get("surface_point"),
                    tool_pose=w.get("tool_pose"),
                )
                for w in data["waypoints"]
            ],
        )

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Trajectory":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class TaskSpec:
    name: str
    region: Dict
    spacing_m: float
    row_spacing_m: float
    standoff_m: float
    speed_m_s: float
    roll_deg: float = 0.0
    approach_clearance_m: float = 0.02
    acceleration_m_s2: float = 0.05
    # Tool X axis reference (model frame). Keeps wrist roll stable along the path.
    x_reference: Optional[List[float]] = None
    max_translation_step_m: float = 0.002
    max_rotation_step_deg: float = 3.0
    # "raster" covers a patch; "arch" walks the dental arch from end to end.
    pattern: str = "raster"
    arch: Dict = field(default_factory=dict)
    # Clamp tool tilt around the arch. None follows the surface normals exactly.
    max_tilt_deg: Optional[float] = None
    tilt_reference: List[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    # Hold the brush at a fixed orientation in the BASE frame instead of
    # following the local surface normal. The real cell keeps the rod roughly
    # 30-40 deg below horizontal pointing away from the robot.
    fixed_rod_direction: Optional[List[float]] = None
    # Follow the surface normal with the tool approach axis instead of holding
    # one fixed orientation. This is the brush tracking the tooth face.
    normal_approach: bool = False
    # Constant flange rotation in the base frame; overrides normal_approach.
    fixed_orientation_deg: Optional[List[float]] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "TaskSpec":
        return cls(
            name=str(data.get("name", "task")),
            region=dict(data.get("region", {"mode": "all"})),
            spacing_m=float(data.get("spacing_m", 0.0025)),
            row_spacing_m=float(data.get("row_spacing_m", 0.003)),
            standoff_m=float(data.get("standoff_m", 0.001)),
            speed_m_s=float(data.get("speed_m_s", 0.01)),
            roll_deg=float(data.get("roll_deg", 0.0)),
            approach_clearance_m=float(data.get("approach_clearance_m", 0.02)),
            acceleration_m_s2=float(data.get("acceleration_m_s2", 0.05)),
            x_reference=data.get("x_reference"),
            max_translation_step_m=float(data.get("max_translation_step_m", 0.002)),
            max_rotation_step_deg=float(data.get("max_rotation_step_deg", 3.0)),
            pattern=str(data.get("pattern", "raster")).lower(),
            arch=dict(data.get("arch", {})),
            max_tilt_deg=(
                None if data.get("max_tilt_deg") is None else float(data["max_tilt_deg"])
            ),
            tilt_reference=[
                float(v) for v in data.get("tilt_reference", [0.0, 1.0, 0.0])
            ],
            fixed_rod_direction=data.get("fixed_rod_direction"),
            normal_approach=bool(data.get("normal_approach", False)),
            fixed_orientation_deg=data.get("fixed_orientation_deg"),
        )


def load_robot(urdf_path, tip_link: str = "Link6"):
    chain = build_chain(urdf_path, tip_link)
    return chain


def _time_parameterize_joint_safe(
    q: np.ndarray,
    distances: np.ndarray,
    cart_speed: float,
    cart_accel: float,
    velocity_limits: Optional[np.ndarray] = None,
    acceleration_limits: Optional[np.ndarray] = None,
    scale: float = 0.8,
    min_dt: float = 0.05,
) -> np.ndarray:
    """Time-parameterize a joint sequence honouring Cartesian and joint limits.

    The Cartesian profile sets the nominal pace; any segment whose joint delta
    would exceed the per-joint velocity limit is stretched, then the result is
    iterated until joint accelerations are also acceptable.
    """
    dq = np.abs(np.diff(q, axis=0))
    n = len(dq)
    if n == 0:
        return np.zeros(1)

    # Nominal timing from the Cartesian arc length (trapezoid per segment).
    dt_cart = np.zeros(n)
    for i, d in enumerate(distances):
        d = float(max(d, 0.0))
        d_ramp = cart_speed * cart_speed / max(cart_accel, 1e-6)
        if d <= d_ramp:
            dt_cart[i] = 2.0 * np.sqrt(d / max(cart_accel, 1e-6))
        else:
            dt_cart[i] = d / max(cart_speed, 1e-6) + cart_speed / max(cart_accel, 1e-6)

    dt = np.maximum(dt_cart, min_dt)
    if velocity_limits is not None:
        vlim = np.asarray(velocity_limits, dtype=float) * scale
        vlim = np.where(vlim > 1e-9, vlim, np.inf)
        dt = np.maximum(dt, np.max(dq / vlim[None, :], axis=1))

    times = np.concatenate([[0.0], np.cumsum(dt)])

    if acceleration_limits is not None and len(q) > 2:
        alim = np.asarray(acceleration_limits, dtype=float) * scale
        alim = np.where(alim > 1e-9, alim, np.inf)
        for _ in range(8):
            vel = np.diff(q, axis=0) / np.maximum(np.diff(times)[:, None], 1e-9)
            acc = np.diff(vel, axis=0) / np.maximum(np.diff(times)[:-1, None], 1e-9)
            worst = np.max(np.abs(acc) / alim[None, :], axis=1)
            if np.all(worst <= 1.0 + 1e-6):
                break
            # Stretch the two segments adjacent to each violating waypoint.
            stretch = np.ones(n)
            for i, w in enumerate(worst):
                if w > 1.0:
                    factor = min(math.sqrt(float(w)), 4.0)
                    if i < n:
                        stretch[i] = max(stretch[i], factor)
                    if i + 1 < n:
                        stretch[i + 1] = max(stretch[i + 1], factor)
            dt = dt * stretch
            times = np.concatenate([[0.0], np.cumsum(dt)])

    times = np.maximum(times, np.arange(len(times)) * min_dt)
    return times


def compile_trajectory(
    chain: Chain,
    poses_model: np.ndarray,
    model_to_base: np.ndarray,
    speed_m_s: float,
    acceleration_m_s2: float,
    surface_points: Optional[np.ndarray] = None,
    reference_q=None,
    restarts: int = 24,
    seed: int = 0,
    min_dt: float = 0.05,
    tip_poses: bool = False,
    poses_are_flange: bool = False,
    tcp_in_link6: Optional[np.ndarray] = None,
    velocity_limits: Optional[np.ndarray] = None,
    acceleration_limits: Optional[np.ndarray] = None,
) -> Tuple[Trajectory, Dict]:
    """IK every pose, time-parameterize, and return a trajectory plus a report.

    ``poses_model`` are TCP (tool tip) poses in the model frame. Set
    ``tip_poses=False`` and pass ``tcp_in_link6`` when the chain ends at the
    last robot link instead of the TCP link.
    """
    n = len(poses_model)
    if n < 2:
        raise ValueError("need at least two tool poses")

    T_base = np.asarray(model_to_base, dtype=float)
    poses_base = np.array([T_base @ T for T in poses_model], dtype=float)
    if poses_are_flange:
        # Caller already produced flange poses (the fixed-axis brush path).
        poses_target = poses_base
    elif not tip_poses:
        if tcp_in_link6 is None:
            raise ValueError("tcp_in_link6 is required when tip_poses=False")
        inv_tcp = np.linalg.inv(np.asarray(tcp_in_link6, dtype=float))
        poses_target = np.array([T @ inv_tcp for T in poses_base], dtype=float)
    else:
        poses_target = poses_base

    q_ref = chain.clamp(reference_q) if reference_q is not None else None
    q_list: List[np.ndarray] = []
    valid_indices: List[int] = []
    failures: List[Dict] = []
    pos_err: List[float] = []
    rot_err: List[float] = []

    # Prefer the branch chosen at the FIRST waypoint for the whole path. Chasing
    # the previous waypoint's solution lets the solver drift onto another branch
    # mid-path, and with a 291 mm tool that shows up as abrupt wrist flips.
    # Solve every pose independently, keeping ALL distinct branches, then pick
    # one branch per waypoint with a shortest-path search over the joint-space
    # graph. Choosing per waypoint (or tracking from the previous solve) both
    # fail here: this pose family has 8 solutions and neighbours frequently sit
    # in different elbow/wrist families, so any greedy choice hops branches and
    # the arm gets 100+ deg single-step commands.
    # The branch set must be COMPLETE: with 12 random seeds only 3-6 of the 8
    # solutions are typically found, and a missing branch makes the DP choose a
    # 180 deg jump it could have avoided. 48 seeds reliably finds all 7-8.
    branch_seeds = max(48, restarts * 3)
    raw_solutions = [
        solve_ik_all_branches(chain, poses_target[i], restarts=branch_seeds, seed=seed + i)
        for i in range(n)
    ]

    usable = [i for i in range(n) if raw_solutions[i]]
    failures: List[Dict] = [
        {"index": i, "reason": "no IK solution for this pose"} for i in range(n) if not raw_solutions[i]
    ]
    if not usable:
        raise ValueError("no waypoint could be solved")

    # Unwrap every solution onto the first waypoint's frame so equivalent
    # angles (+179.9 vs -180.1) compare correctly.
    anchor = np.asarray(raw_solutions[usable[0]][0], dtype=float)
    span = chain.upper - chain.lower
    wrapped: List[List[np.ndarray]] = []
    for i in range(n):
        sols = []
        for q in raw_solutions[i]:
            q = np.asarray(q, dtype=float)
            q = anchor + (q - anchor + np.pi) % (2.0 * np.pi) - np.pi
            sols.append(chain.clamp(q))
        wrapped.append(sols)
    raw_solutions = wrapped

    # DP over the chain of waypoints. cost[k][j] = cheapest total travel to
    # reach solution j of waypoint order[k].
    order = list(usable)
    max_step = np.deg2rad(60.0)
    cost = [{j: 0.0 for j in range(len(raw_solutions[order[0]]))}]
    back: List[Dict[int, int]] = [{}]
    for k in range(1, len(order)):
        prev_i, cur_i = order[k - 1], order[k]
        cur_cost: Dict[int, float] = {}
        cur_back: Dict[int, int] = {}
        for j, qc in enumerate(raw_solutions[cur_i]):
            best_prev, best_val = None, float("inf")
            for jp, qp in enumerate(raw_solutions[prev_i]):
                step = float(np.max(np.abs(qc - qp)))
                if step > max_step:
                    continue
                if jp not in cost[-1]:
                    continue
                val = cost[-1][jp] + step
                if val < best_val:
                    best_prev, best_val = jp, val
            if best_prev is not None:
                cur_cost[j] = best_val
                cur_back[j] = best_prev
        if not cur_cost:
            # Disconnected: allow the cheapest jump rather than dropping points.
            for j, qc in enumerate(raw_solutions[cur_i]):
                best_prev = min(
                    cost[-1],
                    key=lambda jp: float(np.max(np.abs(qc - raw_solutions[prev_i][jp]))),
                )
                cur_cost[j] = cost[-1][best_prev] + float(
                    np.max(np.abs(qc - raw_solutions[prev_i][best_prev]))
                )
                cur_back[j] = best_prev
        cost.append(cur_cost)
        back.append(cur_back)

    chosen = {}
    last_best = min(cost[-1], key=lambda j: cost[-1][j])
    chosen[order[-1]] = last_best
    for k in range(len(order) - 1, 0, -1):
        chosen[order[k - 1]] = back[k][chosen[order[k]]]

    q_list: List[np.ndarray] = []
    valid_indices: List[int] = []
    pos_err: List[float] = []
    rot_err: List[float] = []
    for i in usable:
        q = np.asarray(raw_solutions[i][chosen[i]], dtype=float).copy()
        if q_list:
            prev = q_list[-1]
            for j in range(len(q)):
                if span[j] >= 2.0 * np.pi - 1e-6:
                    q[j] = prev[j] + (q[j] - prev[j] + np.pi) % (2.0 * np.pi) - np.pi
            q = chain.clamp(q)
        q_list.append(q)
        valid_indices.append(i)
        pos_err.append(0.0)
        rot_err.append(0.0)

    if len(q_list) < 2:
        raise ValueError(f"IK produced only {len(q_list)} valid waypoints out of {n}")

    q_arr = np.array(q_list, dtype=float)
    valid_idx = np.asarray(valid_indices, dtype=int)
    tip_points = poses_base[valid_idx][:, :3, 3]
    distances = np.linalg.norm(np.diff(tip_points, axis=0), axis=1)
    times = _time_parameterize_joint_safe(
        q_arr,
        distances=distances,
        cart_speed=speed_m_s,
        cart_accel=acceleration_m_s2,
        velocity_limits=velocity_limits,
        acceleration_limits=acceleration_limits,
        scale=0.8,
        min_dt=min_dt,
    )

    waypoints: List[Waypoint] = []
    for k, i in enumerate(valid_indices):
        sp = None
        if surface_points is not None:
            sp = [float(v) for v in surface_points[i]]
        waypoints.append(
            Waypoint(
                time_s=float(times[k]),
                positions=[float(v) for v in q_arr[k]],
                surface_point=sp,
                tool_pose=poses_base[i].tolist(),
            )
        )

    meta = {
        "waypoint_count": len(waypoints),
        "input_pose_count": n,
        "duration_s": float(times[-1]) if len(times) else 0.0,
        "path_length_m": float(np.sum(distances)),
        "max_position_error_mm": float(np.max(pos_err) * 1000.0) if pos_err else 0.0,
        "max_orientation_error_deg": float(np.rad2deg(np.max(rot_err))) if rot_err else 0.0,
        "ik_failures": failures,
        "model_to_base": T_base.tolist(),
    }
    trajectory = Trajectory(joint_names=chain.joint_names, waypoints=waypoints, meta=meta)
    return trajectory, meta


def validate_trajectory(
    chain: Chain,
    trajectory: Trajectory,
    max_velocity_scale: float = 1.0,
    max_acceleration_scale: float = 1.0,
) -> Dict:
    """Check limits, continuity and singular margin; returns a compact report."""
    q = trajectory.positions()
    t = np.array([w.time_s for w in trajectory.waypoints], dtype=float)
    report = {
        "waypoint_count": len(q),
        "duration_s": float(t[-1]) if len(t) else 0.0,
        "limit_violations": [],
        "velocity_violations": [],
        "acceleration_violations": [],
        "max_joint_step_deg": 0.0,
        "max_velocity_fraction": 0.0,
        "max_acceleration_fraction": 0.0,
        "min_manipulability": None,
        "passed": True,
    }
    if len(q) < 2:
        report["passed"] = False
        report["limit_violations"].append("fewer than 2 waypoints")
        return report

    for j, name in enumerate(chain.joint_names):
        lo, hi = chain.lower[j], chain.upper[j]
        viol = np.flatnonzero((q[:, j] < lo - 1e-9) | (q[:, j] > hi + 1e-9))
        if viol.size:
            report["limit_violations"].append(
                {"joint": name, "count": int(viol.size), "lower": lo, "upper": hi,
                 "min": float(q[:, j].min()), "max": float(q[:, j].max())}
            )

    dt = np.diff(t)
    dt_safe = np.maximum(dt, 1e-6)
    dq = np.diff(q, axis=0) / dt_safe[:, None]
    vlim = chain.velocity * max_velocity_scale
    for j, name in enumerate(chain.joint_names):
        if vlim[j] > 0:
            frac = float(np.max(np.abs(dq[:, j])) / vlim[j])
            report["max_velocity_fraction"] = max(report["max_velocity_fraction"], frac)
            if frac > 1.0:
                report["velocity_violations"].append({"joint": name, "fraction": frac})

    if len(q) > 2:
        dqdt = np.diff(dq, axis=0) / np.maximum(dt[:-1], 1e-6)[:, None]
        alim = 2.0 * vlim  # conservative default
        for j, name in enumerate(chain.joint_names):
            if alim[j] > 0:
                frac = float(np.max(np.abs(dqdt[:, j])) / alim[j])
                report["max_acceleration_fraction"] = max(report["max_acceleration_fraction"], frac)
                if frac > 1.0:
                    report["acceleration_violations"].append({"joint": name, "fraction": frac})

    report["max_joint_step_deg"] = float(np.rad2deg(np.max(np.abs(np.diff(q, axis=0)))))

    manip = []
    from .ik import _jacobian

    for i in range(0, len(q)):
        J = _jacobian(chain, q[i])
        manip.append(float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0))))
    report["min_manipulability"] = float(np.min(manip)) if manip else None

    report["passed"] = (
        not report["limit_violations"]
        and not report["velocity_violations"]
        and not report["acceleration_violations"]
    )
    return report


def plan_task(
    urdf_path,
    mesh_vertices,
    mesh_normals,
    task: TaskSpec,
    model_to_base: np.ndarray,
    tip_link: str = "Link6",
    reference_q=None,
    restarts: int = 24,
    seed: int = 0,
    tcp_in_link6: Optional[np.ndarray] = None,
    velocity_limits: Optional[np.ndarray] = None,
    acceleration_limits: Optional[np.ndarray] = None,
) -> Tuple[Trajectory, SurfacePath, Dict]:
    """Full offline pipeline: region -> surface path -> tool poses -> IK -> trajectory.

    When ``tip_link`` is the robot's last link (no tool link in the URDF) pass
    ``tcp_in_link6`` so the IK targets the tool tip instead of the flange.
    """
    chain = load_robot(urdf_path, tip_link)

    if task.pattern == "arch":
        arch = task.arch
        surface_path = arch_surface_path(
            mesh_vertices,
            mesh_normals,
            spacing_m=task.spacing_m,
            up_direction=arch.get("up_direction", [0.0, 1.0, 0.0]),
            across_direction=arch.get("across_direction"),
            poly_degree=int(arch.get("poly_degree", 4)),
            u_range=arch.get("u_range"),
            lateral_offset_m=float(arch.get("lateral_offset_m", 0.0)),
            band_radius_m=float(arch.get("band_radius_m", 0.004)),
            angular_bins=int(arch.get("angular_bins", 180)),
            height_band_m=float(arch.get("height_band_m", 0.012)),
            min_face_dot=float(arch.get("min_face_dot", 0.4)),
            trim_fraction=float(arch.get("trim_fraction", 0.0)),
        )
        # The arch curve is smooth, so letting X follow the tangent gives a
        # natural "walking along the arch" wrist motion without roll noise.
        x_reference = task.x_reference
    elif task.pattern == "raster":
        mask = select_region(mesh_vertices, mesh_normals, task.region)
        surface_path = raster_surface_path(
            mesh_vertices,
            mesh_normals,
            mask,
            spacing_m=task.spacing_m,
            row_spacing_m=task.row_spacing_m,
        )
        x_reference = task.x_reference
    else:
        raise ValueError(f"unknown task pattern: {task.pattern}")

    if task.max_tilt_deg is not None:
        # Keep the wrist near a nominal orientation instead of following every
        # local normal. On a curved arch the raw normals rotate by ~180 deg,
        # which pushes the IK across a branch switch halfway along the path.
        surface_path.normals = limit_tilt(
            surface_path.normals, task.tilt_reference, task.max_tilt_deg
        )

    if task.fixed_orientation_deg is not None:
        poses_model, tips_model = fixed_orientation_tool_poses(
            surface_path,
            standoff_m=task.standoff_m,
            rotation_base_deg=task.fixed_orientation_deg,
            approach_clearance_m=task.approach_clearance_m,
            model_to_base=model_to_base,
        )
    elif task.normal_approach:
        # Tip poses that follow each surface normal; compile_trajectory applies
        # the TCP inverse to get flange poses.
        poses_model, tips_model = normal_approach_tool_poses(
            surface_path,
            standoff_m=task.standoff_m,
            approach_clearance_m=task.approach_clearance_m,
        )
    elif task.fixed_rod_direction is not None:
        # The brush keeps a fixed axis in the base frame; only its yaw follows
        # the travel direction. This matches the taught poses in the real cell.
        tip_offset = np.linalg.inv(tcp_in_link6) if tcp_in_link6 is not None else np.eye(4)
        rod_l6 = tip_offset[:3, :3].T @ se3.normalize(np.asarray(
            cfg_rod := task.fixed_rod_direction, dtype=float
        )) if False else None
        # The rod in Link6 is the direction from the flange origin to the TCP.
        rod_in_link6 = None
        if tcp_in_link6 is not None:
            off = np.asarray(tcp_in_link6)[:3, 3]
            if np.linalg.norm(off) > 1e-9:
                rod_in_link6 = np.asarray(tcp_in_link6)[:3, :3] @ (off / np.linalg.norm(off))
        poses_model, tips_model = fixed_axis_tool_poses(
            surface_path,
            standoff_m=task.standoff_m,
            rod_direction_base=task.fixed_rod_direction,
            rod_in_link6=rod_in_link6,
            approach_clearance_m=task.approach_clearance_m,
            model_to_base=model_to_base,
        )
    else:
        poses_model, tips_model = tool_poses_from_surface(
            surface_path,
            standoff_m=task.standoff_m,
            roll_deg=task.roll_deg,
            approach_clearance_m=task.approach_clearance_m,
            x_reference=x_reference,
        )
    if task.fixed_rod_direction is None:
        poses_model = densify_poses(
            poses_model,
            max_translation_step_m=task.max_translation_step_m,
            max_rotation_step_deg=task.max_rotation_step_deg,
        )
    else:
        # Fixed-orientation brushing: keep the sparse arch waypoints. They are
        # continuous in joint space (verified: max step 1.7 deg); interpolating
        # in Cartesian space first creates intermediate poses that sit on other
        # IK branches. compile_trajectory resamples in joint space afterwards.
        pass
    tips_model = poses_model[:, :3, 3].copy()
    trajectory, meta = compile_trajectory(
        chain,
        poses_model,
        model_to_base,
        speed_m_s=task.speed_m_s,
        acceleration_m_s2=task.acceleration_m_s2,
        surface_points=tips_model,
        reference_q=reference_q,
        restarts=restarts,
        seed=seed,
        min_dt=0.05,
        tip_poses=(task.fixed_rod_direction is None and tip_link == "tcp_link"),
        # fixed_axis_tool_poses builds true flange poses; the fixed-orientation
        # builder returns tip poses and still needs the TCP inverse.
        poses_are_flange=(task.fixed_rod_direction is not None),
        tcp_in_link6=tcp_in_link6,
        velocity_limits=velocity_limits,
        acceleration_limits=acceleration_limits,
    )
    report = validate_trajectory(chain, trajectory)
    meta.update(
        {
            "pattern": task.pattern,
            "surface_path_points": int(len(surface_path.points)),
            "surface_path_length_m": path_length(surface_path.points),
            "validation": report,
        }
    )
    if task.pattern == "raster":
        meta["region_vertex_count"] = int(len(np.unique(surface_path.source_indices)))
    trajectory.meta = meta
    return trajectory, surface_path, meta
