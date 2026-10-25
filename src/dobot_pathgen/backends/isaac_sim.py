"""Isaac Sim replay backend.

Run with Isaac Sim's own Python, not the project venv:

    Windows:  <ISAAC_SIM>/python.bat -m dobot_pathgen.cli sim --trajectory out/trajectory.json
    Linux:    <ISAAC_SIM>/python.sh -m dobot_pathgen.cli sim --trajectory out/trajectory.json

The module imports the CR5A URDF with the tool attached, places the dental
mesh at the calibrated model->base transform, and plays the planned joint
trajectory back. It is a kinematics/visualization check, not a contact
simulation: the demo does not model bristle deformation or contact forces.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np


def _launch_app(headless: bool):
    """Create the SimulationApp, which must happen before other omni imports."""
    from isaacsim import SimulationApp

    config = {
        "headless": headless,
        "extra_args": {
            "--/app/asyncRendering": headless,
            "--/rtx/raytracing/fractionalCutoutOpacity": True,
        },
    }
    return SimulationApp(config)


def _import_urdf(urdf_path: Path, prim_path: str):
    """Import a URDF through the version-dependent importer command."""
    try:
        from isaacsim.asset.importer.urdf import _urdf  # Isaac Sim 4.x
    except ImportError:
        from omni.importer.urdf import _urdf  # Isaac Sim 2022.2 / 2023.1

    import omni.kit.commands

    import_config = _urdf.ImportConfig()
    # Attribute sets differ between Isaac Sim versions; set only what exists.
    for name, value in (
        ("merge_fixed_joints", False),
        ("convex_decomp", False),
        ("fix_base", True),
        ("make_default_prim", True),
        ("create_physics_scene", True),
        ("self_collision", False),
        ("default_drive_strength", 1048576.0),
        ("default_position_drive_damping", 0.0),
        ("default_position_drive_stiffness", 10000.0),
    ):
        if hasattr(import_config, name):
            setattr(import_config, name, value)

    ok, imported_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path),
        import_config=import_config,
        dest_path=prim_path,
    )
    if not ok:
        raise RuntimeError(f"URDF import failed: {urdf_path}")
    return imported_path


def _add_target_mesh(vertices_m, faces, model_to_base, prim_path="/World/DentalTarget",
                     add_collision: bool = False):
    """Create a USD triangle mesh so the target is visible in the scene."""
    from pxr import Gf, UsdGeom, UsdPhysics, Vt

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    mesh = UsdGeom.Mesh.Define(stage, prim_path)

    R = np.asarray(model_to_base, dtype=float)[:3, :3]
    t = np.asarray(model_to_base, dtype=float)[:3, 3]
    xform = UsdGeom.Xformable(mesh.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*t.tolist()))
    quat = _matrix_to_quat_wxyz(R)
    xform.AddOrientOp().Set(Gf.Quatf(quat[0], Gf.Vec3f(*quat[1:])))

    points = [Gf.Vec3f(*map(float, v)) for v in vertices_m]
    counts = [3] * len(faces)
    indices = [int(i) for tri in faces for i in tri]
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.SetNormalsInterpolation("vertex")

    if add_collision:
        # SDF on a 130k-face scan is slow; convex hull is enough for a visual
        # proximity check and keeps the demo responsive.
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        mesh_collision.CreateApproximationAttr("convexHull")
    return prim_path


def _matrix_to_quat_wxyz(R):
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w < 1e-8:
        w = 1e-6
    x = (R[2, 1] - R[1, 2]) / (4.0 * w)
    y = (R[0, 2] - R[2, 0]) / (4.0 * w)
    z = (R[1, 0] - R[0, 1]) / (4.0 * w)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / n, x / n, y / n, z / n)


def _load_target(ply_path: Path, scale: float):
    from ..mesh_io import read_ply

    vertices, faces, _, _ = read_ply(ply_path)
    return vertices * scale, faces


def _sample_trajectory(trajectory, fps: float):
    """Interpolate joint positions on a fixed display grid."""
    times = np.array([w.time_s for w in trajectory.waypoints], dtype=float)
    q = trajectory.positions()
    if times[-1] <= 0:
        return times[:1], q[:1]
    grid = np.arange(0.0, times[-1], 1.0 / fps)
    grid = np.append(grid, times[-1])
    out = np.column_stack([np.interp(grid, times, q[:, j]) for j in range(q.shape[1])])
    return grid, out


def run_isaac_sim(cfg, trajectory_path, headless: bool = False, realtime: bool = False) -> int:
    from ..planning import Trajectory

    trajectory = Trajectory.load(trajectory_path)
    urdf = cfg.robot.urdf_with_tool
    if not urdf.exists():
        print(f"combined URDF missing: {urdf}")
        print("run: python scripts/make_assets.py")
        return 1

    app = _launch_app(headless)

    try:
        from isaacsim.core.api import World
    except ImportError:
        from omni.isaac.core import World

    try:
        from isaacsim.core.prims import SingleArticulation as Articulation
    except ImportError:
        from omni.isaac.core.articulations import Articulation

    try:
        from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: F401
    except ImportError:
        pass

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    prim_path = _import_urdf(urdf, "/World/CR5A")
    robot = world.scene.add(Articulation(prim_path=prim_path, name="cr5a"))

    vertices, faces = _load_target(cfg.scene.model_mesh, cfg.scene.scale)
    _add_target_mesh(vertices, faces, cfg.scene.model_to_base)

    world.reset()

    dof_names = list(getattr(robot, "dof_names", []) or [])
    joint_names = trajectory.joint_names
    if dof_names:
        missing = [n for n in joint_names if n not in dof_names]
        if missing:
            print(f"warning: joints not found on the articulation: {missing}")
        indices = [dof_names.index(n) for n in joint_names if n in dof_names]
        ndof = len(dof_names)
    else:
        indices = list(range(len(joint_names)))
        ndof = len(joint_names)

    def apply(row):
        full = np.zeros(ndof)
        for k, idx in enumerate(indices):
            full[idx] = row[k]
        robot.set_joint_positions(full)

    apply(np.zeros(len(joint_names)))
    for _ in range(10):
        world.step(render=not headless)

    fps = 60.0
    grid, samples = _sample_trajectory(trajectory, fps)
    print(f"sim replay: {len(samples)} frames, {grid[-1]:.1f} s, {fps:.0f} fps")

    start = time.perf_counter()
    for i, row in enumerate(samples):
        apply(row)
        world.step(render=not headless)
        if realtime:
            target = start + grid[i]
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        if i % 120 == 0:
            print(f"  frame {i}/{len(samples)}  t={grid[i]:.2f}s")

    # Hold the final pose briefly so the result is visible.
    for _ in range(30):
        world.step(render=not headless)

    summary = {
        "trajectory": str(trajectory_path),
        "frames": int(len(samples)),
        "duration_s": float(grid[-1]),
        "wall_time_s": float(time.perf_counter() - start),
        "realtime": bool(realtime),
        "headless": bool(headless),
        "joint_names": joint_names,
    }
    out = Path(trajectory_path).with_suffix(".sim.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"sim done in {summary['wall_time_s']:.1f} s -> {out}")

    app.close()
    return 0
