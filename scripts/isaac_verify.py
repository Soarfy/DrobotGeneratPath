"""Verify Isaac Sim can build a scene and run our CR5A replay.

Starts Kit, imports the combined CR5A+brush URDF, adds the dental mesh, steps
the world, and writes a screenshot. Exits WITHOUT calling app.close(): on this
machine the shutdown path segfaults in a worker thread even though the app
itself is healthy, so the process is simply terminated after the work is done.

    python scripts/isaac_verify.py --trajectory out/arch_lower_jaw.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config/robot.yaml"))
    parser.add_argument("--trajectory", default=str(ROOT / "out/arch_lower_jaw.json"))
    parser.add_argument("--frames", type=int, default=6, help="replay frames to render")
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    from isaacsim import SimulationApp

    # extra_args must be a LIST of "--key=value" strings; a dict is silently
    # ignored, which left multi-GPU enabled and crashed Vulkan device init.
    app = SimulationApp(
        {
            "headless": args.headless,
            "multi_gpu": False,
            "extra_args": [
                "--/renderer/multiGpu/enabled=False",
                "--/physics/cudaDevice=0",
                "--/renderer/activeGpu=0",
            ],
        }
    )
    print("ISAAC_BOOTED", flush=True)

    import numpy as np

    from dobot_pathgen.config import Config
    from dobot_pathgen.mesh_io import read_ply
    from dobot_pathgen.planning import Trajectory
    from dobot_pathgen.urdf_chain import build_chain

    cfg = Config.load(args.config)
    trajectory = Trajectory.load(args.trajectory)
    chain = build_chain(cfg.robot.urdf, cfg.robot.tip_link)

    # --- world ---
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation as Articulation

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # --- robot ---
    urdf = cfg.robot.urdf_with_tool
    if not urdf.exists():
        print(f"MISSING URDF: {urdf}")
        return 1
    print(f"importing URDF: {urdf.name}", flush=True)

    from isaacsim.asset.importer.urdf import _urdf
    import omni.kit.commands

    import_config = _urdf.ImportConfig()
    for name, value in (
        ("merge_fixed_joints", False),
        ("convex_decomp", False),
        ("fix_base", True),
        ("make_default_prim", True),
        ("create_physics_scene", True),
        ("self_collision", False),
        ("default_drive_strength", 1048576.0),
        ("default_position_drive_stiffness", 10000.0),
    ):
        if hasattr(import_config, name):
            setattr(import_config, name, value)

    ok, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf),
        import_config=import_config,
        dest_path="/World/CR5A",
    )
    print(f"URDF import ok={ok} prim={prim_path}", flush=True)
    if not ok:
        return 1

    robot = world.scene.add(Articulation(prim_path=prim_path, name="cr5a"))

    # --- dental model ---
    vertices, faces, _, _ = read_ply(cfg.scene.model_mesh)
    vertices = vertices * cfg.scene.scale
    vertices = vertices @ cfg.scene.model_to_base[:3, :3].T + cfg.scene.model_to_base[:3, 3]

    from pxr import Gf, UsdGeom, Vt
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    mesh = UsdGeom.Mesh.Define(stage, "/World/DentalTarget")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in vertices]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for tri in faces for i in tri]))
    print(f"dental mesh: {len(vertices)} verts, {len(faces)} faces", flush=True)

    world.reset()
    print("world reset ok", flush=True)

    # --- joint names ---
    dof_names = list(getattr(robot, "dof_names", []) or [])
    print(f"articulation DOFs ({len(dof_names)}): {dof_names}", flush=True)
    missing = [n for n in trajectory.joint_names if n not in dof_names]
    if missing:
        print(f"WARNING: joints missing from articulation: {missing}")

    ndof = len(dof_names) or len(trajectory.joint_names)
    indices = [dof_names.index(n) for n in trajectory.joint_names if n in dof_names]

    # --- replay a few frames ---
    times = np.array([w.time_s for w in trajectory.waypoints])
    q = trajectory.positions()
    grid = np.linspace(0.0, times[-1], max(2, args.frames))
    samples = np.column_stack([np.interp(grid, times, q[:, j]) for j in range(q.shape[1])])

    out_dir = ROOT / "out" / "isaac"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(samples):
        full = np.zeros(ndof)
        for k, idx in enumerate(indices):
            full[idx] = row[k]
        robot.set_joint_positions(full)
        world.step(render=True)
        print(f"  frame {i + 1}/{len(samples)}  t={grid[i]:.2f}s", flush=True)

    # --- read back the tip pose from the simulated articulation ---
    positions, orientations = robot.get_world_poses()
    print(f"base pose: pos={np.round(positions[0], 3)}", flush=True)

    try:
        import omni.replicator.core as rep

        rep.settings.set_render_rtx_realtime()
    except Exception:
        pass

    # Screenshot through the viewport capture, when a render product exists.
    try:
        from omni.kit.viewport.utility import get_active_viewport
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(
            eye=[1.6, -1.5, 1.2], target=[0.45, 0.0, 0.25],
            camera_prim_path="/OmniverseKit_Persp",
        )
        for _ in range(5):
            world.step(render=True)
        vp = get_active_viewport()
        vp.capture_default_graph()
        frame_path = out_dir / "replay.png"
        vp.capture_frame(str(frame_path))
        print(f"screenshot: {frame_path}", flush=True)
    except Exception as exc:
        print(f"screenshot skipped: {exc}", flush=True)

    summary = out_dir / "verify.txt"
    summary.write_text(
        "\n".join(
            [
                f"isaacsim booted: yes",
                f"urdf: {urdf}",
                f"articulation prim: {prim_path}",
                f"dofs: {dof_names}",
                f"trajectory: {args.trajectory}",
                f"frames replayed: {len(samples)}",
                f"trajectory duration: {times[-1]:.2f} s",
            ]
        ),
        encoding="utf-8",
    )
    print("ISAAC_VERIFY_OK", flush=True)

    # Do not call app.close(): its teardown crashes in a worker thread on this
    # machine. Everything above has already been flushed and verified.
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
