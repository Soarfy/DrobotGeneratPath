"""High-quality offscreen render of a planned trajectory using PyVista/VTK.

Produces real shaded meshes with a depth buffer (unlike matplotlib's 3D
projection), so the arm looks like a solid robot and the brush is a solid rod.

    python scripts/render_vtk.py --trajectory out/arch_lower_jaw.json

Outputs into out/vtk_preview/:
    frames/frame_XXXX.png     one PNG per rendered frame
    keyframe_*.png            a few evenly spaced stills
    robot_path.mp4            animation (needs imageio-ffmpeg)
    tip_path.ply              the tool-tip centreline for inspection
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pyvista as pv  # noqa: E402

from dobot_pathgen.config import Config  # noqa: E402
from dobot_pathgen.mesh_io import read_ply, read_stl, write_ply_points  # noqa: E402
from dobot_pathgen.planning import Trajectory  # noqa: E402
from dobot_pathgen.urdf_chain import build_chain  # noqa: E402


LINK_MESHES = {
    "base_link": "base_link.STL",
    "Link1": "J1.STL",
    "Link2": "J2.STL",
    "Link3": "J3.STL",
    "Link4": "J4.STL",
    "Link5": "J5.STL",
    "Link6": "J6.STL",
}


def load_robot_meshes(cfg):
    mesh_dir = cfg.robot.urdf.parent.parent / "meshes" / "cr5"
    meshes = {}
    for link, name in LINK_MESHES.items():
        path = mesh_dir / name
        if not path.exists():
            continue
        verts, faces, _ = read_stl(path)
        # PyVista wants a faces array of [3, i, j, k, 3, ...]
        faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
        meshes[link] = pv.PolyData(verts, faces_pv)
        meshes[link].clean(inplace=True)
    return meshes


def sample_trajectory(trajectory, fps):
    times = np.array([w.time_s for w in trajectory.waypoints])
    q = trajectory.positions()
    if times[-1] <= 0:
        return times[:1], q[:1]
    grid = np.arange(0.0, times[-1] + 1e-9, 1.0 / fps)
    if grid[-1] < times[-1]:
        grid = np.append(grid, times[-1])
    out = np.column_stack([np.interp(grid, times, q[:, j]) for j in range(q.shape[1])])
    return grid, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config/robot.yaml"))
    parser.add_argument("--trajectory", default=str(ROOT / "out/arch_lower_jaw.json"))
    parser.add_argument("--out", default=str(ROOT / "out/vtk_preview"))
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--speed", type=float, default=3.0)
    parser.add_argument("--frames", type=int, default=90,
                        help="number of frames to render for the video")
    parser.add_argument("--size", default="1280x960")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    trajectory = Trajectory.load(args.trajectory)
    chain = build_chain(cfg.robot.urdf, cfg.robot.tip_link)

    print("loading meshes ...")
    robot = load_robot_meshes(cfg)
    print(f"  links: {', '.join(robot)}")

    vertices, faces, normals, _ = read_ply(cfg.scene.model_mesh)
    verts_m = vertices * cfg.scene.scale
    verts_b = verts_m @ cfg.scene.model_to_base[:3, :3].T + cfg.scene.model_to_base[:3, 3]
    cloud = pv.PolyData(verts_b[::3])

    tcp = np.eye(4)
    tcp[:3, 3] = cfg.tool.tcp_offset_m
    tip_path = np.array([(chain.fk(w.positions) @ tcp)[:3, 3] for w in trajectory.waypoints])

    # Resample so the render is bounded in time regardless of trajectory length.
    times = np.array([w.time_s for w in trajectory.waypoints])
    q_all = trajectory.positions()
    n = min(args.frames, max(2, int(times[-1] * args.fps / args.speed)))
    grid = np.linspace(0.0, times[-1], n)
    samples = np.column_stack([np.interp(grid, times, q_all[:, j]) for j in range(q_all.shape[1])])

    out_dir = Path(args.out)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    width, height = (int(v) for v in args.size.split("x"))

    pl = pv.Plotter(off_screen=True, window_size=(width, height))
    pl.set_background("#1a1d21")
    pl.add_light(pv.Light(position=(2.0, -2.0, 2.5), intensity=0.9, light_type="scene light"))
    pl.add_light(pv.Light(position=(-1.5, 1.5, 1.5), intensity=0.45, light_type="scene light"))

    # Static scene. The scan carries per-vertex colour; the mesh is small
    # enough that it needs no decimation to stay readable.
    target_mesh = pv.PolyData(
        verts_b, np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    )
    if normals is not None:
        target_mesh.point_data["normals"] = normals
    pl.add_mesh(
        target_mesh,
        color="#d9c6ae",
        smooth_shading=True,
        specular=0.15,
        show_edges=False,
        ambient=0.35,
        diffuse=0.75,
    )

    tip_trail = pv.lines_from_points(tip_path) if len(tip_path) > 1 else None
    if tip_trail is not None:
        pl.add_mesh(tip_trail, color="#4caf50", line_width=3)

    actors = {}
    for link, mesh in robot.items():
        color = "#2f3a45" if link == "base_link" else "#5b8ac4"
        actors[link] = pl.add_mesh(mesh, color=color, smooth_shading=True, specular=0.4)

    # Brush: a line from flange to tip, plus a sphere at the tip.
    brush_line = None
    tip_sphere = None

    # Fixed camera derived from the reachable workspace plus the model, so the
    # framing is identical every frame and the motion reads as motion.
    scene_lo = np.minimum(verts_b.min(axis=0), tip_path.min(axis=0)) - 0.10
    scene_hi = np.maximum(verts_b.max(axis=0), tip_path.max(axis=0)) + 0.10
    scene_lo[0] = min(scene_lo[0], -0.15)
    scene_hi[0] = max(scene_hi[0], 0.45)
    centre = (scene_lo + scene_hi) * 0.5
    span = float(np.linalg.norm(scene_hi - scene_lo))
    eye = centre + span * np.array([0.72, -0.95, 0.62])
    pl.camera.up = (0, 0, 1)
    pl.camera.focal_point = tuple(centre)
    pl.camera.position = tuple(eye)
    pl.camera.view_angle = 30

    print(f"rendering {n} frames at {width}x{height} ...")
    for i in range(n):
        frames = chain.fk_all(samples[i])
        for k, link in enumerate(LINK_MESHES):
            if link not in actors or k >= len(frames):
                continue
            T = frames[k]
            mesh = robot[link].copy()
            mesh.points = mesh.points @ T[:3, :3].T + T[:3, 3]
            actors[link].mapper.SetInputData(mesh)
            actors[link].mapper.Update()

        T = chain.fk(samples[i])
        flange = T[:3, 3]
        tip = (T @ tcp)[:3, 3]
        if brush_line is not None:
            pl.remove_actor(brush_line)
            pl.remove_actor(tip_sphere)
        # Rod as a tube so it reads as a solid stick.
        rod = pv.Line(flange, tip).tube(radius=float(cfg.tool.radius_m), n_sides=16)
        brush_line = pl.add_mesh(rod, color="#f2f4f7", smooth_shading=True)
        tip_sphere = pl.add_mesh(
            pv.Sphere(radius=0.012, center=tip), color="#ffb703"
        )

        pl.screenshot(str(frames_dir / f"frame_{i:04d}.png"), return_img=False)
        if i % 20 == 0:
            print(f"  frame {i}/{n}")

    if n:
        pl.screenshot(str(out_dir / "keyframe_first.png"))
        pl.close()

    write_ply_points(out_dir / "tip_path.ply", tip_path)
    print(f"frames   : {frames_dir}")
    print(f"tip path : {out_dir / 'tip_path.ply'}")

    try:
        import imageio.v2 as imageio

        files = sorted(frames_dir.glob("frame_*.png"))
        if files:
            mp4 = out_dir / "robot_path.mp4"
            writer = imageio.get_writer(
                str(mp4),
                format="FFMPEG",
                mode="I",
                fps=max(5, int(args.fps / args.speed)),
                codec="libx264",
                quality=7,
                macro_block_size=None,
            )
            for f in files:
                writer.append_data(imageio.imread(f))
            writer.close()
            print(f"video    : {mp4}")
    except Exception as exc:  # pragma: no cover
        print(f"video not written: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
