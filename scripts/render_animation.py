"""Render an animated preview of a planned trajectory, with no simulator needed.

Produces:
    <out>/robot_preview.mp4   (or .gif if ffmpeg is unavailable)
    <out>/keyframes/*.png     a handful of stills

This is a kinematic playback: link poses come from the same URDF chain used for
planning, the tooth mesh is drawn as a point cloud, and the brush is drawn as a
line from the flange to the tip. It answers "does the motion look right" before
touching Isaac Sim or a real robot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, PillowWriter  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from dobot_pathgen.config import Config  # noqa: E402
from dobot_pathgen.mesh_io import read_ply, read_stl, vertex_normals  # noqa: E402
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


def _load_link_triangles(cfg):
    """Load each link mesh, decimated to a stable triangle budget."""
    mesh_dir = cfg.robot.urdf.parent.parent / "meshes" / "cr5"
    tris = {}
    for link, name in LINK_MESHES.items():
        path = mesh_dir / name
        if not path.exists():
            continue
        verts, faces, _ = read_stl(path)
        target = 700
        step = max(1, len(faces) // target)
        faces = faces[::step]
        tris[link] = (verts, faces)
        print(f"  {link:10s} {len(verts):6d} verts {len(faces):5d} tris")
    return tris


def _transform(verts, faces, T):
    v = verts @ T[:3, :3].T + T[:3, 3]
    return v[faces]


def _sample_trajectory(trajectory, fps):
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
    parser.add_argument("--out", default=str(ROOT / "out/animation"))
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--speed", type=float, default=4.0,
                        help="playback speed multiplier (4 = 4x faster than planned)")
    parser.add_argument("--keyframes", type=int, default=6)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    trajectory = Trajectory.load(args.trajectory)
    chain = build_chain(cfg.robot.urdf, cfg.robot.tip_link)
    tris = _load_link_triangles(cfg)

    vertices, faces, normals, _ = read_ply(cfg.scene.model_mesh)
    vertices = vertices * cfg.scene.scale
    vertices_b = vertices @ cfg.scene.model_to_base[:3, :3].T + cfg.scene.model_to_base[:3, 3]
    if normals is None:
        normals = vertex_normals(vertices, faces, None)
    cloud = vertices_b[::6]

    tip_offset = np.eye(4)
    tip_offset[:3, 3] = cfg.tool.tcp_offset_m

    # Precompute the tool tip path for the trail.
    tip_path = []
    for w in trajectory.waypoints:
        T = chain.fk(w.positions)
        tip_path.append((T @ tip_offset)[:3, 3])
    tip_path = np.array(tip_path)

    grid, samples = _sample_trajectory(trajectory, args.fps)
    if args.speed > 1:
        grid = grid / args.speed
    print(f"frames: {len(samples)}  span: {grid[-1]:.1f} s  speed: {args.speed}x")

    all_pts = [cloud, np.array([[0, 0, 0]])]
    for link, (v, f) in tris.items():
        all_pts.append(v)
    all_pts = np.vstack(all_pts)
    lo = all_pts.min(axis=0) - 0.05
    hi = all_pts.max(axis=0) + 0.05

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 8), dpi=110)
    ax = fig.add_subplot(111, projection="3d")

    def draw_frame(i):
        ax.clear()
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect((hi - lo))
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.view_init(elev=22, azim=-58)
        ax.set_title(
            f"CR5A arch walk   t={grid[i]:5.1f}s   "
            f"wp {min(int(i / max(len(grid) - 1, 1) * (len(trajectory.waypoints) - 1)), len(trajectory.waypoints) - 1)}"
            f"/{len(trajectory.waypoints) - 1}",
            fontsize=10,
        )

        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=0.6, c="#9aa3ad", alpha=0.55)

        frames = chain.fk_all(samples[i])
        for k, link in enumerate(LINK_MESHES):
            if link not in tris or k >= len(frames):
                continue
            verts, link_faces = tris[link]
            tris_world = _transform(verts, link_faces, frames[k])
            color = "#4c78a8" if k > 0 else "#2b2b2b"
            # Opaque faces: matplotlib's 3D alpha blending mis-sorts large
            # collections and the arm looks like scattered shards.
            ax.add_collection3d(
                Poly3DCollection(
                    tris_world,
                    facecolors=color,
                    linewidths=0,
                    alpha=1.0,
                    shade=True,
                )
            )

        T = chain.fk(samples[i])
        tip = (T @ tip_offset)[:3, 3]
        flange = T[:3, 3]
        ax.plot(
            [flange[0], tip[0]], [flange[1], tip[1]], [flange[2], tip[2]],
            color="#e4572e", linewidth=4, solid_capstyle="round",
        )
        ax.scatter([tip[0]], [tip[1]], [tip[2]], c="#ffd166", s=45, depthshade=False)

        travelled = tip_path[: max(2, int((i / max(len(grid) - 1, 1)) * len(tip_path)))]
        ax.plot(travelled[:, 0], travelled[:, 1], travelled[:, 2],
                color="#7bd389", linewidth=1.6, alpha=0.95)

        ax.scatter([0], [0], [0], c="#222222", s=60, marker="s", depthshade=False)

    # Keyframes
    kf_dir = out_dir / "keyframes"
    kf_dir.mkdir(exist_ok=True)
    indices = np.linspace(0, len(samples) - 1, max(1, args.keyframes)).astype(int)
    for n, i in enumerate(indices):
        draw_frame(i)
        path = kf_dir / f"frame_{n:02d}_t{grid[i]:05.1f}.png"
        fig.savefig(path, bbox_inches="tight")
        print(f"  keyframe {path.name}")
    plt.close(fig)

    # Animation
    fig = plt.figure(figsize=(9, 8), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    mp4 = out_dir / "robot_preview.mp4"
    try:
        # Use the ffmpeg bundled with imageio-ffmpeg when a system one is absent.
        import imageio_ffmpeg

        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = FFMpegWriter(fps=int(args.fps / max(args.speed, 1)), bitrate=2400)
        with writer.saving(fig, str(mp4), dpi=100):
            for i in range(len(samples)):
                draw_frame(i)
                writer.grab_frame()
                if i % 25 == 0:
                    print(f"  frame {i}/{len(samples)}")
        print(f"animation: {mp4}")
    except Exception as exc:
        print(f"ffmpeg unavailable ({exc}); writing GIF instead")
        gif = out_dir / "robot_preview.gif"
        writer = PillowWriter(fps=int(max(args.fps / max(args.speed, 1), 5)))
        stride = max(1, len(samples) // 150)
        with writer.saving(fig, str(gif), dpi=90):
            for i in range(0, len(samples), stride):
                draw_frame(i)
                writer.grab_frame()
        print(f"animation: {gif}")
    plt.close(fig)

    np.save(out_dir / "tip_path.npy", tip_path)
    print(f"tip path: {out_dir / 'tip_path.npy'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
