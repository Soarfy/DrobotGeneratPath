"""Place the brush mesh into the Link6 frame.

Two things are independently controllable here, because they are easy to
confuse and hard to fix by eye:

``--working-end``
    Which end of the brush points at the tooth. The scan is a dumbbell: one end
    has radius ~10.8 mm (the bristle head) and the other ~6.7 mm (the grip).
    The bristle head is the working end.

``--roll-deg``
    Rotation about the brush's own long axis, which sets which way the bristle
    face looks.

The result is written so that:
  * the working end sits at the TCP offset (0, +120, +150) mm in Link6,
  * the brush's long axis is Link6 +Y,
  * the body extends back along -Y toward the flange.

    python scripts/align_brush.py --working-end wide --roll-deg 180
    python scripts/align_brush.py --working-end narrow --roll-deg 0

Never hand-edit the STL: rerun this instead so the placement stays reproducible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dobot_pathgen.mesh_io import write_stl, write_ply_points
from dobot_pathgen import se3


SRC = Path(r"D:\CompanyProject\hand_eye_calibration\UsmileUi\defaultmodel\brushmodel\brushleap.ply")

# brushleap.ply stores colours on the FACE element, before the index list.
# A generic PLY reader mis-parses that and produces garbage triangles.
VERTEX_COUNT = 88447
FACE_COUNT = 174055


def read_brushleap(path: Path):
    data = path.read_bytes()
    marker = data.find(b"end_header")
    start = marker + len(b"end_header")
    if data[start:start + 2] == b"\r\n":
        start += 2
    elif data[start:start + 1] in (b"\n", b"\r"):
        start += 1

    vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    arr = np.frombuffer(data, dtype=vdt, count=VERTEX_COUNT, offset=start)
    verts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(float)

    off = start + VERTEX_COUNT * vdt.itemsize
    fdt = np.dtype([
        ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1"), ("n", "u1"),
        ("v0", "<i4"), ("v1", "<i4"), ("v2", "<i4"),
    ])
    fa = np.frombuffer(data, dtype=fdt, count=FACE_COUNT, offset=off)
    faces = np.stack([fa["v0"], fa["v1"], fa["v2"]], axis=1).astype(np.int64)
    return verts, faces


def analyse(verts):
    """Return (axis, end_wide, end_narrow, radius_wide, radius_narrow)."""
    centre = verts.mean(axis=0)
    _, _, vh = np.linalg.svd(verts - centre, full_matrices=False)
    axis = vh[0]
    proj = (verts - centre) @ axis
    perp = (verts - centre) - np.outer(proj, axis)
    radius = np.linalg.norm(perp, axis=1)

    lo, hi = float(proj.min()), float(proj.max())
    band = 0.15 * (hi - lo)
    near_lo = proj < lo + band
    near_hi = proj > hi - band
    r_lo = float(radius[near_lo].max())
    r_hi = float(radius[near_hi].max())

    end_lo = centre + axis * lo
    end_hi = centre + axis * hi
    if r_lo >= r_hi:
        return axis, end_lo, end_hi, r_lo, r_hi
    return axis, end_hi, end_lo, r_hi, r_lo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(SRC))
    parser.add_argument("--working-end", choices=["wide", "narrow"], default="wide",
                        help="which physical end goes to the tooth (wide = bristle head)")
    parser.add_argument("--roll-deg", type=float, default=180.0,
                        help="rotation about the brush's own long axis")
    parser.add_argument("--offset-mm", nargs=3, type=float, default=[0.0, 120.0, 150.0],
                        help="TCP offset in the Link6 frame")
    parser.add_argument("--out-stl", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    verts, faces = read_brushleap(Path(args.src))
    axis, end_wide, end_narrow, r_wide, r_narrow = analyse(verts)
    length = float(np.ptp((verts - verts.mean(axis=0)) @ axis))

    print(f"brush: {len(verts)} verts, {len(faces)} faces, {length:.1f} mm long")
    print(f"  wide   end radius {r_wide:5.2f} mm at {np.round(end_wide, 1)}")
    print(f"  narrow end radius {r_narrow:5.2f} mm at {np.round(end_narrow, 1)}")

    work_end = end_wide if args.working_end == "wide" else end_narrow
    # Direction from the working end toward the other end.
    body_dir = (end_narrow if args.working_end == "wide" else end_wide) - work_end
    body_dir /= np.linalg.norm(body_dir)

    # We want the body to extend along -Y in Link6, i.e. from the tip back
    # toward the flange.
    target = np.array([0.0, -1.0, 0.0])
    rotation = _rotation_between(body_dir, target)
    # The 180 deg roll is about the brush's own long axis, which after the
    # rotation above is Link6 Y.
    if abs(args.roll_deg) > 1e-9:
        rotation = se3.rotation_about_axis(target, np.deg2rad(args.roll_deg)) @ rotation

    offset = np.asarray(args.offset_mm, dtype=float) / 1000.0
    placed = ((verts - work_end) * 0.001) @ rotation.T + offset

    tip_err = float(np.linalg.norm(placed - offset, axis=1).min()) * 1000
    print(f"\nworking end ('{args.working_end}') at TCP, residual {tip_err:.2f} mm")
    print(f"  Link6 bbox m : {np.round(placed.min(axis=0), 4)} .. {np.round(placed.max(axis=0), 4)}")
    print(f"  roll         : {args.roll_deg:.0f} deg about the long axis")
    print(f"  body extends : from the tip toward {np.round(-target, 2)} (Link6 -Y)")

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    out_stl = Path(args.out_stl) if args.out_stl else (
        ROOT / "assets/robot/cr5a_with_tool/meshes/brushleap.stl"
    )
    write_stl(out_stl, placed, faces, binary=True)
    write_stl(ROOT / "assets/tool/brushleap/brushleap_link6.stl", placed, faces, binary=True)
    write_ply_points(ROOT / "assets/tool/brushleap/brushleap_link6_preview.ply", placed)
    print(f"\nwrote {out_stl}")
    print("run again with a different --working-end / --roll-deg to adjust")
    return 0


def _rotation_between(a, b):
    """Minimal rotation taking unit vector a onto unit vector b."""
    a = np.asarray(a, dtype=float) / np.linalg.norm(a)
    b = np.asarray(b, dtype=float) / np.linalg.norm(b)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(a @ b)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # 180 deg: any perpendicular axis works
        perp = np.array([1.0, 0.0, 0.0])
        if abs(float(perp @ a)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        perp = perp - a * float(perp @ a)
        return se3.rotation_about_axis(perp / np.linalg.norm(perp), np.pi)
    k = v / s
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


if __name__ == "__main__":
    sys.exit(main())
