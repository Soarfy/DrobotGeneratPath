"""Generate self-contained demo assets.

1. A simple brush mesh (STL) used as the tool collision/visual geometry.
2. A combined URDF (CR5A + tool) with relative mesh paths, suitable for both
   the Isaac Sim URDF importer and the planner.
3. A downsampled target STL for fast Isaac Sim import.

Run once after cloning:
    python scripts/make_assets.py
"""

from __future__ import annotations

import math
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dobot_pathgen.mesh_io import read_ply, write_stl, write_ply_points  # noqa: E402


def _box(center, size):
    cx, cy, cz = center
    sx, sy, sz = (s / 2.0 for s in size)
    verts = []
    for dz in (-sz, sz):
        for dy in (-sy, sy):
            for dx in (-sx, sx):
                verts.append([cx + dx, cy + dy, cz + dz])
    # 0..7 as (x,y,z) bit pattern: index = (z<<2)|(y<<1)|x
    faces = [
        [0, 2, 1], [1, 2, 3],      # bottom
        [4, 5, 6], [5, 7, 6],      # top
        [0, 1, 4], [1, 5, 4],      # front
        [2, 6, 3], [3, 6, 7],      # back
        [0, 4, 2], [2, 4, 6],      # left
        [1, 3, 5], [3, 7, 5],      # right
    ]
    return verts, faces


def _cylinder(center, radius, height, segments=24, axis="z"):
    cx, cy, cz = center
    ring = []
    verts = []
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        if axis == "z":
            verts.append([cx + radius * math.cos(a), cy + radius * math.sin(a), cz - height / 2])
        else:
            verts.append([cx + radius * math.cos(a), cy, cz + radius * math.sin(a)])
    n = len(verts)
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        if axis == "z":
            verts.append([cx + radius * math.cos(a), cy + radius * math.sin(a), cz + height / 2])
        else:
            verts.append([cx + radius * math.cos(a), cy, cz + radius * math.sin(a)])
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
        faces.append([n + i, n + j, n + ((i + 1) % segments)])  # cap fan (approx)
    # caps
    for i in range(1, segments - 1):
        faces.append([0, i + 1, i])
        faces.append([n, n + i, n + i + 1])
    return verts, faces


def _tube(start, end, radius, segments=32):
    """Closed cylinder from ``start`` to ``end`` (both in metres)."""
    p0 = np.asarray(start, dtype=float)
    p1 = np.asarray(end, dtype=float)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        raise ValueError("tube endpoints coincide")
    axis /= length

    # Build an orthonormal frame around the axis.
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    verts = []
    for centre in (p0, p1):
        for i in range(segments):
            a = 2.0 * np.pi * i / segments
            verts.append((centre + radius * (np.cos(a) * u + np.sin(a) * v)).tolist())

    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, segments + j])
        faces.append([i, segments + j, segments + i])
    for i in range(1, segments - 1):
        faces.append([0, i + 1, i])
        faces.append([segments, segments + i, segments + i + 1])
    return verts, faces


def build_toothbrush_mesh(path: Path, offset_m, radius_m, segments=32):
    """Straight brush rod from the flange origin to the TCP offset."""
    offset = np.asarray(offset_m, dtype=float)
    length = float(np.linalg.norm(offset))
    if length < 1e-6:
        raise ValueError("tcp offset must be non-zero")

    # Rod from the flange to just short of the tip, plus a flat head at the tip.
    direction = offset / length
    head_len = min(0.03, 0.15 * length)
    body_end = offset - direction * head_len
    body, body_faces = _tube([0.0, 0.0, 0.0], body_end, radius_m * 0.55, segments)
    head, head_faces = _tube(body_end, offset, radius_m, segments)

    verts = body + head
    faces = body_faces + [[i + len(body) for i in tri] for tri in head_faces]
    write_stl(path, verts, faces, binary=True)
    return len(verts), len(faces), length


def _strip_mesh_prefix(urdf_root: ET.Element, mesh_root: str):
    for mesh in urdf_root.iter("mesh"):
        filename = mesh.attrib.get("filename", "")
        if "meshes/cr5/" in filename:
            mesh.set("filename", f"{mesh_root}/cr5/{Path(filename).name}")


def build_combined_urdf(
    robot_urdf: Path,
    out_urdf: Path,
    tcp_offset_m,
    tool_name="toothbrush",
    tool_link_name="brush_link",
    mesh_in_link6=False,
):
    tree = ET.parse(robot_urdf)
    root = tree.getroot()
    root.set("name", "cr5a_with_toothbrush")
    # The combined URDF lives in <out>/urdf/, meshes in <out>/meshes/.
    _strip_mesh_prefix(root, "../meshes")

    # Tool link attached to Link6. With mesh_in_link6 the STL already sits in
    # the Link6 frame (tip on the TCP offset), so the joint origin stays at
    # zero and only the tcp_link below carries the offset.
    link = ET.SubElement(root, "link", {"name": tool_link_name})
    inertial = ET.SubElement(link, "inertial")
    if mesh_in_link6:
        ET.SubElement(inertial, "origin", {"xyz": "-0.005 -0.145 0.213", "rpy": "0 0 0"})
    else:
        ET.SubElement(inertial, "origin", {"xyz": "0.06 0 0.08", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "0.35"})
    ET.SubElement(inertial, "inertia", {
        "ixx": "0.0016", "ixy": "0", "ixz": "0",
        "iyy": "0.0012", "iyz": "0", "izz": "0.0006",
    })
    for kind in ("visual", "collision"):
        geom = ET.SubElement(link, kind)
        ET.SubElement(geom, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ET.SubElement(geom, "geometry")
        ET.SubElement(geometry, "mesh", {"filename": f"../meshes/{tool_name}.stl"})
        if kind == "visual":
            material = ET.SubElement(geom, "material", {"name": "brush_blue"})
            ET.SubElement(material, "color", {"rgba": "0.15 0.45 0.85 1.0"})

    joint = ET.SubElement(root, "joint", {"name": f"{tool_link_name}_joint", "type": "fixed"})
    ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(joint, "parent", {"link": "Link6"})
    ET.SubElement(joint, "child", {"link": tool_link_name})

    # Explicit TCP frame, used by the planner and useful in RViz/Isaac.
    tcp_link = ET.SubElement(root, "link", {"name": "tcp_link"})
    tcp_joint = ET.SubElement(root, "joint", {"name": "tcp_joint", "type": "fixed"})
    offset = [0.0, 0.0, 0.0] if mesh_in_link6 else list(tcp_offset_m)
    ET.SubElement(tcp_joint, "origin", {
        "xyz": " ".join(f"{v:.6f}" for v in offset),
        "rpy": "0 0 0",
    })
    ET.SubElement(tcp_joint, "parent", {"link": tool_link_name})
    ET.SubElement(tcp_joint, "child", {"link": "tcp_link"})

    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_urdf, encoding="utf-8", xml_declaration=True)
    return out_urdf


def build_target_stl(ply_path: Path, out_stl: Path, stride: int = 1):
    verts, faces, normals, _ = read_ply(ply_path)
    verts = verts / 1000.0  # mm -> m
    if stride > 1:
        faces = faces[::stride]
    write_stl(out_stl, verts, faces, binary=True)
    return len(verts), len(faces), verts.min(axis=0), verts.max(axis=0)


def main():
    robot_urdf = ROOT / "assets/robot/cr5a/urdf/cr5_robot.urdf"
    out_dir = ROOT / "assets/robot/cr5a_with_tool"
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # Isaac Sim resolves relative mesh paths against the URDF, so the robot
    # STLs must live next to the combined file.
    import shutil

    robot_mesh_dir = mesh_dir / "cr5"
    robot_mesh_dir.mkdir(parents=True, exist_ok=True)
    for stl in (ROOT / "assets/robot/cr5a/meshes/cr5").glob("*.STL"):
        shutil.copy2(stl, robot_mesh_dir / stl.name)

    # Real brush: brushleap.ply, already rebuilt into the Link6 frame by
    # scripts/align_brush.py (tip on the TCP offset, handle toward the flange).
    brush_src = ROOT / "assets/tool/brushleap/brushleap_link6.stl"
    if not brush_src.exists():
        raise SystemExit(
            "brushleap_link6.stl missing; run: python scripts/align_brush.py"
        )
    shutil.copy2(brush_src, mesh_dir / "brushleap.stl")
    print(f"brush mesh: {mesh_dir / 'brushleap.stl'}")

    tcp_offset = [-0.009748, -0.186313, 0.223253]
    build_combined_urdf(
        robot_urdf,
        out_dir / "urdf/cr5a_with_toothbrush.urdf",
        tcp_offset,
        tool_name="brushleap",
        tool_link_name="brush_link",
        mesh_in_link6=True,
    )
    print(f"combined URDF: {out_dir / 'urdf/cr5a_with_toothbrush.urdf'}")

    target_dir = ROOT / "assets/target"
    target_ply = target_dir / "lower_jaw.ply"
    nv2, nf2, lo, hi = build_target_stl(target_ply, target_dir / "lower_jaw.stl", stride=2)
    print(f"target STL: {nv2} verts, {nf2} faces, bounds {lo.round(3)} .. {hi.round(3)} m")


if __name__ == "__main__":
    main()
