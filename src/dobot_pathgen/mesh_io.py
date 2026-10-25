"""Dependency-free mesh IO for PLY / STL / OBJ plus normal estimation."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Tuple

import numpy as np

_PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

# struct module spellings for the sequential (list-property) parser
_STRUCT_TYPES = {
    "char": "b", "int8": "b",
    "uchar": "B", "uint8": "B",
    "short": "h", "int16": "h",
    "ushort": "H", "uint16": "H",
    "int": "i", "int32": "i",
    "uint": "I", "uint32": "I",
    "float": "f", "float32": "f",
    "double": "d", "float64": "d",
}


def _parse_ply_header(data: bytes):
    marker = data.find(b"end_header")
    if marker < 0:
        raise ValueError("PLY header not found")
    header = data[:marker].decode("ascii", errors="ignore")
    start = marker + len(b"end_header")
    # Consume exactly one line terminator. A "skip while in (\n, \r)" loop eats
    # a leading 0x0D that belongs to the first vertex (binary PLY stores floats
    # little-endian, so 0x0D is a legal high byte) and shifts the whole array.
    if data[start:start + 2] == b"\r\n":
        start += 2
    elif data[start:start + 1] in (b"\n", b"\r"):
        start += 1
    fmt = None
    elements = []
    current = None
    for line in header.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            current = {"name": parts[1], "count": int(parts[2]), "props": []}
            elements.append(current)
        elif parts[0] == "property" and current is not None:
            if parts[1] == "list":
                current["props"].append(("list", parts[2], parts[3], parts[4]))
            else:
                current["props"].append(("scalar", parts[1], parts[2]))
    return fmt, elements, start


def read_ply(path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (vertices Nx3, faces Mx3, normals Nx3 or None, colors Nx3 uint8 or None).

    Vertex coordinates are returned in the file's native units.
    """
    data = Path(path).read_bytes()
    fmt, elements, offset = _parse_ply_header(data)
    if fmt not in ("binary_little_endian", "binary_big_endian", "ascii"):
        raise ValueError(f"unsupported PLY format: {fmt}")

    verts = None
    faces = None
    normals = None
    colors = None

    if fmt == "ascii":
        text = data[offset:].decode("ascii", errors="ignore").split()
        cursor = 0
        for elem in elements:
            props = elem["props"]
            rows = []
            for _ in range(elem["count"]):
                row = {}
                for prop in props:
                    if prop[0] == "scalar":
                        row[prop[2]] = float(text[cursor])
                        cursor += 1
                    else:
                        n = int(float(text[cursor]))
                        cursor += 1
                        row[prop[2]] = [int(float(text[cursor + i])) for i in range(n)]
                        cursor += n
                rows.append(row)
            if elem["name"] == "vertex":
                verts = np.array([[r["x"], r["y"], r["z"]] for r in rows], dtype=float)
                if props and all(k in rows[0] for k in ("nx", "ny", "nz")):
                    normals = np.array([[r["nx"], r["ny"], r["nz"]] for r in rows], dtype=float)
                if all(k in rows[0] for k in ("red", "green", "blue")):
                    colors = np.array([[r["red"], r["green"], r["blue"]] for r in rows], dtype=np.uint8)
            elif elem["name"] == "face":
                faces = np.array([r["vertex_indices"] for r in rows], dtype=np.int64)
    else:
        endian = "<" if fmt == "binary_little_endian" else ">"
        cursor = offset
        for elem in elements:
            props = elem["props"]
            if all(p[0] == "scalar" for p in props):
                dtype = np.dtype([(p[2], endian + _PLY_TYPES[p[1]]) for p in props])
                arr = np.frombuffer(data, dtype=dtype, count=elem["count"], offset=cursor)
                cursor += dtype.itemsize * elem["count"]
                if elem["name"] == "vertex":
                    verts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(float)
                    if all(k in arr.dtype.names for k in ("nx", "ny", "nz")):
                        normals = np.stack([arr["nx"], arr["ny"], arr["nz"]], axis=1).astype(float)
                    if all(k in arr.dtype.names for k in ("red", "green", "blue")):
                        colors = np.stack([arr["red"], arr["green"], arr["blue"]], axis=1).astype(np.uint8)
            else:
                # Mixed scalar/list element: parse sequentially.
                rows = []
                for _ in range(elem["count"]):
                    row = {}
                    for prop in props:
                        if prop[0] == "scalar":
                            fmt = _STRUCT_TYPES[prop[1]]
                            size = struct.calcsize(fmt)
                            row[prop[2]] = struct.unpack_from(endian + fmt, data, cursor)[0]
                            cursor += size
                        else:
                            count_fmt = _STRUCT_TYPES[prop[1]]
                            cursor += struct.calcsize(count_fmt)
                            n = struct.unpack_from(endian + count_fmt, data, cursor - struct.calcsize(count_fmt))[0]
                            item_fmt = _STRUCT_TYPES[prop[2]]
                            item_size = struct.calcsize(item_fmt)
                            row[prop[3]] = list(
                                struct.unpack_from(endian + item_fmt * n, data, cursor)
                            )
                            cursor += item_size * n
                    rows.append(row)
                if elem["name"] == "face":
                    faces = np.array([r["vertex_indices"] for r in rows], dtype=np.int64)

    if verts is None:
        raise ValueError(f"no vertex element in {path}")
    if faces is None or len(faces) == 0:
        faces = np.zeros((0, 3), dtype=np.int64)
    return verts, faces, normals, colors


def write_ply_points(path, points, colors=None, normals=None):
    points = np.asarray(points, dtype=float)
    n = len(points)
    header = ["ply", "format ascii 1.0", f"element vertex {n}"]
    header += ["property float x", "property float y", "property float z"]
    if normals is not None:
        header += ["property float nx", "property float ny", "property float nz"]
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(header) + "\n")
        for i in range(n):
            row = [f"{points[i, 0]:.6f}", f"{points[i, 1]:.6f}", f"{points[i, 2]:.6f}"]
            if normals is not None:
                row += [f"{normals[i, 0]:.6f}", f"{normals[i, 1]:.6f}", f"{normals[i, 2]:.6f}"]
            if colors is not None:
                c = np.clip(colors[i], 0, 255).astype(int)
                row += [str(int(c[0])), str(int(c[1])), str(int(c[2]))]
            f.write(" ".join(row) + "\n")


def write_stl(path, vertices, faces, binary: bool = True):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if not binary:
        with open(path, "w", encoding="ascii", newline="\n") as f:
            f.write("solid mesh\n")
            for tri in faces:
                p = vertices[tri]
                n = np.cross(p[1] - p[0], p[2] - p[0])
                norm = np.linalg.norm(n)
                n = n / norm if norm > 1e-12 else np.zeros(3)
                f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                f.write("    outer loop\n")
                for v in p:
                    f.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
                f.write("    endloop\n  endfacet\n")
            f.write("endsolid mesh\n")
        return
    with open(path, "wb") as f:
        f.write(b"CR5A pathgen binary STL".ljust(80, b" "))
        f.write(struct.pack("<I", len(faces)))
        for tri in faces:
            p = vertices[tri]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            norm = np.linalg.norm(n)
            n = n / norm if norm > 1e-12 else np.zeros(3)
            f.write(struct.pack("<3f", *n))
            for v in p:
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))


def read_stl(path):
    data = Path(path).read_bytes()
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + n * 50 == len(data):
            tri = np.frombuffer(
                data, dtype=np.dtype([("n", "<3f4"), ("v", "<3f4", (3,)), ("a", "<u2")]),
                count=n, offset=84,
            )
            verts = tri["v"].reshape(-1, 3).astype(float)
            faces = np.arange(len(verts), dtype=np.int64).reshape(-1, 3)
            return verts, faces, False
    # ASCII
    verts = []
    for line in data.decode("ascii", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    verts = np.asarray(verts, dtype=float)
    return verts, np.arange(len(verts), dtype=np.int64).reshape(-1, 3), True


def write_obj(path, vertices, faces):
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def vertex_normals(vertices, faces, supplied=None):
    """Area-weighted vertex normals. Uses supplied normals when valid."""
    vertices = np.asarray(vertices, dtype=float)
    if supplied is not None:
        supplied = np.asarray(supplied, dtype=float)
        if supplied.shape == vertices.shape:
            lens = np.linalg.norm(supplied, axis=1)
            valid = lens > 1e-12
            out = np.zeros_like(vertices)
            out[valid] = supplied[valid] / lens[valid, None]
            if valid.mean() > 0.9:
                return out
    out = np.zeros_like(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return out
    p0, p1, p2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    for k in range(3):
        np.add.at(out, faces[:, k], fn)
    lens = np.linalg.norm(out, axis=1)
    valid = lens > 1e-12
    out[valid] /= lens[valid, None]
    return out


def transform(vertices, T):
    vertices = np.asarray(vertices, dtype=float)
    return vertices @ np.asarray(T, dtype=float)[:3, :3].T + T[:3, 3]


def bounds(vertices):
    v = np.asarray(vertices, dtype=float)
    return v.min(axis=0), v.max(axis=0)
