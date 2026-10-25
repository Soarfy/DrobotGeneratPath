"""Load TCP / model calibration results written in whatever format is at hand.

Supported JSON/YAML shapes
--------------------------

1. Explicit TCP offset (millimetres are accepted and converted):

    tcp:
      xyz_mm: [0, 0, 160]
      rpy_deg: [0, 0, 0]

2. Explicit TCP transform as a 4x4 matrix:

    tcp:
      matrix: [[1,0,0,0],[0,1,0,0],[0,0,1,0.16],[0,0,0,1]]

3. Flat keys at the top level:

    tcp_xyz_m: [0, 0, 0.16]
    tcp_rpy_deg: [0, 0, 0]

4. Model (workpiece) pose in the robot base frame, same three variants under
   ``model`` / ``model_to_base`` keys. ``translation_mm`` is also accepted.

5. A 6-element "pose" vector ``[x, y, z, rx, ry, rz]``. Units follow the key
   name: ``pose_mm_deg`` or ``pose_m_deg``.

Text files
----------
A plain text file containing either 16 numbers (row-major 4x4) or 6 numbers
(x, y, z, rx, ry, rz) is accepted. Units are assumed to be metres/degrees; pass
``units="mm"`` to override.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from . import se3


@dataclass
class CalibrationResult:
    tcp: Optional[np.ndarray] = None        # Link6 -> tool tip
    model_to_base: Optional[np.ndarray] = None
    source: str = ""
    notes: str = ""

    def describe(self) -> str:
        lines = [f"source: {self.source}"]
        if self.tcp is not None:
            xyz, rpy = se3.matrix_to_xyz_rpy_deg(self.tcp)
            lines.append(
                f"tcp (Link6 -> tip): xyz_mm=[{xyz[0]*1000:.2f}, {xyz[1]*1000:.2f}, {xyz[2]*1000:.2f}] "
                f"rpy_deg=[{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}]"
            )
        if self.model_to_base is not None:
            xyz, rpy = se3.matrix_to_xyz_rpy_deg(self.model_to_base)
            lines.append(
                f"model->base: xyz_mm=[{xyz[0]*1000:.2f}, {xyz[1]*1000:.2f}, {xyz[2]*1000:.2f}] "
                f"rpy_deg=[{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}]"
            )
        if self.notes:
            lines.append(self.notes)
        return "\n".join(lines)


def _as_float_list(value, n: Optional[int] = None, name: str = "") -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if n is not None and arr.size != n:
        raise ValueError(f"{name or 'value'} must have {n} numbers, got {arr.size}")
    return arr


def _matrix_from_any(data: Any, *, length_scale: float = 1.0, name: str = "") -> np.ndarray:
    """Accept a 4x4 matrix, an xyz/rpy dict, or a 6-vector."""
    if isinstance(data, dict):
        if "matrix" in data:
            M = np.asarray(data["matrix"], dtype=float)
            if M.shape != (4, 4):
                raise ValueError(f"{name}: matrix must be 4x4, got {M.shape}")
            M = M.copy()
            M[:3, 3] *= length_scale
            return M

        for key in ("xyz_m", "xyz_mm", "translation_m", "translation_mm", "position_m", "position_mm"):
            if key in data:
                xyz = _as_float_list(data[key], 3, key)
                if key.endswith("_mm"):
                    xyz = xyz / 1000.0
                break
        else:
            if "pose" in data:
                pose = _as_float_list(data["pose"], 6, "pose")
                return se3.make_transform(
                    se3.rpy_deg_to_matrix(pose[3:6]), pose[:3] * length_scale
                )
            raise ValueError(f"{name}: no position key found ({sorted(data)})")

        rpy = np.zeros(3)
        for key in ("rpy_deg", "rpy", "rotation_deg"):
            if key in data:
                rpy = _as_float_list(data[key], 3, key)
                break
        else:
            if "quat_xyzw" in data:
                R = se3.matrix_from_quat_xyzw(_as_float_list(data["quat_xyzw"], 4, "quat_xyzw"))
                return se3.make_transform(R, xyz)
            if "quat_wxyz" in data:
                q = _as_float_list(data["quat_wxyz"], 4, "quat_wxyz")
                R = se3.matrix_from_quat_xyzw([q[1], q[2], q[3], q[0]])
                return se3.make_transform(R, xyz)
            if "rotvec" in data:
                rv = _as_float_list(data["rotvec"], 3, "rotvec")
                angle = float(np.linalg.norm(rv))
                R = se3.rotation_about_axis(rv / angle, angle) if angle > 1e-12 else np.eye(3)
                return se3.make_transform(R, xyz)
        return se3.make_transform(se3.rpy_deg_to_matrix(rpy), xyz)

    arr = np.asarray(data, dtype=float)
    if arr.shape == (4, 4):
        M = arr.copy()
        M[:3, 3] *= length_scale
        return M
    flat = arr.reshape(-1)
    if flat.size == 16:
        M = flat.reshape(4, 4).copy()
        M[:3, 3] *= length_scale
        return M
    if flat.size == 6:
        return se3.make_transform(
            se3.rpy_deg_to_matrix(flat[3:6]), flat[:3] * length_scale
        )
    raise ValueError(f"{name}: cannot interpret calibration data with shape {arr.shape}")


def _read_structured(path: Path) -> Dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text) or {}
    if suffix == ".json":
        return json.loads(text)
    # Plain text: strip comments, keep numbers.
    numbers = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text)
    values = np.array([float(v) for v in numbers], dtype=float)
    if values.size >= 16:
        return {"matrix": values[:16].reshape(4, 4).tolist()}
    if values.size >= 6:
        return {"pose_m_deg": values[:6].tolist()}
    raise ValueError(f"{path}: expected 16 or 6 numbers, found {values.size}")


def load_calibration(path, *, default_length_scale: float = 1.0) -> CalibrationResult:
    """Parse a calibration file into a :class:`CalibrationResult`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    data = _read_structured(path)
    result = CalibrationResult(source=str(path))

    if not isinstance(data, dict):
        data = {"pose_mm_deg": np.asarray(data, dtype=float).reshape(-1).tolist()}

    tcp_data = data.get("tcp") or data.get("tool") or data.get("tool_frame")
    if tcp_data is None and any(k.startswith("tcp_") for k in data):
        xyz_key = "tcp_xyz_mm" if "tcp_xyz_mm" in data else "tcp_xyz_m"
        tcp_data = {"xyz_mm" if xyz_key.endswith("_mm") else "xyz_m": data[xyz_key]}
        if "tcp_rpy_deg" in data:
            tcp_data["rpy_deg"] = data["tcp_rpy_deg"]
        elif "tcp_rpy" in data:
            tcp_data["rpy_deg"] = data["tcp_rpy"]
    if tcp_data is not None:
        result.tcp = _matrix_from_any(tcp_data, name="tcp")

    model_data = data.get("model_to_base") or data.get("model") or data.get("workpiece")
    if model_data is None and any(k.startswith("model_") for k in data):
        xyz_key = "model_xyz_mm" if "model_xyz_mm" in data else "model_xyz_m"
        model_data = {"xyz_mm" if xyz_key.endswith("_mm") else "xyz_m": data[xyz_key]}
        if "model_rpy_deg" in data:
            model_data["rpy_deg"] = data["model_rpy_deg"]
    if model_data is not None:
        result.model_to_base = _matrix_from_any(model_data, name="model_to_base")

    if result.tcp is None and result.model_to_base is None:
        # Flat pose at the top level is interpreted as the model pose when a
        # "model_pose" style key is absent but a bare transform is present.
        for key in ("pose_mm_deg", "pose_m_deg", "pose", "matrix"):
            if key in data:
                scale = 1.0 / 1000.0 if key.endswith("_mm_deg") else 1.0
                result.model_to_base = _matrix_from_any(data[key], length_scale=scale, name=key)
                break
        else:
            raise ValueError(
                f"{path}: no tcp / model_to_base section found; keys were {sorted(data)}"
            )

    if result.tcp is not None:
        R = result.tcp[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-3) or not math.isclose(
            float(np.linalg.det(R)), 1.0, abs_tol=1e-3
        ):
            raise ValueError(f"{path}: tcp rotation is not orthonormal")

    if "note" in data:
        result.notes = str(data["note"])
    elif "notes" in data:
        result.notes = str(data["notes"])
    return result
