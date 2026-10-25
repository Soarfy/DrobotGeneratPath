"""Configuration loading and path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .calibration import CalibrationResult

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc

from . import se3


ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path)


@dataclass
class RobotConfig:
    name: str
    urdf: Path
    tip_link: str
    base_frame: str
    joint_names: list
    velocity_limits: np.ndarray
    acceleration_limits: np.ndarray
    home_q: np.ndarray
    position_limits: Optional[np.ndarray] = None

    @property
    def urdf_with_tool(self) -> Path:
        return _resolve(
            f"assets/robot/{self.name}_with_tool/urdf/{self.name}_with_toothbrush.urdf"
        )


@dataclass
class SceneConfig:
    model_mesh: Path
    mesh_units: str
    model_to_base: np.ndarray
    table_height_m: float

    @property
    def scale(self) -> float:
        return 0.001 if self.mesh_units.lower() in ("mm", "millimeter", "millimetres") else 1.0


@dataclass
class ToolConfig:
    name: str
    mesh: Path
    tcp_offset_m: np.ndarray
    tcp_rpy_deg: np.ndarray
    length_m: float
    radius_m: float


@dataclass
class PlanningConfig:
    speed_m_s: float
    acceleration_m_s2: float
    min_waypoint_dt_s: float
    ik_restarts: int


@dataclass
class Config:
    robot: RobotConfig
    scene: SceneConfig
    tool: ToolConfig
    planning: PlanningConfig
    raw: Dict[str, Any]
    calibration: Optional["CalibrationResult"] = None

    @classmethod
    def load(cls, path) -> "Config":
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        r = data["robot"]
        s = data["scene"]
        t = data["tool"]
        p = data["planning"]

        # Optional calibration files override the nominal TCP / model pose.
        calibration = None
        calib_spec = data.get("calibration")
        if calib_spec:
            from .calibration import load_calibration

            tcp_file = calib_spec.get("tcp_file") if isinstance(calib_spec, dict) else None
            model_file = calib_spec.get("model_file") if isinstance(calib_spec, dict) else None
            single = (
                calib_spec.get("file")
                if isinstance(calib_spec, dict)
                else (calib_spec if isinstance(calib_spec, str) else None)
            )
            if single:
                calibration = load_calibration(_resolve(single))
            elif tcp_file or model_file:
                from .calibration import CalibrationResult

                merged = CalibrationResult(source=", ".join(str(x) for x in (tcp_file, model_file) if x))
                if tcp_file:
                    part = load_calibration(_resolve(tcp_file))
                    merged.tcp = part.tcp
                if model_file:
                    part = load_calibration(_resolve(model_file))
                    merged.model_to_base = part.model_to_base
                calibration = merged

        world = s.get("model_to_base", {})
        translation = np.asarray(world.get("translation_m", [0.45, 0.0, 0.2]), dtype=float)
        rpy = np.asarray(world.get("rpy_deg", [90.0, 0.0, 0.0]), dtype=float)
        model_to_base = se3.make_transform(se3.rpy_deg_to_matrix(rpy), translation)
        if calibration is not None and calibration.model_to_base is not None:
            model_to_base = calibration.model_to_base

        tcp_offset = np.asarray(t.get("tcp_offset_m", [0.0, 0.0, 0.16]), dtype=float)
        tcp_rpy = np.asarray(t.get("tcp_rpy_deg", [0.0, 0.0, 0.0]), dtype=float)
        if calibration is not None and calibration.tcp is not None:
            tcp_offset = calibration.tcp[:3, 3].copy()
            tcp_rpy = se3.matrix_to_rpy_deg(calibration.tcp[:3, :3])

        return cls(
            robot=RobotConfig(
                name=r.get("name", "cr5a"),
                urdf=_resolve(r["urdf"]),
                tip_link=r.get("tip_link", "Link6"),
                base_frame=r.get("base_frame", "base_link"),
                joint_names=list(r.get("joint_names", [])),
                velocity_limits=np.asarray(r.get("velocity_limits_rad_s", []), dtype=float),
                acceleration_limits=np.asarray(r.get("acceleration_limits_rad_s2", []), dtype=float),
                home_q=np.asarray(r.get("home_q_rad", [0.0] * 6), dtype=float),
                position_limits=(
                    np.asarray(r["position_limits_rad"], dtype=float)
                    if "position_limits_rad" in r
                    else None
                ),
            ),
            scene=SceneConfig(
                model_mesh=_resolve(s["model_mesh"]),
                mesh_units=s.get("mesh_units", "mm"),
                model_to_base=model_to_base,
                table_height_m=float(s.get("table_height_m", 0.0)),
            ),
            tool=ToolConfig(
                name=t.get("name", "brush_demo"),
                mesh=_resolve(t.get("mesh", "assets/robot/cr5a_with_tool/meshes/toothbrush.stl")),
                tcp_offset_m=tcp_offset,
                tcp_rpy_deg=tcp_rpy,
                length_m=float(t.get("length_m", 0.16)),
                radius_m=float(t.get("radius_m", 0.012)),
            ),
            planning=PlanningConfig(
                speed_m_s=float(p.get("speed_m_s", 0.01)),
                acceleration_m_s2=float(p.get("acceleration_m_s2", 0.05)),
                min_waypoint_dt_s=float(p.get("min_waypoint_dt_s", 0.05)),
                ik_restarts=int(p.get("ik_restarts", 24)),
            ),
            raw=data,
            calibration=calibration,
        )

    def tcp_pose_in_link6(self) -> np.ndarray:
        """Fixed transform from the last link (Link6) to the tool tip."""
        return se3.make_transform(
            se3.rpy_deg_to_matrix(self.tool.tcp_rpy_deg), self.tool.tcp_offset_m
        )
