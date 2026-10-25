import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dobot_pathgen import se3
from dobot_pathgen.calibration import load_calibration
from dobot_pathgen.config import Config
from dobot_pathgen.mesh_io import read_ply, vertex_normals
from dobot_pathgen.pathgen import arch_surface_path, path_length
from dobot_pathgen.planning import TaskSpec, plan_task


@pytest.fixture(scope="module")
def cfg():
    return Config.load(ROOT / "config/robot.yaml")


@pytest.fixture(scope="module")
def arch(cfg):
    vertices, faces, normals, _ = read_ply(cfg.scene.model_mesh)
    vertices = vertices * cfg.scene.scale
    return vertices, faces, vertex_normals(vertices, faces, normals)


class TestCampilationFormats:
    def test_yaml_tcp_and_model(self, tmp_path):
        path = tmp_path / "calib.yaml"
        path.write_text(
            "tcp:\n  xyz_mm: [0, 0, 160]\n  rpy_deg: [0, 0, 0]\n"
            "model_to_base:\n  translation_mm: [450, 0, 250]\n  rpy_deg: [90, 0, 90]\n",
            encoding="utf-8",
        )
        result = load_calibration(path)
        assert result.tcp is not None and result.model_to_base is not None
        assert np.allclose(result.tcp[:3, 3], [0, 0, 0.16], atol=1e-9)
        assert np.allclose(result.model_to_base[:3, 3], [0.45, 0.0, 0.25], atol=1e-9)
        assert np.allclose(result.model_to_base[:3, :3], se3.rpy_deg_to_matrix([90, 0, 90]), atol=1e-9)

    def test_json_matrix(self, tmp_path):
        path = tmp_path / "calib.json"
        path.write_text(
            json.dumps({"tcp": {"matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.12], [0, 0, 0, 1]]}}),
            encoding="utf-8",
        )
        result = load_calibration(path)
        assert np.allclose(result.tcp[:3, 3], [0, 0, 0.12])

    def test_flat_keys(self, tmp_path):
        path = tmp_path / "calib.json"
        path.write_text(
            json.dumps({"tcp_xyz_mm": [1, 2, 3], "tcp_rpy_deg": [0, 0, 0]}), encoding="utf-8"
        )
        result = load_calibration(path)
        assert np.allclose(result.tcp[:3, 3], [0.001, 0.002, 0.003])

    def test_plain_text_pose(self, tmp_path):
        path = tmp_path / "calib.txt"
        path.write_text("0.45 0.0 0.25 90 0 90\n", encoding="utf-8")
        result = load_calibration(path)
        assert np.allclose(result.model_to_base[:3, 3], [0.45, 0.0, 0.25])

    def test_plain_text_matrix(self, tmp_path):
        path = tmp_path / "calib.txt"
        matrix = np.eye(4)
        matrix[:3, 3] = [0.4, 0.1, 0.3]
        path.write_text(" ".join(f"{v:.6f}" for v in matrix.reshape(-1)), encoding="utf-8")
        result = load_calibration(path)
        assert np.allclose(result.model_to_base[:3, 3], [0.4, 0.1, 0.3])

    def test_pose_vector_mm(self, tmp_path):
        path = tmp_path / "calib.json"
        path.write_text(json.dumps({"pose_mm_deg": [450, 0, 250, 90, 0, 90]}), encoding="utf-8")
        result = load_calibration(path)
        assert np.allclose(result.model_to_base[:3, 3], [0.45, 0.0, 0.25])

    def test_rejects_non_orthonormal(self, tmp_path):
        path = tmp_path / "calib.json"
        path.write_text(
            json.dumps({"tcp": {"matrix": [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_calibration(path)

    def test_config_overrides_with_calibration(self, tmp_path):
        path = tmp_path / "calib.yaml"
        path.write_text(
            "tcp:\n  xyz_m: [0, 0, 0.2]\n  rpy_deg: [0, 0, 0]\n"
            "model_to_base:\n  xyz_m: [0.5, 0.0, 0.3]\n  rpy_deg: [0, 0, 0]\n",
            encoding="utf-8",
        )
        cfg_text = (ROOT / "config/robot.yaml").read_text(encoding="utf-8")
        # Insert a calibration block before the tool section. Built from chr(10)
        # on purpose: escape sequences inside the test file are easy to get
        # wrong and a literal TAB silently breaks the match.
        head, sep, tail = cfg_text.partition("tool:")
        assert sep, "robot.yaml has no tool section"
        cfg_text = (
            head
            + "calibration:" + chr(10)
            + f"  file: {path.as_posix()}" + chr(10) + chr(10)
            + sep + tail
        )
        cfg_path = tmp_path / "robot.yaml"
        cfg_path.write_text(cfg_text, encoding="utf-8")
        cfg = Config.load(cfg_path)
        assert np.allclose(cfg.tool.tcp_offset_m, [0, 0, 0.2])
        assert np.allclose(cfg.scene.model_to_base[:3, 3], [0.5, 0.0, 0.3])


class TestArchPath:
    def test_ridge_is_u_shaped(self, arch):
        vertices, _, normals = arch
        path = arch_surface_path(vertices, normals, spacing_m=0.004)
        assert len(path.points) > 20
        # The arch opens toward -Z (front teeth at +Z).
        assert path.points[:, 2].max() > path.points[:, 2].min() + 0.03
        assert path_length(path.points) > 0.12

    def test_spacing_respected(self, arch):
        vertices, _, normals = arch
        path = arch_surface_path(vertices, normals, spacing_m=0.005)
        steps = np.linalg.norm(np.diff(path.points, axis=0), axis=1)
        assert np.median(steps) == pytest.approx(0.005, abs=0.002)

    def test_u_range_limits_span(self, arch):
        vertices, _, normals = arch
        full = arch_surface_path(vertices, normals, spacing_m=0.004)
        part = arch_surface_path(
            vertices, normals, spacing_m=0.004, u_range=[-0.010, 0.010]
        )
        assert len(part.points) < len(full.points)
        assert path_length(part.points) < 0.5 * path_length(full.points)

    def test_normals_point_outward(self, arch):
        vertices, _, normals = arch
        path = arch_surface_path(vertices, normals, spacing_m=0.004)
        # Occlusal-facing normals have a positive Y component on this model.
        assert np.mean(path.normals[:, 1] > 0) > 0.8

    def test_plan_arch_task(self, cfg, arch):
        vertices, _, normals = arch
        task = TaskSpec.from_dict(
            {
                "name": "arch",
                "pattern": "arch",
                "arch": {"up_direction": [0, 1, 0]},
                "spacing_m": 0.005,
                "standoff_m": 0.002,
                "speed_m_s": 0.01,
                # A full arch rotates the surface normal by ~180 deg; clamping
                # the tilt is what keeps the wrist on one IK branch.
                "max_tilt_deg": 40.0,
                "tilt_reference": [0, 1, 0],
                "x_reference": [1, 0, 0],
            }
        )
        trajectory, surface_path, meta = plan_task(
            cfg.robot.urdf,
            vertices,
            normals,
            task,
            cfg.scene.model_to_base,
            reference_q=cfg.robot.home_q,
            restarts=8,
            tcp_in_link6=cfg.tcp_pose_in_link6(),
            velocity_limits=cfg.robot.velocity_limits,
            acceleration_limits=cfg.robot.acceleration_limits,
        )
        assert meta["pattern"] == "arch"
        assert len(trajectory.waypoints) > 20
        assert len(meta["ik_failures"]) == 0
        assert meta["validation"]["passed"]
