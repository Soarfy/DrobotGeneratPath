import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dobot_pathgen import se3
from dobot_pathgen.collision import ToolGeometry, check_clearances, load_link_clouds, tool_points
from dobot_pathgen.config import Config
from dobot_pathgen.mesh_io import read_ply, vertex_normals
from dobot_pathgen.planning import TaskSpec, Trajectory, plan_task
from dobot_pathgen.urdf_chain import build_chain


@pytest.fixture(scope="module")
def cfg():
    return Config.load(ROOT / "config/robot.yaml")


@pytest.fixture(scope="module")
def chain(cfg):
    return build_chain(cfg.robot.urdf, cfg.robot.tip_link)


@pytest.fixture(scope="module")
def jaw(cfg):
    vertices, faces, normals, _ = read_ply(cfg.scene.model_mesh)
    vertices = vertices * cfg.scene.scale
    return vertices, faces, vertex_normals(vertices, faces, normals)


@pytest.fixture(scope="module")
def planned(cfg, jaw):
    vertices, _, normals = jaw
    task = TaskSpec.from_dict(
        {
            "name": "arch_clearance",
            "pattern": "arch",
            "arch": {"up_direction": [0, 1, 0]},
            "spacing_m": 0.008,
            "standoff_m": 0.002,
            "speed_m_s": 0.01,
            "fixed_orientation_deg": [180.0, 0.0, 150.0],
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
    return trajectory, surface_path, meta


class TestToolGeometry:
    def test_offset_is_y120_z150(self, cfg):
        # Spec: the tip is offset only in Y and Z, no X.
        offset = np.asarray(cfg.tool.tcp_offset_m)
        assert offset[0] == pytest.approx(0.0, abs=1e-9)
        assert offset[1] == pytest.approx(0.120, abs=1e-9)
        assert offset[2] == pytest.approx(0.150, abs=1e-9)

    def test_no_rotation(self, cfg):
        assert np.allclose(cfg.tool.tcp_rpy_deg, [0.0, 0.0, 0.0])
        assert np.allclose(cfg.tcp_pose_in_link6()[:3, :3], np.eye(3))

    def test_geometry_length_axis(self, cfg):
        geom = ToolGeometry(
            tcp_offset_m=np.asarray(cfg.tool.tcp_offset_m), radius_m=cfg.tool.radius_m
        )
        offset = np.asarray(cfg.tool.tcp_offset_m)
        assert geom.length == pytest.approx(float(np.linalg.norm(offset)), abs=1e-9)
        assert np.allclose(geom.axis, offset / np.linalg.norm(offset))

    def test_tool_points_span_the_rod(self, cfg):
        geom = ToolGeometry(
            tcp_offset_m=np.asarray(cfg.tool.tcp_offset_m), radius_m=cfg.tool.radius_m
        )
        pts = tool_points(geom, np.eye(4))
        assert pts.shape[1] == 3
        # Includes the flange origin and the tip.
        assert np.min(np.linalg.norm(pts, axis=1)) < 1e-9
        tip = np.asarray(cfg.tool.tcp_offset_m)
        assert np.min(np.linalg.norm(pts - tip, axis=1)) < 0.05
        # Rod radius respected.
        lateral = pts - np.outer(pts @ tip / np.linalg.norm(tip), tip / np.linalg.norm(tip))
        assert np.linalg.norm(lateral, axis=1).max() <= cfg.tool.radius_m * 1.3


class TestLowerJaw:
    def test_mesh_loaded(self, cfg, jaw):
        vertices, faces, normals = jaw
        assert len(vertices) > 100_000
        assert len(faces) > 100_000
        # Lower jaw scan is roughly 71 x 40 x 58 mm.
        extent = (vertices.max(axis=0) - vertices.min(axis=0)) * 1000
        assert 60 < extent[0] < 80
        assert 30 < extent[1] < 50

    def test_normals_point_up(self, jaw):
        _, _, normals = jaw
        assert np.mean(normals[:, 1] > 0.5) > 0.2

    def test_plan_reaches_all_waypoints(self, planned):
        trajectory, _, meta = planned
        assert len(trajectory.waypoints) > 50
        assert len(meta["ik_failures"]) == 0
        assert meta["validation"]["passed"]

    def test_arch_spans_both_sides(self, planned):
        _, surface_path, _ = planned
        xs = surface_path.points[:, 0]
        assert xs.min() < -0.015
        assert xs.max() > 0.015


class TestClearance:
    def test_tool_clears_the_arm(self, cfg, chain, planned):
        trajectory, _, _ = planned
        geom = ToolGeometry(
            tcp_offset_m=np.asarray(cfg.tool.tcp_offset_m),
            radius_m=cfg.tool.radius_m,
            head_radius_m=cfg.tool.radius_m * 1.25,
            head_length_m=0.02,
        )
        clouds = load_link_clouds(cfg.robot.urdf)
        report = check_clearances(chain, trajectory, geom, clouds, self_margin_m=0.01)
        # The 120 mm reach of this brush keeps ~77 mm from the arm on the
        # shipped task; allow a generous floor but catch real collisions.
        assert report["self_min_distance_mm"] > 20.0, report["self_violations"][:3]

    def test_tool_touches_workpiece_only_at_the_tip(self, cfg, chain, planned, jaw):
        trajectory, _, _ = planned
        vertices, _, _ = jaw
        points = (vertices @ cfg.scene.model_to_base[:3, :3].T) + cfg.scene.model_to_base[:3, 3]
        geom = ToolGeometry(
            tcp_offset_m=np.asarray(cfg.tool.tcp_offset_m),
            radius_m=cfg.tool.radius_m,
            head_radius_m=cfg.tool.radius_m * 1.25,
            head_length_m=0.02,
        )
        clouds = load_link_clouds(cfg.robot.urdf)
        report = check_clearances(
            chain, trajectory, geom, clouds, workpiece_points=points[::8]
        )
        # Contact is intentional: the tip rides 2 mm off the enamel.
        assert report["workpiece_min_distance_mm"] < 3.0
        assert report["workpiece_min_distance_mm"] > -1.0
