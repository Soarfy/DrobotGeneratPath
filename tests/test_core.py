import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dobot_pathgen import se3
from dobot_pathgen.config import Config
from dobot_pathgen.ik import solve_ik_multistart, solve_ik_tracking
from dobot_pathgen.mesh_io import read_ply, read_stl, vertex_normals, write_ply_points
from dobot_pathgen.pathgen import densify_poses, raster_surface_path, select_region, tool_poses_from_surface
from dobot_pathgen.planning import TaskSpec, Trajectory, plan_task, validate_trajectory
from dobot_pathgen.urdf_chain import build_chain


@pytest.fixture(scope="module")
def cfg():
    return Config.load(ROOT / "config/robot.yaml")


@pytest.fixture(scope="module")
def chain(cfg):
    return build_chain(cfg.robot.urdf, cfg.robot.tip_link)


@pytest.fixture(scope="module")
def arch(cfg):
    vertices, faces, normals, _ = read_ply(cfg.scene.model_mesh)
    vertices = vertices * cfg.scene.scale
    return vertices, faces, vertex_normals(vertices, faces, normals)


class TestSe3:
    def test_rpy_roundtrip(self):
        rpy = np.array([17.0, -23.0, 41.0])
        R = se3.rpy_deg_to_matrix(rpy)
        assert np.allclose(se3.matrix_to_rpy_deg(R), rpy, atol=1e-9)

    def test_quaternion_roundtrip(self):
        R = se3.rpy_deg_to_matrix([12.0, 34.0, -56.0])
        q = se3.quat_xyzw_from_matrix(R)
        assert np.allclose(se3.matrix_from_quat_xyzw(q), R, atol=1e-12)

    def test_rotvec_roundtrip(self):
        R = se3.rotation_about_axis([1, 2, 3], 0.7)
        rv = se3.rotvec_from_matrix(R)
        assert np.allclose(se3.rotation_about_axis([1, 2, 3], np.linalg.norm(rv)), R, atol=1e-9)


class TestUrdfChain:
    def test_chain_shape(self, chain):
        assert chain.dof == 6
        assert chain.joint_names == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

    def test_limits(self, chain):
        assert np.allclose(chain.lower, [-6.28, -6.28, -2.792, -6.28, -6.28, -6.28])
        assert np.allclose(chain.upper, [6.28, 6.28, 2.792, 6.28, 6.28, 6.28])

    def test_fk_is_rigid(self, chain):
        T = chain.fk(np.zeros(6))
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert math.isclose(np.linalg.det(R), 1.0, abs_tol=1e-12)

    def test_fk_matches_numeric_jacobian(self, chain):
        from dobot_pathgen.ik import _jacobian

        q = np.array([0.3, -0.4, 0.5, -0.2, 0.6, 0.1])
        Ja = _jacobian(chain, q)
        eps = 1e-7
        T0 = chain.fk(q)
        Jn = np.zeros((6, 6))
        for i in range(6):
            dq = np.zeros(6)
            dq[i] = eps
            T1 = chain.fk(q + dq)
            Jn[:3, i] = (T1[:3, 3] - T0[:3, 3]) / eps
            Jn[3:, i] = se3.rotvec_from_matrix(T1[:3, :3] @ T0[:3, :3].T) / eps
        assert np.allclose(Ja, Jn, atol=1e-5)

    def test_clamp(self, chain):
        q = chain.clamp([10.0, -10.0, 0.0, 0.0, 0.0, 0.0])
        assert q[0] == chain.upper[0] and q[1] == chain.lower[1]


class TestIK:
    def test_roundtrip(self, chain):
        q = np.array([0.3, -0.4, 0.5, -0.2, 0.6, 0.1])
        result = solve_ik_multistart(chain, chain.fk(q), reference=q, restarts=4)
        assert result.success
        assert np.allclose(result.q, q, atol=1e-5)

    def test_downward_pose_reachable(self, chain):
        R = np.column_stack([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        T = se3.make_transform(R, np.array([0.45, 0.0, 0.25]))
        result = solve_ik_multistart(chain, T, restarts=30)
        assert result.success
        assert result.position_error_m < 1e-3

    def test_tracking_prefers_continuity(self, chain):
        q0 = np.array([0.2, -1.2, 1.0, 1.2, -0.9, 0.4])
        result = solve_ik_tracking(chain, chain.fk(q0), reference=q0, restarts=4)
        assert result.success
        assert np.max(np.abs(result.q - q0)) < 1e-4


class TestMeshIO:
    def test_read_arch(self, cfg):
        vertices, faces, normals, colors = read_ply(cfg.scene.model_mesh)
        assert vertices.shape[1] == 3
        assert len(faces) > 0
        assert colors is not None
        # Normals may come from the file or be computed afterwards; both paths
        # must produce a usable unit normal field.
        computed = vertex_normals(vertices, faces, normals)
        assert computed.shape == vertices.shape
        assert np.allclose(np.linalg.norm(computed, axis=1), 1.0, atol=1e-6)

    def test_ply_roundtrip(self, tmp_path):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        out = tmp_path / "pts.ply"
        write_ply_points(out, pts)
        v, f, n, c = read_ply(out)
        assert np.allclose(v, pts)
        assert len(f) == 0

    def test_stl_roundtrip(self, tmp_path):
        from dobot_pathgen.mesh_io import write_stl

        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        out = tmp_path / "tri.stl"
        write_stl(out, verts, faces)
        v2, f2, is_ascii = read_stl(out)
        assert not is_ascii
        assert len(f2) == 1
        assert np.allclose(v2, verts, atol=1e-5)


class TestPathGen:
    def test_region_bbox(self, arch):
        vertices, _, _ = arch
        mask = select_region(
            vertices,
            np.zeros_like(vertices),
            {"mode": "bbox", "min_m": [-0.03, 0.0, -0.03], "max_m": [-0.01, 0.04, -0.01]},
        )
        assert mask.any()
        assert np.all(vertices[mask, 0] <= -0.01)

    def test_raster_produces_serpentine(self, arch):
        vertices, _, normals = arch
        mask = select_region(
            vertices,
            normals,
            {"mode": "bbox", "min_m": [-0.03, 0.005, -0.028], "max_m": [-0.012, 0.04, -0.012]},
        )
        path = raster_surface_path(vertices, normals, mask, spacing_m=0.003, row_spacing_m=0.003)
        assert len(path.points) > 20
        assert np.allclose(np.linalg.norm(path.normals, axis=1), 1.0, atol=1e-6)

    def test_tool_poses_face_surface(self, arch):
        vertices, _, normals = arch
        mask = select_region(
            vertices,
            normals,
            {"mode": "bbox", "min_m": [-0.03, 0.005, -0.028], "max_m": [-0.012, 0.04, -0.012]},
        )
        path = raster_surface_path(vertices, normals, mask, spacing_m=0.003, row_spacing_m=0.003)
        poses, tips = tool_poses_from_surface(path, standoff_m=0.002)
        offsets = tips - path.points
        assert np.allclose(offsets, path.normals * 0.002, atol=1e-9)
        # Tool Z points opposite the surface normal.
        for i in range(0, len(poses), 7):
            assert np.dot(poses[i][:3, 2], path.normals[i]) < -0.99

    def test_densify_reduces_steps(self):
        p0 = se3.make_transform(np.eye(3), np.zeros(3))
        p1 = se3.make_transform(se3.rpy_deg_to_matrix([0, 0, 30]), np.array([0.02, 0, 0]))
        dense = densify_poses(np.array([p0, p1]), 0.002, 3.0)
        assert len(dense) >= 10
        steps = np.linalg.norm(np.diff(dense[:, :3, 3], axis=0), axis=1)
        assert steps.max() <= 0.0021


class TestPlanning:
    def test_plan_demo_task(self, cfg, arch):
        # Verify the shipped task file itself, not a hand-tuned copy.
        import yaml

        data = yaml.safe_load(
            (ROOT / "config/tasks/demo_occlusal_patch.yaml").read_text(encoding="utf-8")
        )
        vertices, _, normals = arch
        task = TaskSpec.from_dict(data)
        # The demo raster task keeps the tilt-clamp approach; the arch task is
        # the one that pins a single tool orientation.
        assert task.max_tilt_deg is not None or task.fixed_orientation_deg is not None
        trajectory, surface_path, meta = plan_task(
            cfg.robot.urdf,
            vertices,
            normals,
            task,
            cfg.scene.model_to_base,
            reference_q=cfg.robot.home_q,
            restarts=cfg.planning.ik_restarts,
            tcp_in_link6=cfg.tcp_pose_in_link6(),
            velocity_limits=cfg.robot.velocity_limits,
            acceleration_limits=cfg.robot.acceleration_limits,
        )
        assert len(trajectory.waypoints) > 50
        assert len(meta["ik_failures"]) == 0
        assert meta["validation"]["passed"]
        assert meta["validation"]["max_joint_step_deg"] < 20.0

    def test_trajectory_roundtrip(self, tmp_path, cfg, arch):
        vertices, _, normals = arch
        task = TaskSpec.from_dict(
            {
                "region": {
                    "mode": "bbox",
                    "min_m": [-0.025, 0.01, -0.022],
                    "max_m": [-0.015, 0.035, -0.014],
                },
                "spacing_m": 0.005,
                "row_spacing_m": 0.005,
            }
        )
        trajectory, _, _ = plan_task(
            cfg.robot.urdf,
            vertices,
            normals,
            task,
            cfg.scene.model_to_base,
            restarts=4,
            tcp_in_link6=cfg.tcp_pose_in_link6(),
        )
        out = tmp_path / "traj.json"
        trajectory.save(out)
        loaded = Trajectory.load(out)
        assert loaded.joint_names == trajectory.joint_names
        assert np.allclose(loaded.positions(), trajectory.positions())

    def test_validation_flags_limit_violation(self, chain):
        w = [
            {"time_s": 0.0, "positions": [0.0] * 6},
            {"time_s": 0.1, "positions": [0.0, 0.0, 5.0, 0.0, 0.0, 0.0]},
        ]
        traj = Trajectory.from_dict({"joint_names": chain.joint_names, "waypoints": w})
        report = validate_trajectory(chain, traj)
        assert not report["passed"]
        assert report["limit_violations"]
