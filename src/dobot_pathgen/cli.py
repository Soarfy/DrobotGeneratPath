"""Command-line entry point: plan, simulate, execute."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import se3
from .config import Config
from .mesh_io import read_ply, vertex_normals
from .planning import TaskSpec, Trajectory, plan_task, validate_trajectory
from .urdf_chain import build_chain


def _load_task(path) -> TaskSpec:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TaskSpec.from_dict(data)


def cmd_calib(args) -> int:
    from .calibration import load_calibration

    result = load_calibration(args.file)
    print(result.describe())

    if args.apply:
        cfg_path = Path(args.config)
        text = cfg_path.read_text(encoding="utf-8")
        import yaml

        data = yaml.safe_load(text)
        if result.tcp is not None:
            xyz, rpy = se3.matrix_to_xyz_rpy_deg(result.tcp)
            data["tool"]["tcp_offset_m"] = [float(v) for v in xyz]
            data["tool"]["tcp_rpy_deg"] = [float(v) for v in rpy]
        if result.model_to_base is not None:
            xyz, rpy = se3.matrix_to_xyz_rpy_deg(result.model_to_base)
            data["scene"].setdefault("model_to_base", {})
            data["scene"]["model_to_base"]["translation_m"] = [float(v) for v in xyz]
            data["scene"]["model_to_base"]["rpy_deg"] = [float(v) for v in rpy]
        cfg_path.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        print(f"\nupdated {cfg_path}")
    return 0


def _clearance_report(cfg, trajectory, workpiece_points):
    from .collision import ToolGeometry, check_clearances, load_link_clouds
    from .urdf_chain import build_chain

    chain = build_chain(cfg.robot.urdf, cfg.robot.tip_link)
    geom = ToolGeometry(
        tcp_offset_m=np.asarray(cfg.tool.tcp_offset_m, dtype=float),
        radius_m=cfg.tool.radius_m,
        head_radius_m=cfg.tool.radius_m * 1.25,
        head_length_m=0.02,
    )
    clouds = load_link_clouds(cfg.robot.urdf)
    return check_clearances(
        chain,
        trajectory,
        geom,
        clouds,
        workpiece_points=workpiece_points[::4],
        self_margin_m=0.01,
        workpiece_margin_m=0.0,
    )


def cmd_plan(args) -> int:
    cfg = Config.load(args.config)
    task = _load_task(args.task)
    if args.speed is not None:
        task.speed_m_s = args.speed

    vertices, faces, normals, _ = read_ply(cfg.scene.model_mesh)
    vertices = vertices * cfg.scene.scale
    normals = vertex_normals(vertices, faces, normals)

    trajectory, surface_path, meta = plan_task(
        cfg.robot.urdf,
        vertices,
        normals,
        task,
        cfg.scene.model_to_base,
        tip_link=cfg.robot.tip_link,
        reference_q=cfg.robot.home_q,
        restarts=cfg.planning.ik_restarts,
        tcp_in_link6=cfg.tcp_pose_in_link6(),
        velocity_limits=cfg.robot.velocity_limits,
        acceleration_limits=cfg.robot.acceleration_limits,
    )
    trajectory.meta["task"] = task.name
    trajectory.meta["config"] = str(cfg.raw.get("robot", {}).get("name", "cr5a"))

    out = Path(args.out)
    trajectory.save(out)
    meta_path = out.with_suffix(".meta.json")

    # Proximity gate: a 200 mm brush mounted at an angle can graze the arm even
    # when every waypoint is reachable, so check it before declaring success.
    clearance = _clearance_report(
        cfg,
        trajectory,
        (vertices @ cfg.scene.model_to_base[:3, :3].T) + cfg.scene.model_to_base[:3, 3],
    )
    meta["clearance"] = clearance
    meta_path.write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")

    print(f"task            : {task.name}")
    print(f"pattern         : {meta.get('pattern', 'raster')}")
    if "region_vertex_count" in meta:
        print(f"region vertices : {meta['region_vertex_count']}")
    print(f"surface points  : {meta['surface_path_points']}")
    print(f"waypoints       : {meta['waypoint_count']}")
    print(f"duration        : {meta['duration_s']:.1f} s")
    print(f"path length     : {meta['path_length_m']:.3f} m")
    print(f"IK failures     : {len(meta['ik_failures'])}")
    print(f"max joint step  : {meta['validation']['max_joint_step_deg']:.1f} deg")
    print(f"velocity usage  : {meta['validation']['max_velocity_fraction'] * 100:.0f}%")
    print(f"validation      : {'PASS' if meta['validation']['passed'] else 'FAIL'}")
    if clearance.get("self_min_distance_mm") is not None:
        print(
            f"tool-arm gap    : {clearance['self_min_distance_mm']:.1f} mm "
            f"(min at wp {clearance['self_min_index']}, {clearance['self_min_link']})"
        )
        print(
            f"tool-workpiece  : {clearance['workpiece_min_distance_mm']:.1f} mm "
            f"(min at wp {clearance['workpiece_min_index']})"
        )
        print(f"proximity gate  : {'PASS' if clearance['passed'] else 'FAIL'}")
    print(f"trajectory      : {out}")
    print(f"report          : {meta_path}")

    if not meta["validation"]["passed"]:
        print("\nValidation problems:")
        print(json.dumps(meta["validation"], indent=2, default=float)[:2000])
        return 2
    if meta["ik_failures"]:
        # Dropped waypoints silently shorten the path, which would leave parts
        # of the arch uncleaned without anyone noticing.
        print(
            f"\nIK dropped {len(meta['ik_failures'])} of {meta['input_pose_count']} "
            "poses. Lower max_tilt_deg, narrow the region, or re-seat the model."
        )
        print(json.dumps(meta["ik_failures"][:5], indent=2, default=float))
        return 4
    if not clearance["passed"]:
        print("\nProximity problems (tool collision risk):")
        print(json.dumps(clearance, indent=2, default=float)[:2000])
        return 3
    return 0


def cmd_check(args) -> int:
    cfg = Config.load(args.config)
    chain = build_chain(cfg.robot.urdf, cfg.robot.tip_link)
    trajectory = Trajectory.load(args.trajectory)
    report = validate_trajectory(chain, trajectory)
    print(json.dumps(report, indent=2, default=float))
    return 0 if report["passed"] else 2


def cmd_sim(args) -> int:
    from .backends.isaac_sim import run_isaac_sim

    cfg = Config.load(args.config)
    return run_isaac_sim(cfg, args.trajectory, headless=args.headless, realtime=args.realtime)


def cmd_execute(args) -> int:
    cfg = Config.load(args.config)
    if args.backend == "ros2":
        from .backends.ros2_backend import run_ros2

        return run_ros2(cfg, args.trajectory, speed_ratio=args.speed_ratio, dry_run=args.dry_run)
    from .backends.dobot_tcp import run_direct

    return run_direct(
        cfg,
        args.trajectory,
        ip=args.ip,
        speed_ratio=args.speed_ratio,
        dry_run=args.dry_run,
        feedback_port=args.feedback_port,
    )


def cmd_plot(args) -> int:
    from .backends.replay import write_path_preview

    cfg = Config.load(args.config)
    trajectory = Trajectory.load(args.trajectory)
    out = write_path_preview(cfg, trajectory, Path(args.out))
    print(f"preview written: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dobot-pathgen",
        description="Dobot CR5A dental-cleaning path generation demo",
    )
    parser.add_argument("--config", default="config/robot.yaml", help="robot/scene config")
    sub = parser.add_subparsers(dest="command", required=True)

    p_calib = sub.add_parser("calib", help="inspect a TCP / model calibration file")
    p_calib.add_argument("--file", required=True, help="JSON / YAML / plain-text calibration file")
    p_calib.add_argument("--apply", action="store_true",
                         help="write the parsed values into config/robot.yaml")
    p_calib.set_defaults(func=cmd_calib)

    p_plan = sub.add_parser("plan", help="generate a trajectory and a validation report")
    p_plan.add_argument("--task", default="config/tasks/demo_occlusal_patch.yaml")
    p_plan.add_argument("--out", default="out/trajectory.json")
    p_plan.add_argument("--speed", type=float, default=None, help="override speed [m/s]")
    p_plan.set_defaults(func=cmd_plan)

    p_check = sub.add_parser("check", help="validate an existing trajectory")
    p_check.add_argument("--trajectory", default="out/trajectory.json")
    p_check.set_defaults(func=cmd_check)

    p_sim = sub.add_parser("sim", help="replay the trajectory in Isaac Sim")
    p_sim.add_argument("--trajectory", default="out/trajectory.json")
    p_sim.add_argument("--headless", action="store_true")
    p_sim.add_argument("--realtime", action="store_true", help="advance at wall-clock speed")
    p_sim.set_defaults(func=cmd_sim)

    p_exec = sub.add_parser("execute", help="run the trajectory on the real robot")
    p_exec.add_argument("--trajectory", default="out/trajectory.json")
    p_exec.add_argument("--backend", choices=["ros2", "direct"], default="direct")
    p_exec.add_argument("--ip", default=None, help="controller IP for the direct backend")
    p_exec.add_argument("--speed-ratio", type=float, default=10.0,
                        help="percent of programmed speed passed to the controller")
    p_exec.add_argument("--feedback-port", type=int, default=30004)
    p_exec.add_argument("--dry-run", action="store_true", help="print commands without sending")
    p_exec.set_defaults(func=cmd_execute)

    p_plot = sub.add_parser("preview", help="write a lightweight HTML/PLY path preview")
    p_plot.add_argument("--trajectory", default="out/trajectory.json")
    p_plot.add_argument("--out", default="out/preview")
    p_plot.set_defaults(func=cmd_plot)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
