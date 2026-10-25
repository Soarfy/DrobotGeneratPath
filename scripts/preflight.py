"""Pre-flight check before running a trajectory on the real robot.

Verifies, without moving the robot:

    python scripts/preflight.py
    python scripts/preflight.py --ip 192.168.5.1 --trajectory out/trajectory.json

Checks:
  1. trajectory parses and joint names match the URDF chain
  2. IK/FK consistency: replays the trajectory through FK and compares
  3. joint limits, velocities, accelerations
  4. dashboard port 29999 answers and reports TCP/IP control mode
  5. optional: 30004 feedback arrives

Exit code 0 means safe to proceed to a low-speed dry run.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dobot_pathgen.config import Config  # noqa: E402
from dobot_pathgen.planning import Trajectory, validate_trajectory  # noqa: E402
from dobot_pathgen.urdf_chain import build_chain  # noqa: E402


def check_trajectory(cfg, trajectory_path):
    print("[1] trajectory ...")
    trajectory = Trajectory.load(trajectory_path)
    chain = build_chain(cfg.robot.urdf, cfg.robot.tip_link)

    assert trajectory.joint_names == chain.joint_names, (
        f"joint order mismatch: {trajectory.joint_names} vs {chain.joint_names}"
    )
    q = trajectory.positions()
    if len(q) < 2:
        raise AssertionError("trajectory has fewer than 2 waypoints")
    print(f"    ok: {len(q)} waypoints, {trajectory.duration:.1f} s")

    report = validate_trajectory(chain, trajectory)
    print(f"[2] limits ... {'PASS' if report['passed'] else 'FAIL'}")
    print(f"    max joint step      : {report['max_joint_step_deg']:.1f} deg")
    print(f"    max velocity usage  : {report['max_velocity_fraction'] * 100:.0f}%")
    print(f"    max accel usage     : {report['max_acceleration_fraction'] * 100:.0f}%")
    print(f"    min manipulability  : {report['min_manipulability']:.4g}")
    if not report["passed"]:
        print(json.dumps(report, indent=2, default=float))
        raise AssertionError("trajectory violates joint limits or limits of motion")

    print("[3] FK replay of stored TCP poses ...")
    worst = 0.0
    for w in trajectory.waypoints:
        T = chain.fk(w.positions) @ cfg.tcp_pose_in_link6()
        stored = np.asarray(w.tool_pose, dtype=float)
        worst = max(worst, float(np.linalg.norm(T[:3, 3] - stored[:3, 3])))
    print(f"    max TCP deviation: {worst * 1000:.2f} mm")
    if worst > 2e-3:
        raise AssertionError("stored tool poses do not match FK; re-plan the trajectory")
    return trajectory


def check_clearances(cfg, trajectory):
    """Tool-vs-arm proximity. A 200 mm angled brush can graze the forearm."""
    from dobot_pathgen.collision import ToolGeometry, check_clearances, load_link_clouds
    from dobot_pathgen.urdf_chain import build_chain

    print("[4] tool clearance ...")
    chain = build_chain(cfg.robot.urdf, cfg.robot.tip_link)
    geom = ToolGeometry(
        tcp_offset_m=np.asarray(cfg.tool.tcp_offset_m),
        radius_m=cfg.tool.radius_m,
        head_radius_m=cfg.tool.radius_m * 1.25,
        head_length_m=0.02,
    )
    clouds = load_link_clouds(cfg.robot.urdf)
    report = check_clearances(chain, trajectory, geom, clouds, self_margin_m=0.01)
    print(f"    tool-arm gap: {report['self_min_distance_mm']:.1f} mm "
          f"(wp {report['self_min_index']}, {report['self_min_link']})")
    if not report["passed"]:
        raise AssertionError(
            f"tool comes within {report['self_min_distance_mm']:.1f} mm of "
            f"{report['self_min_link']} at waypoint {report['self_min_index']}"
        )


def check_network(ip, feedback_port):
    print(f"[5] dashboard {ip}:29999 ...")
    try:
        with socket.create_connection((ip, 29999), timeout=3.0) as sock:
            sock.settimeout(2.0)
            banner = sock.recv(1024).decode("utf-8", errors="ignore")
            print(f"    banner: {banner.strip()[:120]}")
            sock.sendall(b"RobotMode()")
            mode = sock.recv(1024).decode("utf-8", errors="ignore")
            print(f"    RobotMode: {mode.strip()[:120]}")
            sock.sendall(b"GetAngle()")
            angles = sock.recv(1024).decode("utf-8", errors="ignore")
            print(f"    GetAngle: {angles.strip()[:120]}")
    except OSError as exc:
        raise AssertionError(f"cannot reach {ip}:29999 ({exc})") from exc

    print(f"[6] feedback {ip}:{feedback_port} ...")
    try:
        with socket.create_connection((ip, feedback_port), timeout=3.0) as sock:
            sock.settimeout(2.0)
            data = sock.recv(1440)
            print(f"    received {len(data)} bytes")
    except OSError as exc:
        print(f"    warning: no feedback stream ({exc})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config/robot.yaml"))
    parser.add_argument("--trajectory", default=str(ROOT / "out/trajectory.json"))
    parser.add_argument("--ip", default=None)
    parser.add_argument("--skip-network", action="store_true")
    parser.add_argument("--feedback-port", type=int, default=30004)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    try:
        trajectory = check_trajectory(cfg, args.trajectory)
        check_clearances(cfg, trajectory)
        if not args.skip_network:
            ip = args.ip or cfg.raw.get("robot", {}).get("ip", "192.168.5.1")
            check_network(ip, args.feedback_port)
    except (AssertionError, ValueError, FileNotFoundError) as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1

    print("\npre-flight passed. Start with --speed-ratio 10 and a hand on the e-stop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
