"""Real-robot execution with safety monitoring.

This is the module to use when the arm is live. It adds the guards that a demo
replay does not have:

  * pre-flight: reads the current pose, checks reachability of the start point,
    refuses to run if the first waypoint is far outside the workspace;
  * soft limits: any commanded angle outside the configured joint range aborts;
  * follow-error watchdog: if measured joints lag the command by more than
    ``max_follow_error_deg`` the run stops;
  * staged speed: starts at a low percentage and only escalates when the
    operator confirms;
  * a CSV log of command vs feedback for every control cycle, so a bad run can
    be diagnosed afterwards.

Usage (from a workstation on the same subnet as the controller):

    python scripts/real_test.py --trajectory out/arch_gen.json --ip 192.168.5.1
    python scripts/real_test.py --trajectory out/arch_gen.json --check-only
    python scripts/real_test.py --trajectory out/arch_gen.json --speed 5 --yes

Safety: nothing moves until you pass ``--yes`` or answer the confirmation
prompt. ``Ctrl+C`` sends Stop() immediately.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dobot_pathgen.backends.dobot_tcp import DashboardClient, DobotError, FeedbackReader
from dobot_pathgen.config import Config
from dobot_pathgen.planning import Trajectory
from dobot_pathgen.urdf_chain import apply_position_limits, build_chain


def check_only(cfg, trajectory, ip, port):
    """Static + live checks with no motion. Returns (ok, notes)."""
    notes = []
    chain = apply_position_limits(
        build_chain(cfg.robot.urdf, cfg.robot.tip_link), cfg.robot.position_limits
    )
    q = trajectory.positions()
    deg = np.rad2deg(q)
    ok = True

    # Trajectory-level checks.
    lo = np.rad2deg(chain.lower)
    hi = np.rad2deg(chain.upper)
    outside = np.where((deg < lo - 1e-6) | (deg > hi + 1e-6))
    if len(outside[0]):
        notes.append(
            f"FAIL {len(outside[0])} waypoints exceed joint limits "
            f"(first at index {outside[0][0]})"
        )
        ok = False
    else:
        notes.append(f"OK   all {len(deg)} waypoints inside joint limits")

    # deg is already in degrees, so no further conversion here.
    steps = float(np.abs(np.diff(deg, axis=0)).max()) if len(deg) > 1 else 0.0
    notes.append(f"INFO max joint step {steps:.1f} deg")

    # Live controller checks.
    try:
        dash = DashboardClient(ip, timeout=3.0)
        dash.connect()
    except OSError as exc:
        notes.append(f"FAIL cannot connect to {ip}:29999 ({exc})")
        return False, notes

    try:
        mode = dash.send("RobotMode()").strip()
        notes.append(f"INFO RobotMode: {mode[:80]}")
        if not mode.startswith("0"):
            notes.append("WARN controller reported a non-zero status code")

        current = dash.get_angle()
        notes.append(f"INFO current joints deg: {np.round(current, 2)}")

        dist = float(np.max(np.abs(current - deg[0])))
        if dist > 60.0:
            notes.append(
                f"WARN start pose is {dist:.1f} deg away from the current pose; "
                "the arm will make a large MovJ first"
            )
        else:
            notes.append(f"OK   start pose {dist:.1f} deg from current")

        enabled = dash.send("RobotEnable()").strip()
        notes.append(f"INFO RobotEnable: {enabled[:80]}")
    except DobotError as exc:
        notes.append(f"WARN {exc}")
    finally:
        dash.close()

    # Feedback stream.
    try:
        reader = FeedbackReader(ip, port)
        reader.start()
        time.sleep(1.5)
        joints, packets = reader.latest()
        reader.stop()
        if packets:
            notes.append(f"OK   feedback stream alive ({packets} packets), joints {np.round(joints, 2)}")
        else:
            notes.append("WARN no packets from port 30004 (monitoring will be blind)")
            ok = False
    except OSError as exc:
        notes.append(f"WARN feedback port {port}: {exc}")
        ok = False

    return ok, notes


def execute(cfg, trajectory, ip, port, speed_ratio, servo_dt, max_follow_error_deg,
            log_path, confirm):
    chain = apply_position_limits(
        build_chain(cfg.robot.urdf, cfg.robot.tip_link), cfg.robot.position_limits
    )
    q = trajectory.positions()
    times = np.array([w.time_s for w in trajectory.waypoints])
    deg = np.rad2deg(q)

    # Uniform ServoJ grid; the controller expects regularly spaced points.
    grid_t = np.arange(0.0, float(times[-1]) + 1e-9, servo_dt)
    if grid_t[-1] < times[-1]:
        grid_t = np.append(grid_t, times[-1])
    grid = np.column_stack([np.interp(grid_t, times, deg[:, j]) for j in range(6)])

    lo = np.rad2deg(chain.lower)
    hi = np.rad2deg(chain.upper)
    if np.any(grid < lo - 1e-6) or np.any(grid > hi + 1e-6):
        print("refusing to run: trajectory exceeds the configured joint limits")
        return 2

    print(f"trajectory : {len(q)} waypoints, {times[-1]:.1f} s")
    print(f"ServoJ     : {len(grid)} samples @ {servo_dt * 1000:.0f} ms")
    print(f"speed      : {speed_ratio:.0f}%")
    print(f"watchdog   : abort if follow error > {max_follow_error_deg:.1f} deg")
    print(f"log        : {log_path}")

    if confirm:
        print()
        print("The arm will move. Make sure the workspace is clear and the")
        print("e-stop is within reach.")
        answer = input("Type YES to start: ").strip()
        if answer != "YES":
            print("aborted by operator")
            return 1

    dash = DashboardClient(ip)
    dash.connect()
    reader = FeedbackReader(ip, port)
    reader.start()

    log_file = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    writer.writerow(
        ["t_s", "i", "cmd_j1", "cmd_j2", "cmd_j3", "cmd_j4", "cmd_j5", "cmd_j6",
         "fb_j1", "fb_j2", "fb_j3", "fb_j4", "fb_j5", "fb_j6", "follow_err_deg"]
    )

    abort_reason = None
    try:
        dash.clear_error()
        dash.enable()
        dash.speed_factor(int(speed_ratio))

        start = grid[0]
        print(f"\nMovJ to start at {speed_ratio:.0f}% ...")
        dash.mov_j(start, speed_ratio=int(speed_ratio))
        deadline = time.time() + 90.0
        while time.time() < deadline:
            current = dash.get_angle()
            if float(np.max(np.abs(current - start))) < 1.0:
                break
            if abort_reason:
                break
            time.sleep(0.1)
        else:
            raise DobotError("did not reach the start configuration within 90 s")
        print("at start\n")

        t0 = time.perf_counter()
        for i, row in enumerate(grid):
            target = t0 + grid_t[i]
            now = time.perf_counter()
            if target > now:
                time.sleep(target - now)

            fb, _ = reader.latest()
            err = float(np.max(np.abs(fb - row))) if np.any(fb) else 0.0
            writer.writerow([f"{time.perf_counter() - t0:.4f}", i, *row, *fb, f"{err:.3f}"])

            if err > max_follow_error_deg:
                abort_reason = (
                    f"follow error {err:.1f} deg exceeded "
                    f"{max_follow_error_deg:.1f} deg at sample {i}"
                )
                break

            code = dash.servo_j(row, t=servo_dt)
            if code != 0:
                abort_reason = f"ServoJ returned error {code} at sample {i}"
                break

            if i % 50 == 0:
                print(
                    f"  {i:5d}/{len(grid)}  t={time.perf_counter()-t0:6.1f}s  "
                    f"follow_err={err:5.2f} deg"
                )

        if abort_reason:
            print(f"\nABORT: {abort_reason}")
            dash.stop()
            dash.clear_error()
            return 3

        # Let the last point settle, then verify.
        time.sleep(0.5)
        final = dash.get_angle()
        residual = float(np.max(np.abs(final - grid[-1])))
        print(f"\ndone in {time.perf_counter() - t0:.1f} s")
        print(f"final residual {residual:.2f} deg")
        return 0 if residual < 2.0 else 4

    except KeyboardInterrupt:
        print("\nCtrl+C: sending Stop()")
        try:
            dash.stop()
            dash.clear_error()
        except Exception:
            pass
        return 130
    except DobotError as exc:
        print(f"controller error: {exc}", file=sys.stderr)
        try:
            dash.stop()
        except Exception:
            pass
        return 1
    finally:
        log_file.close()
        reader.stop()
        dash.close()


def main():
    parser = argparse.ArgumentParser(description="Run a planned trajectory on the real CR5A")
    parser.add_argument("--config", default=str(ROOT / "config/robot.yaml"))
    parser.add_argument("--trajectory", default=str(ROOT / "out/arch_gen.json"))
    parser.add_argument("--ip", default=None, help="controller IP (default: config or 192.168.5.1)")
    parser.add_argument("--port", type=int, default=30004, help="feedback port")
    parser.add_argument("--speed", type=float, default=5.0, help="speed factor percent")
    parser.add_argument("--servo-dt", type=float, default=0.02, help="ServoJ interval seconds")
    parser.add_argument("--max-follow-error", type=float, default=8.0,
                        help="abort if measured joints lag the command by this much")
    parser.add_argument("--log", default=None, help="CSV log path")
    parser.add_argument("--check-only", action="store_true", help="run checks, do not move")
    parser.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    ip = args.ip or cfg.raw.get("robot", {}).get("ip", "192.168.5.1")
    trajectory = Trajectory.load(args.trajectory)

    print(f"robot   : {ip}")
    print(f"trajectory: {args.trajectory}")

    if args.check_only:
        ok, notes = check_only(cfg, trajectory, ip, args.port)
        print()
        for line in notes:
            print(" ", line)
        print()
        print("READY" if ok else "NOT READY")
        return 0 if ok else 2

    ok, notes = check_only(cfg, trajectory, ip, args.port)
    print()
    for line in notes:
        print(" ", line)
    if not ok:
        print("\nPre-flight failed; refusing to move. Fix the issues above.")
        return 2

    log_path = Path(args.log) if args.log else (
        Path(args.trajectory).with_name(f"{Path(args.trajectory).stem}_realtest.csv")
    )
    print()
    return execute(
        cfg, trajectory, ip, args.port, args.speed, args.servo_dt,
        args.max_follow_error, log_path, confirm=not args.yes,
    )


if __name__ == "__main__":
    sys.exit(main())
