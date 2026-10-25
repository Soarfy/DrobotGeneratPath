"""Dobot TCP/IP V4 direct backend (no ROS required).

Implements the subset of the dashboard protocol the demo needs:
    EnableRobot, SpeedFactor, ServoJ, Stop, ClearError
and the 30004 real-time feedback stream for joint monitoring.

Protocol reference: Dobot TCP/IP V4.x secondary development interface guide.
The controller accepts one client on 29999, so make sure DobotStudio Pro is
disconnected before running this.
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..planning import Trajectory


class DobotError(RuntimeError):
    pass


class DashboardClient:
    """Synchronous dashboard connection (port 29999)."""

    def __init__(self, ip: str, port: int = 29999, timeout: float = 5.0):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def connect(self):
        self.sock = socket.create_connection((self.ip, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)
        # The controller greets with "Control Mode,0,RobotMode(...);"
        try:
            self._recv_until_semicolon()
        except socket.timeout:
            pass

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def _recv_until_semicolon(self) -> str:
        assert self.sock is not None
        buf = b""
        while not buf.endswith(b";"):
            chunk = self.sock.recv(1024)
            if not chunk:
                raise DobotError("dashboard connection closed by controller")
            buf += chunk
        return buf.decode("utf-8", errors="ignore")

    def send(self, command: str) -> str:
        if self.sock is None:
            raise DobotError("dashboard not connected")
        with self._lock:
            self.sock.sendall(command.encode("utf-8"))
            reply = self._recv_until_semicolon()
        return reply

    def command_ok(self, command: str) -> int:
        reply = self.send(command)
        # Reply format: "ErrorID,{...};"  ErrorID 0 means success.
        head = reply.split(",", 1)[0].strip()
        try:
            return int(head)
        except ValueError:
            raise DobotError(f"unexpected reply to {command!r}: {reply!r}")

    def enable(self):
        return self.command_ok("EnableRobot()")

    def speed_factor(self, ratio: int):
        ratio = max(1, min(100, int(ratio)))
        return self.command_ok(f"SpeedFactor({ratio})")

    def clear_error(self):
        return self.command_ok("ClearError()")

    def stop(self):
        return self.command_ok("Stop()")

    def get_angle(self):
        reply = self.send("GetAngle()")
        # Reply: "0,{j1,j2,j3,j4,j5,j6};"
        start = reply.find("{")
        end = reply.find("}", start)
        if start < 0 or end < 0:
            raise DobotError(f"unexpected GetAngle reply: {reply!r}")
        values = [float(v) for v in reply[start + 1:end].split(",") if v.strip()]
        return np.array(values[:6], dtype=float)

    def mov_j(self, joints_deg, speed_ratio: int = 10) -> int:
        """Joint move; use this to reach the trajectory start, never ServoJ."""
        args = ",".join(f"{v:.3f}" for v in joints_deg)
        return self.command_ok(f"MovJ({args},SpeedFactor={int(speed_ratio)})")

    def servo_j(self, joints_deg, t: float, aheadtime: float = 50.0, gain: float = 500.0) -> int:
        # ServoJ(J1..J6, t, aheadtime, gain)
        # t        : duration of this segment [0.004, 3600] s
        # aheadtime: derivative term, default 50
        # gain     : proportional term, default 500
        t = min(max(float(t), 0.004), 3600.0)
        args = ",".join(f"{v:.3f}" for v in joints_deg)
        cmd = f"ServoJ({args},t={t:.3f},aheadtime={aheadtime:.0f},gain={gain:.0f})"
        return self.command_ok(cmd)


class FeedbackReader(threading.Thread):
    """Reads the 30004 real-time stream and keeps the latest joint angles."""

    def __init__(self, ip: str, port: int = 30004):
        super().__init__(daemon=True)
        self.ip = ip
        self.port = port
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._joints_deg = np.zeros(6)
        self._num = 0
        self._error: Optional[str] = None

    def run(self):
        while not self._stop.is_set():
            try:
                with socket.create_connection((self.ip, self.port), timeout=3.0) as sock:
                    sock.settimeout(2.0)
                    while not self._stop.is_set():
                        data = self._recv_packet(sock)
                        if data is None:
                            break
                        self._parse(data)
            except (OSError, socket.timeout) as exc:
                self._error = str(exc)
                self._stop.wait(0.5)

    def _recv_packet(self, sock: socket.socket) -> Optional[bytes]:
        header = b""
        while len(header) < 8:
            chunk = sock.recv(8 - len(header))
            if not chunk:
                return None
            header += chunk
        length = struct.unpack_from("<H", header, 0)[0]
        body = b""
        while len(body) < length - 8:
            chunk = sock.recv(length - 8 - len(body))
            if not chunk:
                return None
            body += chunk
        return header + body

    def _parse(self, packet: bytes):
        # V4 real-time packet (1440 bytes): q_actual starts at byte 432.
        # Layout (little-endian doubles): q_target@192, qd_target@240,
        # qdd_target@288, i_target@336, m_target@384, q_actual@432.
        if len(packet) < 480:
            return
        joints = struct.unpack_from("<6d", packet, 432)
        with self._lock:
            self._joints_deg = np.array(joints, dtype=float)
            self._num += 1

    def latest(self):
        with self._lock:
            return self._joints_deg.copy(), self._num

    def stop(self):
        self._stop.set()


def _interpolate_uniform(points: np.ndarray, times: np.ndarray, dt: float):
    """Resample a joint trajectory onto a uniform ServoJ grid."""
    if len(points) < 2:
        raise ValueError("trajectory needs at least 2 points")
    t_end = float(times[-1])
    grid = np.arange(0.0, t_end + 1e-9, dt)
    if grid[-1] < t_end:
        grid = np.append(grid, t_end)
    out = np.column_stack([np.interp(grid, times, points[:, j]) for j in range(points.shape[1])])
    return out, grid


def run_direct(
    cfg,
    trajectory_path,
    ip: Optional[str] = None,
    speed_ratio: float = 10.0,
    dry_run: bool = False,
    feedback_port: int = 30004,
    servo_dt: float = 0.02,
) -> int:
    trajectory = Trajectory.load(trajectory_path)
    q = trajectory.positions()
    t = np.array([w.time_s for w in trajectory.waypoints])
    deg = np.rad2deg(q)
    grid_deg, grid_t = _interpolate_uniform(deg, t, servo_dt)

    print(f"trajectory : {len(q)} points, {t[-1]:.1f} s")
    print(f"ServoJ     : {len(grid_deg)} samples @ {servo_dt * 1000:.0f} ms")
    print(f"speed      : {speed_ratio:.0f}%")

    if dry_run:
        print("\nFirst 8 commands:")
        for i in range(min(8, len(grid_deg))):
            print(f"  ServoJ({grid_deg[i, 0]:.3f},...,{grid_deg[i, 5]:.3f},t={servo_dt:.3f})")
        print("dry run: nothing sent")
        return 0

    target_ip = ip or cfg.raw.get("robot", {}).get("ip", "192.168.5.1")
    print(f"connecting to {target_ip} ...")
    dash = DashboardClient(target_ip)
    dash.connect()
    reader = FeedbackReader(target_ip, feedback_port)
    reader.start()

    try:
        dash.clear_error()
        dash.enable()
        dash.speed_factor(int(speed_ratio))

        # Reach the trajectory start with a normal joint move. Using ServoJ for
        # a long approach would command one huge sweep instead of a safe move.
        start_deg = grid_deg[0]
        print(f"MovJ to start at {speed_ratio:.0f}% ...")
        dash.mov_j(start_deg, speed_ratio=int(speed_ratio))
        deadline = time.time() + 60.0
        while time.time() < deadline:
            current = dash.get_angle()
            if float(np.max(np.abs(current - start_deg))) < 1.0:
                break
            time.sleep(0.1)
        else:
            raise DobotError("robot did not reach the start configuration within 60 s")
        print("at start")

        t0 = time.perf_counter()
        for i, row in enumerate(grid_deg):
            target = t0 + grid_t[i]
            now = time.perf_counter()
            if target > now:
                time.sleep(target - now)
            dash.servo_j(row, t=servo_dt)
            if i % 50 == 0:
                joints, count = reader.latest()
                print(
                    f"  {i:5d}/{len(grid_deg)}  "
                    f"cmd6={row[-1]:7.2f}  fb6={joints[-1]:7.2f}  fb_packets={count}"
                )
        elapsed = time.perf_counter() - t0
        print(f"done in {elapsed:.1f} s")
    except KeyboardInterrupt:
        print("\ninterrupted, sending Stop()")
        try:
            dash.stop()
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
        reader.stop()
        dash.close()
    return 0
