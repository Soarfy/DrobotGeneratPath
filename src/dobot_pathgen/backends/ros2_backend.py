"""ROS2 backend: reuse the official Dobot driver and MoveIt action server.

This backend assumes the official workspace is running:

    export IP_address=192.168.5.1
    export DOBOT_TYPE=cr5
    ros2 launch dobot_bringup_v4 dobot_bringup_ros2.launch.py
    ros2 launch dobot_moveit dobot_moveit.launch.py

It publishes the planned trajectory on a FollowJointTrajectory action, so the
same JSON file drives both this backend and the direct TCP one.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from ..planning import Trajectory


def run_ros2(cfg, trajectory_path, speed_ratio: float = 10.0, dry_run: bool = False) -> int:
    try:
        import rclpy
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from builtin_interfaces.msg import Duration
    except ImportError:
        print(
            "ROS2 Python packages not found. Run this command inside the ROS2 "
            "environment (Ubuntu 22.04 + ROS2 Humble), not the Windows venv.",
            file=sys.stderr,
        )
        return 1

    trajectory = Trajectory.load(trajectory_path)
    joint_names = trajectory.joint_names

    msg = JointTrajectory()
    msg.joint_names = joint_names
    for w in trajectory.waypoints:
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in w.positions]
        point.velocities = [0.0] * len(joint_names)
        sec = int(w.time_s)
        nsec = int((w.time_s - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nsec)
        msg.points.append(point)

    node_name = f"{cfg.robot.name}_group_controller"
    action_name = f"/{node_name}/follow_joint_trajectory"

    print(f"trajectory : {len(trajectory.waypoints)} points, {trajectory.duration:.1f} s")
    print(f"action     : {action_name}")
    if dry_run:
        print("dry run: action not sent")
        return 0

    rclpy.init()
    node = Node("dobot_pathgen_client")
    client = ActionClient(node, FollowJointTrajectory, action_name)

    try:
        if not client.wait_for_server(timeout_sec=10.0):
            print(f"action server not available: {action_name}", file=sys.stderr)
            return 1

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = msg
        goal.goal_time_tolerance = Duration(sec=2, nanosec=0)

        print("sending goal ...")
        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            print("goal rejected", file=sys.stderr)
            return 1

        print("goal accepted, waiting for completion ...")
        result_future = goal_handle.get_result_async()
        deadline = time.time() + trajectory.duration + 60.0
        while not result_future.done() and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not result_future.done():
            print("timeout waiting for the trajectory to finish", file=sys.stderr)
            return 1

        result = result_future.result()
        code = result.result.error_code
        print(f"finished with error_code={code}")
        return 0 if code == 0 else 2
    except KeyboardInterrupt:
        print("interrupted")
        return 130
    finally:
        node.destroy_node()
        rclpy.shutdown()
