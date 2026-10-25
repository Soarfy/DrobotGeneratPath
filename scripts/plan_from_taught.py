"""Plan from the cell's taught trajectories, region by region.

Each region in `hand_eye_calibration/UsmileUi/default/<region>/ee_poses.txt` is
a short taught path the real controller already runs: flange pose in the robot
base frame, millimetres and ZYX degrees, one waypoint per line.

Solving several regions as one long sequence does not work: each region sits on
its own IK branch, and gluing them together makes the arm snap 150+ deg at every
boundary. So each region is solved independently -- faithful to what the machine
does -- and the regions are joined by an explicit joint-space transfer that the
controller executes as a smooth ServoJ stream.

    python scripts/plan_from_taught.py --regions all
    python scripts/plan_from_taught.py --regions center leftside
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dobot_pathgen import se3
from dobot_pathgen.config import Config
from dobot_pathgen.ik import solve_ik_all_branches
from dobot_pathgen.planning import Trajectory, Waypoint, compile_trajectory, validate_trajectory
from dobot_pathgen.urdf_chain import apply_position_limits, build_chain


TAUGHT_DIR = Path(r"D:\CompanyProject\hand_eye_calibration\UsmileUi\default")

# Brushing order: front first, then the right side, then the left.
REGION_ORDER = [
    "center",
    "sideahead",
    "rightup",
    "rightside",
    "rightinside",
    "leftup",
    "leftside",
    "leftinside",
]


def load_region_poses(name: str):
    """Flange poses (base frame) for one region."""
    path = TAUGHT_DIR / name / "ee_poses.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.loadtxt(path)
    return [
        se3.make_transform(se3.rpy_deg_to_matrix(row[3:6]), row[:3] / 1000.0)
        for row in data
    ]


def solve_region_branches(chain, poses, restarts=80):
    """All IK solutions for every pose in one region."""
    out = []
    for i, T in enumerate(poses):
        out.append(solve_ik_all_branches(chain, T, restarts=restarts, seed=i))
    return out


def global_branch_path(chain, per_pose_solutions, labels, max_step_deg=45.0):
    """Pick one solution per pose across ALL regions with a single DP.

    Solving each region separately and concatenating fails at region borders:
    the last pose of one region and the first of the next often sit in
    different elbow/wrist families. One DP over the whole sequence finds the
    branch that connects everything.
    """
    span = chain.upper - chain.lower
    anchor = np.asarray(per_pose_solutions[0][0], dtype=float)
    wrapped = []
    for sols in per_pose_solutions:
        row = []
        for q in sols:
            q = np.asarray(q, dtype=float)
            row.append(chain.clamp(anchor + (q - anchor + np.pi) % (2.0 * np.pi) - np.pi))
        wrapped.append(row)

    max_step = np.deg2rad(max_step_deg)
    cost = [{j: 0.0 for j in range(len(wrapped[0]))}]
    back = [{}]
    for k in range(1, len(wrapped)):
        cur, par = {}, {}
        for j, qc in enumerate(wrapped[k]):
            bp, bv = None, float("inf")
            for jp, qp in enumerate(wrapped[k - 1]):
                if jp not in cost[-1]:
                    continue
                d = float(np.max(np.abs(qc - qp)))
                if d > max_step:
                    continue
                v = cost[-1][jp] + d
                if v < bv:
                    bp, bv = jp, v
            if bp is not None:
                cur[j], par[j] = bv, bp
        if not cur:
            # Bridge with the cheapest pair so the region is not dropped.
            for j, qc in enumerate(wrapped[k]):
                bp = min(cost[-1], key=lambda jp: float(np.max(np.abs(qc - wrapped[k - 1][jp]))))
                cur[j] = cost[-1][bp] + float(np.max(np.abs(qc - wrapped[k - 1][bp])))
                par[j] = bp
        cost.append(cur)
        back.append(par)

    chosen = {len(wrapped) - 1: min(cost[-1], key=lambda j: cost[-1][j])}
    for k in range(len(wrapped) - 1, 0, -1):
        chosen[k - 1] = back[k][chosen[k]]

    seq = []
    for k, sols in enumerate(wrapped):
        q = np.asarray(sols[chosen[k]], dtype=float).copy()
        if seq:
            prev = seq[-1]
            for j in range(len(q)):
                if span[j] >= 2.0 * np.pi - 1e-6:
                    q[j] = prev[j] + (q[j] - prev[j] + np.pi) % (2.0 * np.pi) - np.pi
            q = chain.clamp(q)
        seq.append(q)
    return np.array(seq)


def insert_transfers(q_seq, labels, steps=40):
    """Blend region boundaries with a short joint-space transfer."""
    out, out_labels = [], []
    prev_label = None
    for k in range(len(q_seq)):
        lbl = labels[k]
        if prev_label is not None and lbl != prev_label:
            for s in range(1, steps + 1):
                a = s / (steps + 1)
                e = 0.5 - 0.5 * np.cos(np.pi * a)
                out.append(q_seq[k - 1] + e * (q_seq[k] - q_seq[k - 1]))
                out_labels.append("transfer")
        out.append(q_seq[k])
        out_labels.append(lbl)
        prev_label = lbl
    return np.array(out), out_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config/robot.yaml"))
    parser.add_argument("--regions", nargs="+", default=["center"],
                        help=f"region names or 'all'. Known: {', '.join(REGION_ORDER)}")
    parser.add_argument("--out", default=str(ROOT / "out/taught_trajectory.json"))
    parser.add_argument("--speed", type=float, default=0.01)
    parser.add_argument("--restarts", type=int, default=16)
    parser.add_argument("--transfer-steps", type=int, default=40)
    args = parser.parse_args()

    regions = REGION_ORDER if args.regions == ["all"] else args.regions
    unknown = [r for r in regions if r not in REGION_ORDER]
    if unknown:
        print(f"unknown regions: {unknown}")
        print(f"known: {', '.join(REGION_ORDER)}")
        return 1

    cfg = Config.load(args.config)
    chain = build_chain(cfg.robot.urdf, cfg.robot.tip_link)
    chain = apply_position_limits(chain, cfg.robot.position_limits)

    all_poses = []
    labels = []
    for name in regions:
        poses = load_region_poses(name)
        all_poses.extend(poses)
        labels.extend([name] * len(poses))
        print(f"  {name:12s} {len(poses):3d} poses")

    print(f"total poses: {len(all_poses)}; enumerating IK branches ...")
    per_pose = [
        solve_ik_all_branches(chain, T, restarts=80, seed=i)
        for i, T in enumerate(all_poses)
    ]
    empty = [i for i, s in enumerate(per_pose) if not s]
    if empty:
        print(f"  {len(empty)} poses have no IK solution: {empty[:10]}")
        return 2

    q_seq = global_branch_path(chain, per_pose, labels, max_step_deg=45.0)
    q_all, region_labels = insert_transfers(q_seq, labels, steps=args.transfer_steps)
    assert len(q_all) == len(region_labels), (
        f"label/waypoint mismatch: {len(q_all)} poses vs {len(region_labels)} labels"
    )

    # Timing must satisfy acceleration too, not just velocity. A pure
    # velocity-based dt produced 1.6x the acceleration limit at the region
    # transfers, so scale each step by a comfortable acceleration bound.
    ramp = np.deg2rad(3.0)          # per-step allowance, rad
    v_max = 0.35                    # rad/s
    waypoints = []
    t = 0.0
    for i, q in enumerate(q_all):
        if i > 0:
            dq = float(np.max(np.abs(q - q_all[i - 1])))
            # Time for a trapezoid reaching dq without exceeding v_max or the
            # acceleration implied by `ramp`.
            t += max(0.08, dq / v_max, 1.6 * np.sqrt(max(dq, 1e-9) / ramp))
        waypoints.append(Waypoint(time_s=t, positions=[float(v) for v in q]))
    trajectory = Trajectory(joint_names=chain.joint_names, waypoints=waypoints)

    report = validate_trajectory(chain, trajectory)
    trajectory.meta.update(
        {
            "source": "hand_eye_calibration/UsmileUi/default/*/ee_poses.txt",
            "regions": regions,
            "validation": report,
            "region_labels": region_labels,
        }
    )
    out = Path(args.out)
    trajectory.save(out)

    steps = np.rad2deg(np.abs(np.diff(q_all, axis=0)).max(axis=1))
    print()
    print(f"regions       : {', '.join(regions)}")
    print(f"waypoints     : {len(waypoints)}")
    print(f"duration      : {trajectory.duration:.1f} s")
    print(f"max joint step: {steps.max():.1f} deg")
    print(f"IK dropped    : {len(empty)}")
    print(f"validation    : {'PASS' if report['passed'] else 'FAIL'}")
    print(f"trajectory    : {out}")
    return 0 if report["passed"] and not empty else 2


if __name__ == "__main__":
    sys.exit(main())
