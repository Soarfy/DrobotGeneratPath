"""Damped-least-squares numerical IK for a serial chain.

Deliberately small and dependency-free. The demo does not need a full planning
stack; it needs deterministic, inspectable IK with joint-limit handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from . import se3
from .urdf_chain import Chain


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    position_error_m: float
    orientation_error_rad: float
    iterations: int
    residual: float
    used_seed: int


def _pose_error(T_current: np.ndarray, T_target: np.ndarray):
    dp = T_target[:3, 3] - T_current[:3, 3]
    dR = T_target[:3, :3] @ T_current[:3, :3].T
    dr = se3.rotvec_from_matrix(dR)
    return dp, dr


def _jacobian(chain: Chain, q: np.ndarray) -> np.ndarray:
    """Geometric Jacobian (world frame) for a serial revolute chain."""
    points, axes, T_tip = chain.joint_frames(q)
    p_tip = T_tip[:3, 3]
    J = np.zeros((6, chain.dof))
    for i in range(chain.dof):
        J[:3, i] = np.cross(axes[i], p_tip - points[i])
        J[3:, i] = axes[i]
    return J


def solve_ik(
    chain: Chain,
    T_target: np.ndarray,
    seed,
    max_iter: int = 200,
    tol_pos: float = 1e-4,
    tol_rot: float = 1e-3,
    damping: float = 0.05,
    max_step_rad: float = 0.35,
) -> IKResult:
    """Solve one IK problem from a single seed."""
    q = chain.clamp(np.asarray(seed, dtype=float))
    best = q.copy()
    best_res = np.inf
    best_iter = 0

    for it in range(max_iter):
        T = chain.fk(q)
        dp, dr = _pose_error(T, T_target)
        err = np.concatenate([dp, dr])
        res = float(np.linalg.norm(dp) + np.linalg.norm(dr))
        if res < best_res:
            best_res = res
            best = q.copy()
            best_iter = it
        if np.linalg.norm(dp) < tol_pos and np.linalg.norm(dr) < tol_rot:
            return IKResult(q, True, float(np.linalg.norm(dp)), float(np.linalg.norm(dr)), it, res, 0)

        J = _jacobian(chain, q)
        # Weight position and orientation so both converge at similar rates.
        W = np.diag([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])
        A = W @ J
        b = W @ err
        dq = A.T @ np.linalg.solve(A @ A.T + (damping ** 2) * np.eye(6), b)

        step_norm = float(np.linalg.norm(dq))
        if step_norm > max_step_rad:
            dq *= max_step_rad / step_norm

        # Backtracking line search on the combined residual.
        alpha = 1.0
        improved = False
        for _ in range(8):
            q_try = chain.clamp(q + alpha * dq)
            T_try = chain.fk(q_try)
            dp_t, dr_t = _pose_error(T_try, T_target)
            res_try = float(np.linalg.norm(dp_t) + np.linalg.norm(dr_t))
            if res_try < res:
                q = q_try
                improved = True
                break
            alpha *= 0.5
        if not improved:
            # Damping increase would be the alternative; stop this seed.
            break

    T = chain.fk(best)
    dp, dr = _pose_error(T, T_target)
    success = np.linalg.norm(dp) < tol_pos and np.linalg.norm(dr) < tol_rot
    return IKResult(
        best,
        success,
        float(np.linalg.norm(dp)),
        float(np.linalg.norm(dr)),
        best_iter,
        best_res,
        0,
    )


def _random_seeds(chain: Chain, n: int, rng: np.random.Generator, reference=None) -> List[np.ndarray]:
    seeds: List[np.ndarray] = []
    if reference is not None:
        seeds.append(chain.clamp(reference))
    seeds.append(np.zeros(chain.dof))
    span = chain.upper - chain.lower
    span = np.where(np.isfinite(span) & (span > 1e-6), span, 2.0 * np.pi)
    for _ in range(n):
        seeds.append(chain.lower + rng.random(chain.dof) * span)
    return seeds


def _wrap_near(q: np.ndarray, reference: np.ndarray, chain: Chain) -> np.ndarray:
    """Shift each revolute joint by whole turns toward ``reference``.

    Only joints that can physically do so (range >= 2*pi) are shifted, and the
    shifted value is kept only if it stays inside the joint limits, so a wrist
    flip never turns into a 350 degree sweep nor an out-of-range command.
    """
    q = np.asarray(q, dtype=float).copy()
    if reference is None:
        return q
    span = chain.upper - chain.lower
    for j in range(len(q)):
        if span[j] >= 2.0 * np.pi - 1e-6:
            candidates = [q[j]]
            for turns in (-2, -1, 1, 2):
                candidates.append(q[j] + turns * 2.0 * np.pi)
            valid = [c for c in candidates if chain.lower[j] - 1e-9 <= c <= chain.upper[j] + 1e-9]
            if valid:
                q[j] = min(valid, key=lambda c: abs(c - reference[j]))
    return q


def solve_ik_tracking(
    chain: Chain,
    T_target: np.ndarray,
    reference,
    restarts: int = 24,
    seed: int = 0,
    max_joint_jump_rad: float = 1.2,
    max_iter: int = 200,
) -> IKResult:
    """Continuity-first IK for a waypoint sequence.

    Only solutions within ``max_joint_jump_rad`` of the previous configuration
    are accepted. A wrist/elbow branch flip would otherwise turn into a huge
    ServoJ command even though the Cartesian step was tiny. If no continuous
    solution exists the result is marked as a failure instead of silently
    switching branch.
    """
    ref_arr = None if reference is None else np.asarray(reference, dtype=float)

    if ref_arr is not None:
        local = solve_ik(chain, T_target, ref_arr, max_iter=max_iter)
        if local.success:
            local.q = _wrap_near(local.q, ref_arr, chain)
            jump = float(np.max(np.abs(local.q - ref_arr)))
            if jump <= max_joint_jump_rad:
                return local

    result = solve_ik_multistart(
        chain, T_target, reference=ref_arr, restarts=restarts, seed=seed, max_iter=max_iter
    )
    if not result.success:
        return result
    result.q = _wrap_near(result.q, ref_arr, chain)
    if ref_arr is not None:
        jump = float(np.max(np.abs(result.q - ref_arr)))
        if jump > max_joint_jump_rad:
            result.success = False
            result.position_error_m = float(np.linalg.norm(T_target[:3, 3] - chain.fk(result.q)[:3, 3]))
    return result


def solve_ik_multistart(
    chain: Chain,
    T_target: np.ndarray,
    reference=None,
    restarts: int = 24,
    seed: int = 0,
    joint_weights: Optional[Sequence[float]] = None,
    max_iter: int = 200,
) -> IKResult:
    """Try several seeds, preferring solutions close to ``reference``.

    Distance to the reference is used as a tie-breaker so that consecutive
    waypoints stay on the same IK branch (important for smooth ServoJ motion).
    """
    rng = np.random.default_rng(seed)
    best: Optional[IKResult] = None
    best_cost = np.inf
    weights = np.ones(chain.dof) if joint_weights is None else np.asarray(joint_weights, dtype=float)
    ref_arr = None if reference is None else np.asarray(reference, dtype=float)

    for idx, s in enumerate(_random_seeds(chain, restarts, rng, reference)):
        result = solve_ik(chain, T_target, s, max_iter=max_iter)
        result.used_seed = idx
        if not result.success:
            continue
        result.q = _wrap_near(result.q, ref_arr, chain)
        if ref_arr is not None:
            cost = float(np.sum(weights * np.abs(result.q - ref_arr) ** 2))
        else:
            cost = float(np.sum(weights * result.q ** 2))
        # Prefer solutions away from joint limits, but never let this dominate
        # continuity of the joint-space path.
        if np.all(np.isfinite(chain.upper - chain.lower)):
            margin = np.minimum(result.q - chain.lower, chain.upper - result.q)
            cost += float(np.sum(1.0 / np.maximum(margin, 1e-3))) * 1e-6
        if cost < best_cost:
            best_cost = cost
            best = result

    if best is not None:
        return best

    # No success: return the closest attempt for diagnostics.
    fallback = solve_ik(chain, T_target, reference if reference is not None else np.zeros(chain.dof))
    fallback.success = False
    return fallback

def solve_ik_all_branches(
    chain: Chain,
    T_target: np.ndarray,
    restarts: int = 24,
    seed: int = 0,
    tol_pos: float = 1e-4,
    tol_rot: float = 1e-3,
) -> List[np.ndarray]:
    """Return every distinct joint solution for one pose.

    A 6-axis arm typically has up to 8 solutions for a given pose. Collecting
    them lets the caller keep one branch across a whole path instead of
    re-deciding per waypoint and hopping between elbow families.
    """
    rng = np.random.default_rng(seed)
    span = chain.upper - chain.lower
    span = np.where(np.isfinite(span) & (span > 1e-6), span, 2.0 * np.pi)

    found: List[np.ndarray] = []
    for _ in range(int(restarts)):
        s = chain.lower + rng.random(chain.dof) * span
        r = solve_ik(chain, T_target, s, tol_pos=tol_pos, tol_rot=tol_rot)
        if not r.success:
            continue
        q = (r.q + np.pi) % (2.0 * np.pi) - np.pi
        if all(float(np.max(np.abs(q - e))) > 0.05 for e in found):
            found.append(q)

    # Sort deterministically: the DP compares solutions across waypoints, and
    # an unstable ordering silently pairs the wrong branches.
    if found:
        found.sort(key=lambda q: tuple(np.round(q, 3)))
    return found
