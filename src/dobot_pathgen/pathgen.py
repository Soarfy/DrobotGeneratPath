"""Surface coverage path generation from a segmented dental mesh."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import se3


@dataclass
class SurfacePath:
    points: np.ndarray          # Nx3 surface points (model frame)
    normals: np.ndarray         # Nx3 unit normals (model frame)
    tangents: np.ndarray        # Nx3 path tangent (model frame)
    source_indices: np.ndarray  # index into the input vertex array


def select_region(vertices, normals, region: dict) -> np.ndarray:
    """Return a boolean mask for the requested region.

    ``region`` may contain one of ``mode`` or a list ``modes``; every sub-mode
    must agree (AND).
    """
    vertices = np.asarray(vertices, dtype=float)
    normals = np.asarray(normals, dtype=float)
    modes = region.get("modes") or [region]
    mask = np.ones(len(vertices), dtype=bool)

    for mode in modes:
        kind = mode.get("mode", "all")
        if kind == "all":
            continue
        if kind == "normal_cone":
            direction = np.asarray(mode.get("direction", [0, 0, 1]), dtype=float)
            direction = se3.normalize(direction)
            max_angle = np.deg2rad(float(mode.get("max_angle_deg", 60.0)))
            cosang = np.clip(normals @ direction, -1.0, 1.0)
            mask &= cosang >= np.cos(max_angle)
        elif kind == "sphere":
            center = np.asarray(mode.get("center_m", [0, 0, 0]), dtype=float)
            radius = float(mode.get("radius_m", 0.01))
            mask &= np.linalg.norm(vertices - center, axis=1) <= radius
        elif kind == "bbox":
            lo = np.asarray(mode["min_m"], dtype=float)
            hi = np.asarray(mode["max_m"], dtype=float)
            mask &= np.all((vertices >= lo) & (vertices <= hi), axis=1)
        elif kind == "face_index":
            indices = np.asarray(mode["indices"], dtype=int)
            m = np.zeros(len(vertices), dtype=bool)
            m[indices] = True
            mask &= m
        else:
            raise ValueError(f"unknown region mode: {kind}")
    return mask


def _plane_basis(points: np.ndarray, plane_normal) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = se3.normalize(plane_normal)
    # Project and use PCA to align the raster with the patch's long axis.
    proj = points - (points @ n)[:, None] * n
    centered = proj - proj.mean(axis=0)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigvecs columns are sorted by ascending eigenvalue
    long_axis = eigvecs[:, -1]
    long_axis = long_axis - n * float(long_axis @ n)
    if np.linalg.norm(long_axis) < 1e-8:
        long_axis = np.cross(n, [1.0, 0.0, 0.0])
        if np.linalg.norm(long_axis) < 1e-8:
            long_axis = np.cross(n, [0.0, 1.0, 0.0])
    long_axis = se3.normalize(long_axis)
    short_axis = se3.normalize(np.cross(n, long_axis))
    return long_axis, short_axis, n


def raster_surface_path(
    vertices,
    normals,
    mask,
    spacing_m: float = 0.0025,
    row_spacing_m: float = 0.003,
    plane_normal=None,
    max_distance_factor: float = 0.9,
) -> SurfacePath:
    """Serpentine raster over the selected surface patch.

    The algorithm projects the selected vertices onto a plane, walks a grid,
    and snaps every grid cell to the nearest vertex. It is intentionally simple:
    it produces an inspectable coverage path without needing a mesh parameterisation.
    """
    vertices = np.asarray(vertices, dtype=float)
    normals = np.asarray(normals, dtype=float)
    idx = np.flatnonzero(mask)
    if idx.size < 10:
        raise ValueError(f"region too small: {idx.size} vertices selected")

    pts = vertices[idx]
    nrms = normals[idx]
    if plane_normal is None:
        # Smallest-variance direction of the patch is a good local plane normal.
        centered = pts - pts.mean(axis=0)
        eigvals, eigvecs = np.linalg.eigh(centered.T @ centered)
        plane_normal = eigvecs[:, 0]
    n = se3.normalize(plane_normal)
    long_axis, short_axis, n = _plane_basis(pts, n)

    u = pts @ long_axis
    v = pts @ short_axis

    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())
    n_rows = max(1, int(np.ceil((v_max - v_min) / row_spacing_m)) + 1)
    n_cols = max(1, int(np.ceil((u_max - u_min) / spacing_m)) + 1)
    row_values = np.linspace(v_min, v_max, n_rows)
    col_values = np.linspace(u_min, u_max, n_cols)

    max_dist = max_distance_factor * float(np.hypot(spacing_m, row_spacing_m))
    chosen: List[int] = []
    for r, vv in enumerate(row_values):
        columns = col_values if r % 2 == 0 else col_values[::-1]
        for uu in columns:
            d2 = (u - uu) ** 2 + (v - vv) ** 2
            j = int(np.argmin(d2))
            if float(np.sqrt(d2[j])) <= max_dist:
                if not chosen or chosen[-1] != j:
                    chosen.append(j)

    if len(chosen) < 2:
        raise ValueError("raster produced fewer than 2 waypoints; increase spacing or region")

    chosen_arr = np.asarray(chosen, dtype=int)
    path_pts = pts[chosen_arr]
    path_nrm = nrms[chosen_arr]

    # Smooth positions and normals along the path (reduces normal noise from the scan).
    smooth_pts = path_pts.copy()
    smooth_nrm = path_nrm.copy()
    if len(path_pts) >= 3:
        for i in range(1, len(path_pts) - 1):
            smooth_pts[i] = (path_pts[i - 1] + 2.0 * path_pts[i] + path_pts[i + 1]) / 4.0
            smooth_nrm[i] = path_nrm[i - 1] + 2.0 * path_nrm[i] + path_nrm[i + 1]
    lens = np.linalg.norm(smooth_nrm, axis=1)
    valid = lens > 1e-9
    smooth_nrm[valid] /= lens[valid, None]
    smooth_nrm[~valid] = path_nrm[~valid]

    tangents = np.zeros_like(smooth_pts)
    tangents[1:-1] = smooth_pts[2:] - smooth_pts[:-2]
    tangents[0] = smooth_pts[1] - smooth_pts[0]
    tangents[-1] = smooth_pts[-1] - smooth_pts[-2]
    tlen = np.linalg.norm(tangents, axis=1)
    valid_t = tlen > 1e-12
    tangents[valid_t] /= tlen[valid_t, None]
    tangents[~valid_t] = long_axis

    return SurfacePath(
        points=smooth_pts,
        normals=smooth_nrm,
        tangents=tangents,
        source_indices=idx[chosen_arr],
    )


def tool_poses_from_surface(
    path: SurfacePath,
    standoff_m: float,
    roll_deg: float = 0.0,
    approach_clearance_m: float = 0.0,
    x_reference=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build 4x4 tool poses (model frame) with tool +Z pointing into the surface.

    ``x_reference`` picks the tool's X axis: when given, X is the projection of
    that world direction onto the tangent plane. This keeps the wrist roll
    stable along a serpentine path instead of following noisy tangents.
    Returns (poses Nx4x4, tip_points Nx3).
    """
    n = len(path.points)
    poses = np.zeros((n, 4, 4))
    tips = np.zeros((n, 3))
    roll = np.deg2rad(float(roll_deg))
    ref = None if x_reference is None else se3.normalize(np.asarray(x_reference, dtype=float))

    for i in range(n):
        surface_p = path.points[i]
        normal = se3.normalize(path.normals[i])
        z_axis = -normal  # tool approaches along -normal

        if ref is not None:
            x_axis = ref - z_axis * float(ref @ z_axis)
        else:
            tangent = path.tangents[i]
            x_axis = tangent - z_axis * float(tangent @ z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            fallback = np.array([1.0, 0.0, 0.0])
            if abs(float(fallback @ z_axis)) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0])
            x_axis = fallback - z_axis * float(fallback @ z_axis)
        x_axis = se3.normalize(x_axis)
        y_axis = se3.normalize(np.cross(z_axis, x_axis))
        x_axis = se3.normalize(np.cross(y_axis, z_axis))
        R = np.column_stack([x_axis, y_axis, z_axis])
        if abs(roll) > 1e-12:
            R = R @ se3.rotation_about_axis([0, 0, 1], roll)

        tip = surface_p + normal * standoff_m
        poses[i] = se3.make_transform(R, tip)
        tips[i] = tip

    if approach_clearance_m > 0.0 and n > 0:
        pre = poses[0].copy()
        pre[:3, 3] = tips[0] + path.normals[0] * approach_clearance_m
        post = poses[-1].copy()
        post[:3, 3] = tips[-1] + path.normals[-1] * approach_clearance_m
        poses = np.concatenate([[pre], poses, [post]], axis=0)
        tips = np.concatenate([[pre[:3, 3]], tips, [post[:3, 3]]], axis=0)

    return poses, tips


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _smooth_path(points, normals, fallback):
    """Light smoothing + tangents, shared by raster and arch paths."""
    pts = np.asarray(points, dtype=float)
    nrms = np.asarray(normals, dtype=float).copy()
    if len(pts) >= 3:
        sp = pts.copy()
        for i in range(1, len(pts) - 1):
            sp[i] = (pts[i - 1] + 2.0 * pts[i] + pts[i + 1]) / 4.0
            nrms[i] = normals[i - 1] + 2.0 * normals[i] + normals[i + 1]
        pts = sp
    lens = np.linalg.norm(nrms, axis=1)
    valid = lens > 1e-9
    nrms[valid] /= lens[valid, None]
    nrms[~valid] = np.asarray(normals, dtype=float)[~valid]

    tangents = np.zeros_like(pts)
    if len(pts) >= 2:
        tangents[1:-1] = pts[2:] - pts[:-2]
        tangents[0] = pts[1] - pts[0]
        tangents[-1] = pts[-1] - pts[-2]
    tlen = np.linalg.norm(tangents, axis=1)
    valid_t = tlen > 1e-12
    tangents[valid_t] /= tlen[valid_t, None]
    tangents[~valid_t] = fallback
    return pts, nrms, tangents


def limit_tilt(normals, reference_normal, max_tilt_deg: float):
    """Clamp each normal's deviation from ``reference_normal``.

    Following every local surface normal around a full arch swings the wrist
    through a very large arc, which pushes the IK across branch switches
    mid-path. Real brushing keeps the tool near a nominal tilt and lets the
    bristles absorb the difference, so clamp the tilt instead.
    """
    nrms = np.asarray(normals, dtype=float).copy()
    ref = se3.normalize(reference_normal)
    cos_max = np.cos(np.deg2rad(float(max_tilt_deg)))
    for i in range(len(nrms)):
        n = se3.normalize(nrms[i])
        c = float(np.clip(n @ ref, -1.0, 1.0))
        if c >= cos_max:
            continue
        denom = c - 1.0
        if abs(denom) < 1e-12:
            nrms[i] = ref
            continue
        t = float(np.clip((c - cos_max) / denom, 0.0, 1.0))
        nrms[i] = se3.normalize(n + t * (ref - n))
    return nrms


def arch_surface_path(
    vertices,
    normals,
    spacing_m: float = 0.004,
    up_direction=(0.0, 1.0, 0.0),
    across_direction=None,
    poly_degree: int = 4,
    u_range=None,
    lateral_offset_m: float = 0.0,
    band_radius_m: float = 0.004,
    bin_width_m: float = 0.0015,
    angular_bins: int = 180,
    height_band_m: float = 0.012,
    min_face_dot: float = 0.4,
    smoothing_residual_m: float = 0.0007,
    trim_fraction: float = 0.0,
) -> SurfacePath:
    """Walk along the dental arch (mesiodistal direction), molar to molar.

    A dental arch is a horseshoe, so binning along one Cartesian axis samples
    the front teeth twice and misses the molars. This sweeps by angle around
    the arch centre instead:

    1. Estimate the occlusal plane from the outward-facing normals.
    2. Keep only vertices near the occlusal height band and facing outward,
       which removes the model base and the side walls.
    3. In each angular bin take the highest remaining vertex -> crest point.
    4. Open the loop at the largest angular gap, smooth the crest with an
       arc-length spline (cusps make raw picks zig-zag), then resample.
    5. Snap every sample onto the nearest surface vertex and take its normal.
    """
    vertices = np.asarray(vertices, dtype=float)
    normals = np.asarray(normals, dtype=float)
    up = se3.normalize(up_direction)

    facing = normals @ up
    if np.mean(facing > 0.25) < 0.05:
        raise ValueError("up_direction points into the model; flip it")

    plane_normal = se3.normalize(normals[facing > 0.3].mean(axis=0))
    if float(plane_normal @ up) < 0:
        plane_normal = -plane_normal

    if across_direction is None:
        across = np.array([1.0, 0.0, 0.0])
        if abs(float(across @ plane_normal)) > 0.9:
            across = np.array([0.0, 0.0, 1.0])
    else:
        across = np.asarray(across_direction, dtype=float)
    across = across - plane_normal * float(across @ plane_normal)
    across = se3.normalize(across)
    side = se3.normalize(np.cross(plane_normal, across))

    height = vertices @ plane_normal
    centre = vertices.mean(axis=0)
    du = vertices @ across - float(centre @ across)
    dv = vertices @ side - float(centre @ side)
    angle = np.arctan2(dv, du) % (2.0 * np.pi)

    outward = facing > 0.25
    if outward.sum() < 100:
        raise ValueError("too few outward-facing vertices; check up_direction")
    height_mid = float(np.median(height[outward]))
    band = (
        outward
        & (np.abs(height - height_mid) < float(height_band_m))
        & (normals @ plane_normal > float(min_face_dot))
    )
    if band.sum() < 200:
        # Relax the band rather than failing on an unusual scan.
        band = outward & (np.abs(height - height_mid) < 2.0 * float(height_band_m))

    bins = np.clip(
        (angle / (2.0 * np.pi) * angular_bins).astype(int), 0, angular_bins - 1
    )
    crest_ids = []
    crest_angles = []
    for b in range(angular_bins):
        mask = (bins == b) & band
        if mask.sum() < 5:
            continue
        ids = np.flatnonzero(mask)
        best = int(ids[np.argmax(height[ids])])
        crest_ids.append(best)
        crest_angles.append(float(angle[best]))

    if len(crest_ids) < 12:
        raise ValueError("could not extract an arch ridge; check up_direction")

    crest_ids = np.asarray(crest_ids, dtype=int)
    crest_angles = np.asarray(crest_angles, dtype=float)
    order = np.argsort(crest_angles)
    crest_ids = crest_ids[order]
    crest_angles = crest_angles[order]

    # Open the horseshoe at the largest angular gap (back of the mouth).
    gaps = np.diff(np.concatenate([crest_angles, crest_angles[:1] + 2.0 * np.pi]))
    start = int(np.argmax(gaps)) + 1
    crest_ids = crest_ids[np.roll(np.arange(len(crest_ids)), -start)]

    if u_range is not None:
        u_min, u_max = float(u_range[0]), float(u_range[1])
        u_coords = vertices[crest_ids] @ across
        keep = (u_coords >= u_min) & (u_coords <= u_max)
        if keep.sum() >= 8:
            crest_ids = crest_ids[keep]
            ca = np.sort(crest_angles)[: len(crest_ids)] if False else None
            # Re-open the loop on the trimmed set.
            sub_angles = angle[crest_ids]
            g2 = np.diff(np.concatenate([sub_angles, sub_angles[:1] + 2 * np.pi]))
            start2 = int(np.argmax(g2)) + 1
            crest_ids = crest_ids[np.roll(np.arange(len(crest_ids)), -start2)]

    crest_points = vertices[crest_ids]

    # Arc-length spline: removes the cusp-to-cusp zig-zag while preserving the
    # horseshoe (a low-order polynomial in angle would cut the corners).
    seg = np.linalg.norm(np.diff(crest_points, axis=0), axis=1)
    s_raw = np.concatenate([[0.0], np.cumsum(seg)])
    if float(s_raw[-1]) < spacing_m:
        raise ValueError("arch span shorter than one path step")

    s_dense = np.linspace(0.0, float(s_raw[-1]), 3000)
    try:
        from scipy.interpolate import UnivariateSpline

        lam = (float(smoothing_residual_m) ** 2) * len(crest_points)
        curve = np.column_stack(
            [UnivariateSpline(s_raw, crest_points[:, d], s=lam)(s_dense) for d in range(3)]
        )
    except ImportError:  # pragma: no cover - scipy is optional
        window = 9
        kernel = np.ones(window) / window
        padded = np.pad(crest_points, ((window // 2, window // 2), (0, 0)), mode="edge")
        smooth = np.column_stack(
            [np.convolve(padded[:, d], kernel, mode="valid") for d in range(3)]
        )
        s_smooth = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(smooth, axis=0), axis=1))]
        )
        curve = np.column_stack(
            [np.interp(s_dense, s_smooth, smooth[:, d]) for d in range(3)]
        )

    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])

    if trim_fraction > 0:
        s_lo, s_hi = total * float(trim_fraction), total * (1.0 - float(trim_fraction))
        if s_hi - s_lo > 6 * spacing_m:
            keep = (s >= s_lo) & (s <= s_hi)
            s = s[keep] - s_lo
            curve = curve[keep]
            total = float(s[-1])

    n_samples = max(2, int(np.floor(total / spacing_m)) + 1)
    s_grid = np.linspace(0.0, total, n_samples)
    samples = np.column_stack([np.interp(s_grid, s, curve[:, d]) for d in range(3)])

    # Snap to the nearest surface vertex so the tip really touches the scan.
    try:
        from scipy.spatial import cKDTree

        _, nearest = cKDTree(vertices).query(samples, k=1)
        nearest = np.atleast_1d(nearest)
        sampled = vertices[nearest].copy()
        normal_idx = nearest.astype(int)
    except ImportError:  # pragma: no cover
        sampled = samples.copy()
        normal_idx = np.zeros(len(samples), dtype=int)
        for i, p in enumerate(samples):
            normal_idx[i] = int(np.argmin(np.sum((vertices - p) ** 2, axis=1)))
            sampled[i] = vertices[normal_idx[i]]

    if abs(lateral_offset_m) > 1e-12:
        tangents = np.gradient(sampled, axis=0)
        in_plane_t = tangents - np.outer(tangents @ plane_normal, plane_normal)
        lateral = np.cross(plane_normal, in_plane_t)
        norms = np.linalg.norm(lateral, axis=1)
        norms = np.where(norms > 1e-12, norms, 1.0)
        sampled = sampled + (lateral / norms[:, None]) * lateral_offset_m

    snapped_normals = normals[normal_idx].copy()
    fallback_tangent = se3.normalize(sampled[-1] - sampled[0])
    pts, nrms, tangents = _smooth_path(sampled, snapped_normals, fallback_tangent)
    return SurfacePath(
        points=pts,
        normals=nrms,
        tangents=tangents,
        source_indices=normal_idx,
    )


def _slerp(q0, q1, t):
    """Spherical linear interpolation between two xyzw quaternions."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    dot = float(np.clip(q0 @ q1, -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (np.sin((1.0 - t) * theta) / sin_theta) * q0 + (np.sin(t * theta) / sin_theta) * q1


def normal_approach_tool_poses(
    path: SurfacePath,
    standoff_m: float,
    approach_clearance_m: float = 0.0,
):
    """Tool poses whose approach follows the surface normal (tip poses).

    Used when the brush should track the tooth surface rather than hold a fixed
    orientation. Returns model-frame tip poses; the caller converts to flange
    poses with the TCP transform.
    """
    n = len(path.points)
    poses = np.zeros((n, 4, 4))
    tips = np.zeros((n, 3))
    for i in range(n):
        nrm = se3.normalize(path.normals[i])
        z = -nrm                       # tool +Z points into the surface
        x = path.tangents[i] - z * float(path.tangents[i] @ z)
        if np.linalg.norm(x) < 1e-8:
            x = np.array([1.0, 0.0, 0.0])
        x = se3.normalize(x)
        y = se3.normalize(np.cross(z, x))
        x = se3.normalize(np.cross(y, z))
        tip = path.points[i] + nrm * standoff_m
        poses[i] = se3.make_transform(np.column_stack([x, y, z]), tip)
        tips[i] = tip
    if approach_clearance_m > 0.0 and n > 0:
        pre = poses[0].copy(); pre[:3, 3] = tips[0] + path.normals[0] * approach_clearance_m
        post = poses[-1].copy(); post[:3, 3] = tips[-1] + path.normals[-1] * approach_clearance_m
        poses = np.concatenate([[pre], poses, [post]], axis=0)
        tips = np.concatenate([[pre[:3, 3]], tips, [post[:3, 3]]], axis=0)
    return poses, tips


def fixed_axis_tool_poses(
    path: SurfacePath,
    standoff_m: float,
    rod_direction_base=(0.0, 0.0, -1.0),
    rod_in_link6=None,
    approach_clearance_m: float = 0.0,
    model_to_base=None,
):
    """Poses that hold the brush axis at a fixed direction in the BASE frame.

    The TCP has no rotation relative to the flange (BrushTooth.cpp sets
    ``{x,y,z,0,0,0}``), so the rod's direction in Link6 is a constant. To make
    that constant direction point along ``rod_direction_base``, the flange
    rotation must be ``R_base * R_rod_link6^-1``. Solving that directly keeps
    the tool frame honest instead of rotating the TCP definition, which would
    desynchronise the home posture from the real cell.

    Returns poses in the MODEL frame, ready for ``compile_trajectory``.
    """
    n = len(path.points)
    poses = np.zeros((n, 4, 4))
    tips = np.zeros((n, 3))
    R_mb = np.eye(3) if model_to_base is None else np.asarray(model_to_base)[:3, :3]

    z_base = se3.normalize(np.asarray(rod_direction_base, dtype=float))
    rod_l6 = se3.normalize(
        np.asarray(rod_in_link6 if rod_in_link6 is not None else [0.0, 0.0, 1.0], dtype=float)
    )

    # Build a base-frame rotation whose +Z is the rod, then rotate it so that
    # the flange's own rod axis lands there.
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(z_base @ ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    x_b = se3.normalize(np.cross(ref, z_base))
    y_b = se3.normalize(np.cross(z_base, x_b))
    R_want = np.column_stack([x_b, y_b, z_base])

    # R_flange such that R_flange @ rod_l6 == z_base.
    ref2 = np.array([0.0, 0.0, 1.0])
    if abs(float(rod_l6 @ ref2)) > 0.9:
        ref2 = np.array([1.0, 0.0, 0.0])
    x_l = se3.normalize(np.cross(ref2, rod_l6))
    y_l = se3.normalize(np.cross(rod_l6, x_l))
    R_rod = np.column_stack([x_l, y_l, rod_l6])

    R_flange = R_want @ R_rod.T
    # FLANGE pose expressed in the model frame. compile_trajectory multiplies by
    # model_to_base and, when tip_poses is False, applies one TCP inverse; both
    # are accounted for here, so callers must pass tip_poses=False.
    R_model = R_mb.T @ R_flange

    for i in range(n):
        normal = se3.normalize(path.normals[i])
        tip = path.points[i] + normal * standoff_m
        poses[i] = se3.make_transform(R_model, tip)
        tips[i] = tip

    if approach_clearance_m > 0.0 and n > 0:
        back = -R_mb.T @ z_base * approach_clearance_m
        pre = poses[0].copy()
        pre[:3, 3] = tips[0] + back
        post = poses[-1].copy()
        post[:3, 3] = tips[-1] + back
        poses = np.concatenate([[pre], poses, [post]], axis=0)
        tips = np.concatenate([[pre[:3, 3]], tips, [post[:3, 3]]], axis=0)

    return poses, tips


def fixed_orientation_tool_poses(
    path: SurfacePath,
    standoff_m: float,
    rotation_base_deg=(180.0, 0.0, 150.0),
    approach_clearance_m: float = 0.0,
    model_to_base=None,
):
    """TIP poses (position at the brush tip) with a constant flange rotation.

    Position is the tip; rotation is the flange rotation, so the caller must
    still apply the TCP transform to obtain the flange pose.

    Following the surface normal around a complete arch rotates the tool by
    ~176 deg, which no 6-axis arm can do without flips. Holding one flange
    orientation keeps the wrist still; the brush simply meets the tooth at
    whatever angle that orientation produces.

    ``rotation_base_deg`` is the flange rotation in the BASE frame.
    """
    n = len(path.points)
    poses = np.zeros((n, 4, 4))
    tips = np.zeros((n, 3))
    R_mb = np.eye(3) if model_to_base is None else np.asarray(model_to_base)[:3, :3]

    R_base = se3.rpy_deg_to_matrix(rotation_base_deg)
    R_model = R_mb.T @ R_base

    for i in range(n):
        nrm = se3.normalize(path.normals[i])
        tip = path.points[i] + nrm * standoff_m
        poses[i] = se3.make_transform(R_model, tip)
        tips[i] = tip

    if approach_clearance_m > 0.0 and n > 0:
        pre = poses[0].copy(); pre[:3, 3] = tips[0] + path.normals[0] * approach_clearance_m
        post = poses[-1].copy(); post[:3, 3] = tips[-1] + path.normals[-1] * approach_clearance_m
        poses = np.concatenate([[pre], poses, [post]], axis=0)
        tips = np.concatenate([[pre[:3, 3]], tips, [post[:3, 3]]], axis=0)
    return poses, tips


def densify_positions(poses, max_translation_step_m: float = 0.002):
    """Insert poses along the path while keeping orientation constant."""
    poses = np.asarray(poses, dtype=float)
    if len(poses) < 2:
        return poses
    out = [poses[0]]
    for i in range(1, len(poses)):
        T0, T1 = poses[i - 1], poses[i]
        d = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
        n = max(1, int(np.ceil(d / max(max_translation_step_m, 1e-6))))
        for k in range(1, n):
            t_ = k / n
            T = T0.copy()
            T[:3, 3] = T0[:3, 3] + t_ * (T1[:3, 3] - T0[:3, 3])
            out.append(T)
        out.append(T1)
    return np.array(out, dtype=float)


def densify_poses(
    poses: np.ndarray,
    max_translation_step_m: float = 0.002,
    max_rotation_step_deg: float = 3.0,
) -> np.ndarray:
    """Insert interpolated poses so each Cartesian step stays small.

    Raster paths can jump across the arch between rows; feeding those jumps
    straight into IK produces large joint steps that ServoJ streams badly.
    """
    poses = np.asarray(poses, dtype=float)
    if len(poses) < 2:
        return poses
    max_rot = np.deg2rad(max_rotation_step_deg)
    out = [poses[0]]
    for i in range(1, len(poses)):
        T0, T1 = poses[i - 1], poses[i]
        d = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
        R_rel = T1[:3, :3] @ T0[:3, :3].T
        ang = float(np.linalg.norm(se3.rotvec_from_matrix(R_rel)))
        n = int(np.ceil(max(d / max(max_translation_step_m, 1e-6),
                           ang / max(max_rot, 1e-6))))
        n = max(1, n)
        q0 = se3.quat_xyzw_from_matrix(T0[:3, :3])
        q1 = se3.quat_xyzw_from_matrix(T1[:3, :3])
        for k in range(1, n):
            t = k / n
            T = np.eye(4)
            T[:3, :3] = se3.matrix_from_quat_xyzw(_slerp(q0, q1, t))
            T[:3, 3] = T0[:3, 3] + t * (T1[:3, 3] - T0[:3, 3])
            out.append(T)
        out.append(T1)
    return np.array(out, dtype=float)
