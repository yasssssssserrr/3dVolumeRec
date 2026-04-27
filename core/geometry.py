"""
core/geometry.py
================
Calibration-aware coordinate geometry for the PNN reconstruction.

Transform chain
---------------
  Given:
    • T_calib = T_ImageToProbe  (4×4, from calibration file)
    • T_frame  = T_ProbeToTracker (4×4, per-frame from MHD header)
    • ps       = pixel_spacing_mm (scalar, ~0.0885 mm)

  For pixel (u, v):
    p_image_mm  = [u·ps,  v·ps,  0,  1]ᵀ
    p_probe     = T_calib @ p_image_mm
    p_tracker   = T_frame  @ p_probe

  Pre-multiplied effective transform (computed once per frame):
    T_eff = T_frame @ T_calib
    p_tracker = T_eff @ p_image_mm

  When calibration is disabled (Identity):
    T_eff = T_frame
    p_tracker = T_frame @ [u·ps, v·ps, 0, 1]ᵀ

Volume grid (ZYX)
-----------------
  Internal arrays use shape (NZ, NY, NX) with axis order matching:
    NZ = dimension along tracker Z axis
    NY = dimension along tracker Y axis
    NX = dimension along tracker X axis
  This corresponds to standard volumetric depth-row-column ordering.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path as _Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from us3d_io.mhd_reader import SequenceData

logger = logging.getLogger(__name__)


# ── Calibration orthonormalisation ────────────────────────────────────────────

def orthonormalize_rotation(T: np.ndarray) -> np.ndarray:
    """
    Project the rotation submatrix of a 4x4 homogeneous transform onto SO(3)
    via SVD, correcting the ~1.7% scale error in the averaged calibration
    (det(R)=0.9826 → 1.0000 after orthonormalisation).

    Parameters
    ----------
    T : (4, 4) float64 – input transform (may have imperfect R)

    Returns
    -------
    T_out : (4, 4) float64 – same translation, orthonormal rotation
    """
    T_out = T.copy()
    U, _, Vt = np.linalg.svd(T[:3, :3])
    R_fixed = U @ Vt
    # Ensure proper rotation (det = +1, not reflection)
    if np.linalg.det(R_fixed) < 0:
        U[:, -1] *= -1
        R_fixed = U @ Vt
    T_out[:3, :3] = R_fixed
    logger.info(
        "Orthonormalised calibration R: det %.6f -> %.6f",
        np.linalg.det(T[:3, :3]),
        np.linalg.det(R_fixed),
    )
    return T_out


# ── Pre-compute effective per-frame transform ─────────────────────────────────

def effective_transform(
    probe_to_tracker: np.ndarray,
    image_to_probe: Optional[np.ndarray],
) -> np.ndarray:
    """
    Return the single matrix that maps image-mm coordinates directly to tracker.

    Parameters
    ----------
    probe_to_tracker : (4,4) frame-level ProbeToTracker matrix
    image_to_probe   : (4,4) calibration matrix; None means Identity

    Returns
    -------
    T_eff : (4,4) float64 – p_tracker = T_eff @ [u·ps, v·ps, 0, 1]ᵀ
    """
    if image_to_probe is None:
        return probe_to_tracker
    return probe_to_tracker @ image_to_probe


# ── Pixel-grid factory ────────────────────────────────────────────────────────

def make_image_points(
    height: int,
    width: int,
    pixel_spacing: float,
    downsample: int = 1,
) -> np.ndarray:
    """
    Build homogeneous image-plane mm coordinates for all sampled pixels.
    Both rows and columns are sub-sampled by `downsample` (2-D sub-sampling).

    Parameters
    ----------
    height, width  : image dimensions (pixels)
    pixel_spacing  : mm per pixel (isotropic)
    downsample     : keep every N-th row AND column

    Returns
    -------
    pts : (4, ⌈H/ds⌉ · ⌈W/ds⌉) float64 – [u·ps, v·ps, 0, 1] per pixel
    """
    rows = np.arange(0, height, downsample, dtype=np.float64)
    cols = np.arange(0, width,  downsample, dtype=np.float64)
    V, U = np.meshgrid(rows, cols, indexing="ij")
    N = V.size
    pts = np.empty((4, N), dtype=np.float64)
    pts[0] = U.ravel() * pixel_spacing     # x = lateral (mm)
    pts[1] = V.ravel() * pixel_spacing     # y = depth   (mm)
    pts[2] = 0.0                           # z = 0       (in-plane)
    pts[3] = 1.0                           # homogeneous
    return pts


def transform_to_tracker(
    pts_image_mm: np.ndarray,
    T_eff: np.ndarray,
) -> np.ndarray:
    """
    Apply an effective transform to (4,N) image-mm points.

    Returns
    -------
    (3, N) float64 – tracker-space XYZ coordinates
    """
    return (T_eff @ pts_image_mm)[:3]


# ── Bounding-box computation ───────────────────────────────────────────────────

def compute_volume_bounds(
    sequences: List[SequenceData],
    image_to_probe: Optional[np.ndarray],
    downsample: int = 8,
    margin_mm: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the tracker-space AABB enclosing all pixel positions across all frames,
    optionally expanded by `margin_mm` on all sides.

    Parameters
    ----------
    sequences      : loaded SequenceData objects
    image_to_probe : calibration matrix (None = Identity)
    downsample     : sub-sampling for speed (8 ≈ 1% of pixels)
    margin_mm      : extra padding added to each face of the bounding box

    Returns
    -------
    bb_min : (3,) float64 – minimum [X, Y, Z] in tracker mm
    bb_max : (3,) float64 – maximum [X, Y, Z] in tracker mm
    """
    gmin = np.full(3,  np.inf, dtype=np.float64)
    gmax = np.full(3, -np.inf, dtype=np.float64)

    for seq in sequences:
        h, w = seq.image_height, seq.image_width
        for frame in seq.frames:
            ps    = frame.pixel_spacing_mm
            T_eff = effective_transform(frame.probe_to_tracker, image_to_probe)
            pts   = make_image_points(h, w, ps, downsample=downsample)
            pts_t = transform_to_tracker(pts, T_eff)
            gmin  = np.minimum(gmin, pts_t.min(axis=1))
            gmax  = np.maximum(gmax, pts_t.max(axis=1))

    if margin_mm > 0:
        gmin -= margin_mm
        gmax += margin_mm

    logger.info(
        "Bounding box (mm)  min %s   max %s   extent %s  (margin=%.1f mm)",
        gmin.round(1), gmax.round(1), (gmax - gmin).round(1), margin_mm,
    )
    return gmin, gmax


# ── Voxel grid ────────────────────────────────────────────────────────────────

def build_voxel_grid(
    bb_min: np.ndarray,
    bb_max: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Compute voxel-grid origin and shape from a tracker-space bounding box.

    Returns
    -------
    origin : (3,) float64 – tracker position of voxel (0,0,0)
    shape  : (NZ, NY, NX) int  – volume dimensions
    """
    extent    = bb_max - bb_min
    shape_xyz = np.ceil(extent / voxel_size).astype(int) + 1
    nx, ny, nz = shape_xyz
    logger.info(
        "Voxel grid: %d×%d×%d  (%.1f × %.1f × %.1f mm,  vox=%.2f mm)",
        nz, ny, nx,
        nz * voxel_size, ny * voxel_size, nx * voxel_size,
        voxel_size,
    )
    return bb_min.copy(), (nz, ny, nx)


def tracker_to_voxel(
    pts_tracker: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """
    Map (3, N) tracker-mm coordinates to (3, N) integer voxel indices [ix, iy, iz].
    """
    return np.round((pts_tracker - origin[:, None]) / voxel_size).astype(np.int32)
