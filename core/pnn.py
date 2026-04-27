"""
core/pnn.py – Calibration-aware PNN reconstructor v3
Improvements over v2:
  • Trilinear scatter (IDW) instead of nearest-voxel → eliminates stripe artifacts
  • np.bincount replaces np.add.at → 5-10x faster scatter
  • Gaussian weights in physical mm (was sub-voxel no-op)
  • 2-D pixel downsampling (rows AND cols)
  • Hole-fill bias removed (no more -0.03 clip)
  • Bounding-box margin support
"""
from __future__ import annotations
import logging, time, sys
from pathlib import Path as _Path
from typing import List, Optional, Tuple
import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter, gaussian_filter

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from config import PipelineConfig
from core.geometry import (
    build_voxel_grid, compute_volume_bounds, effective_transform,
    make_image_points, tracker_to_voxel, transform_to_tracker,
)
from us3d_io.mhd_reader import SequenceData

logger = logging.getLogger(__name__)


class PNNReconstructor:
    def __init__(self, config: PipelineConfig,
                 image_to_probe: Optional[np.ndarray] = None):
        self.cfg     = config
        self.T_calib = image_to_probe
        self._volume = self._mask = self._origin = self._shape = None

    @property
    def volume(self):
        if self._volume is None: raise RuntimeError("Call reconstruct() first.")
        return self._volume
    @property
    def mask(self):
        if self._mask is None: raise RuntimeError("Call reconstruct() first.")
        return self._mask
    @property
    def origin(self):     return self._origin
    @property
    def voxel_size(self): return self.cfg.voxel_size_mm
    @property
    def shape(self):      return self._shape

    def reconstruct(self, sequences: List[SequenceData]) -> np.ndarray:
        t0, cfg = time.perf_counter(), self.cfg

        # ── Step 1: Bounding box ──────────────────────────────────────────────
        logger.info("== Step 1/4  Bounding-box ==")
        bb_min, bb_max = compute_volume_bounds(
            sequences, self.T_calib, downsample=8,
            margin_mm=cfg.bb_margin_mm,
        )
        self._origin, self._shape = build_voxel_grid(bb_min, bb_max, cfg.voxel_size_mm)
        nz, ny, nx = self._shape
        logger.info("Voxels %d×%d×%d = %s  (%.0f MB f32)",
                    nz, ny, nx, f"{nz*ny*nx:,}", nz*ny*nx*4/1e6)

        # ── Step 2: Bin-fill (trilinear or nearest-voxel) ─────────────────────
        mode = "trilinear" if cfg.use_trilinear else "nearest-voxel"
        logger.info("== Step 2/4  PNN bin filling (%s) ==", mode)
        sz = nz * ny * nx
        vox_sum   = np.zeros(sz, dtype=np.float64)
        vox_count = np.zeros(sz, dtype=np.float64)
        self._bin_fill(sequences, vox_sum, vox_count, nz, ny, nx)

        vox_sum   = vox_sum.reshape(nz, ny, nx)
        vox_count = vox_count.reshape(nz, ny, nx)
        filled_mask = vox_count > 0
        mean_vol    = np.zeros((nz, ny, nx), dtype=np.float32)
        mean_vol[filled_mask] = (
            vox_sum[filled_mask] / vox_count[filled_mask]
        ).astype(np.float32)
        del vox_sum, vox_count

        # ── Step 3: Hole-fill ─────────────────────────────────────────────────
        logger.info("== Step 3/4  Hole filling ==")
        volume = self._hole_fill(mean_vol, filled_mask, cfg.hole_fill_radius)

        # ── Step 4: Gaussian smoothing ────────────────────────────────────────
        if cfg.gaussian_sigma > 0:
            logger.info("== Step 4/4  Gaussian smoothing (sigma=%.1f vox) ==",
                        cfg.gaussian_sigma)
            volume = self._smooth(volume, filled_mask, cfg.gaussian_sigma)
        else:
            logger.info("== Step 4/4  Gaussian smoothing skipped ==")

        self._volume = volume
        self._mask   = filled_mask

        elapsed  = time.perf_counter() - t0
        pct_fill = 100.0 * filled_mask.sum() / filled_mask.size
        logger.info("Done %.1f s  |  %.1f%% filled", elapsed, pct_fill)
        return volume

    # ── Bin-fill dispatcher ───────────────────────────────────────────────────

    def _bin_fill(self, sequences, vox_sum, vox_count, nz, ny, nx):
        cfg    = self.cfg
        origin = self._origin
        vs     = cfg.voxel_size_mm
        ds     = cfg.downsample_factor
        sz     = nz * ny * nx
        total  = sum(s.n_frames for s in sequences)
        done   = 0
        pts_cache: dict = {}

        # Physical Gaussian sigma converted to voxel units for the weight kernel
        sigma_vox = cfg.gaussian_sigma_mm / vs

        for seq in sequences:
            h, w = seq.image_height, seq.image_width
            for frame in seq.frames:
                ps    = frame.pixel_spacing_mm
                T_eff = effective_transform(frame.probe_to_tracker, self.T_calib)

                key = (h, w, ps, ds)
                if key not in pts_cache:
                    pts_cache[key] = make_image_points(h, w, ps, downsample=ds)
                pts_img = pts_cache[key]

                pts_t = transform_to_tracker(pts_img, T_eff)   # (3, N)
                exact = (pts_t - origin[:, None]) / vs          # float voxel coords

                # Pixel intensities (2-D downsampled: both rows AND cols)
                pix_flat = frame.image[::ds, ::ds].ravel().astype(np.float64)

                if cfg.use_trilinear:
                    _scatter_trilinear(
                        exact, pix_flat, vox_sum, vox_count,
                        nx, ny, nz, sz, sigma_vox,
                    )
                else:
                    _scatter_nearest(
                        exact, pix_flat, vox_sum, vox_count,
                        nx, ny, nz, sz, sigma_vox,
                    )

                done += 1
                if done % 100 == 0 or done == total:
                    logger.info("  Bin-fill: %d/%d (%.0f%%)",
                                done, total, 100 * done / total)

    # ── Hole-fill ─────────────────────────────────────────────────────────────

    @staticmethod
    def _hole_fill(volume, filled_mask, radius):
        vol, mask = volume.copy(), filled_mask.copy()
        struct   = np.ones((3, 3, 3), dtype=bool)
        holes0   = int((~mask).sum())
        logger.info("  Holes before: %d", holes0)
        for it in range(radius):
            dil = binary_dilation(mask, structure=struct)
            new = dil & ~mask
            if not new.any():
                logger.info("  Converged at iteration %d", it + 1)
                break
            fm = mask.astype(np.float32)
            si = uniform_filter(vol * fm, size=3, mode="constant") * 27.0
            sc = uniform_filter(fm,       size=3, mode="constant") * 27.0
            with np.errstate(invalid="ignore", divide="ignore"):
                mn = np.where(sc > 0, si / sc, 0.0).astype(np.float32)
            vol[new]  = mn[new]
            mask[new] = True
            closed = new.sum()
            if (it + 1) % 2 == 0:
                logger.info("  Pass %d: closed %d holes", it + 1, closed)
        holes1 = int((~mask).sum())
        logger.info("  Holes after (%d passes): %d  (%.1f%% removed)",
                    radius, holes1, 100 * (holes0 - holes1) / max(holes0, 1))
        return np.clip(vol, 0.0, 255.0).astype(np.float32)

    # ── Gaussian smoothing (mask-preserving) ──────────────────────────────────

    @staticmethod
    def _smooth(volume, filled_mask, sigma):
        """Gaussian smooth only within the filled region, preserving edges."""
        sm = gaussian_filter(volume * filled_mask.astype(np.float32), sigma=sigma)
        wt = gaussian_filter(filled_mask.astype(np.float32),           sigma=sigma)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(wt > 1e-6, sm / wt, volume).astype(np.float32)
        out[~filled_mask] = 0.0
        return np.clip(out, 0.0, 255.0)


# ── Scatter kernels (module-level for easy profiling / reuse) ─────────────────

def _scatter_nearest(exact, pix_flat, vox_sum, vox_count,
                     nx, ny, nz, sz, sigma_vox):
    """Nearest-voxel bincount scatter with physical Gaussian weights."""
    idx = np.round(exact).astype(np.int32)
    ix, iy, iz = idx[0], idx[1], idx[2]
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
    if not valid.any():
        return
    ix, iy, iz = ix[valid], iy[valid], iz[valid]
    # Gaussian weight by sub-voxel distance in physical mm (sigma_vox in voxels)
    dist2   = ((exact[:, valid] - np.round(exact[:, valid])) ** 2).sum(axis=0)
    weights = np.exp(-0.5 * dist2 / (sigma_vox ** 2))
    flat    = iz.astype(np.int64) * ny * nx + iy * nx + ix
    pv      = pix_flat[valid]
    vox_sum   += np.bincount(flat, weights=pv * weights, minlength=sz)
    vox_count += np.bincount(flat, weights=weights,      minlength=sz)


def _scatter_trilinear(exact, pix_flat, vox_sum, vox_count,
                       nx, ny, nz, sz, sigma_vox):
    """
    Trilinear scatter with a single np.bincount call per frame.

    Each pixel distributes its value to the 8 surrounding voxels with
    trilinear weights, further modulated by a Gaussian based on the
    sub-voxel distance (in physical voxel units).
    """
    fx, fy, fz = exact[0], exact[1], exact[2]

    ix0 = np.floor(fx).astype(np.int32)
    iy0 = np.floor(fy).astype(np.int32)
    iz0 = np.floor(fz).astype(np.int32)

    dx = fx - ix0   # [0, 1)
    dy = fy - iy0
    dz = fz - iz0

    # Gaussian weight (physical sigma in voxel units)
    gauss = np.exp(-0.5 * (dx**2 + dy**2 + dz**2) / (sigma_vox**2))

    ix1, iy1, iz1 = ix0 + 1, iy0 + 1, iz0 + 1
    _1dx, _1dy, _1dz = 1.0 - dx, 1.0 - dy, 1.0 - dz

    # Build (flat_index, weight) pairs for all 8 corners at once
    all_flat: list[np.ndarray] = []
    all_w:    list[np.ndarray] = []
    all_pv:   list[np.ndarray] = []

    corners = (
        (ix0, iy0, iz0, _1dx * _1dy * _1dz),
        (ix1, iy0, iz0,  dx  * _1dy * _1dz),
        (ix0, iy1, iz0, _1dx *  dy  * _1dz),
        (ix1, iy1, iz0,  dx  *  dy  * _1dz),
        (ix0, iy0, iz1, _1dx * _1dy *  dz ),
        (ix1, iy0, iz1,  dx  * _1dy *  dz ),
        (ix0, iy1, iz1, _1dx *  dy  *  dz ),
        (ix1, iy1, iz1,  dx  *  dy  *  dz ),
    )

    for cix, ciy, ciz, tri_w in corners:
        valid = (
            (cix >= 0) & (cix < nx) &
            (ciy >= 0) & (ciy < ny) &
            (ciz >= 0) & (ciz < nz)
        )
        if not valid.any():
            continue
        flat = (ciz[valid].astype(np.int64) * ny * nx
                + ciy[valid] * nx
                + cix[valid])
        w = (tri_w[valid] * gauss[valid]).astype(np.float64)
        all_flat.append(flat)
        all_w.append(w)
        all_pv.append(pix_flat[valid] * w)

    if not all_flat:
        return

    flat_cat = np.concatenate(all_flat)
    w_cat    = np.concatenate(all_w)
    pv_cat   = np.concatenate(all_pv)

    vox_sum   += np.bincount(flat_cat, weights=pv_cat, minlength=sz)
    vox_count += np.bincount(flat_cat, weights=w_cat,  minlength=sz)

