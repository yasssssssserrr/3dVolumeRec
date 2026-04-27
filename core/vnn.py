"""
core/vnn.py – Calibration-aware VNN reconstructor

Voxel Nearest Neighbor (VNN) differs from Pixel Nearest Neighbor (PNN).
Instead of pushing pixels into voxels and smoothing the gaps, VNN guarantees
a perfectly dense volume by making every voxel query the exact nearest pixel 
from the 2D scan planes.

This is implemented efficiently using SciPy's Exact Distance Transform (EDT).
"""
from __future__ import annotations
import logging, time, sys
from pathlib import Path as _Path
from typing import List, Optional

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from config import PipelineConfig
from core.geometry import (
    build_voxel_grid, compute_volume_bounds, effective_transform,
    make_image_points, transform_to_tracker,
)
from us3d_io.mhd_reader import SequenceData

logger = logging.getLogger(__name__)


class VNNReconstructor:
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

        # ── Step 2: Initialize anchor grid ────────────────────────────────────
        # We use a simple nearest-voxel scatter to map pixels onto the voxel grid.
        # This provides the "seed" pixels for the Distance Transform.
        logger.info("== Step 2/4  VNN anchor initialization ==")
        sz = nz * ny * nx
        vox_sum   = np.zeros(sz, dtype=np.float64)
        vox_count = np.zeros(sz, dtype=np.float64)
        self._bin_fill_anchors(sequences, vox_sum, vox_count, nz, ny, nx)

        vox_sum   = vox_sum.reshape(nz, ny, nx)
        vox_count = vox_count.reshape(nz, ny, nx)
        anchor_mask = vox_count > 0
        
        anchor_vol = np.zeros((nz, ny, nx), dtype=np.float32)
        anchor_vol[anchor_mask] = (
            vox_sum[anchor_mask] / vox_count[anchor_mask]
        ).astype(np.float32)
        del vox_sum, vox_count

        # ── Step 3: Exact Distance Transform (VNN) ────────────────────────────
        logger.info("== Step 3/4  Exact Distance Transform (VNN) ==")
        t_edt = time.perf_counter()
        
        # Calculate distances from every empty voxel to the nearest filled voxel,
        # and get the (z,y,x) indices of that nearest voxel.
        distances, nearest_idx = distance_transform_edt(
            ~anchor_mask, 
            return_distances=True, 
            return_indices=True
        )
        
        # Apply the VNN mapping: every voxel gets the value of its nearest anchor
        vnn_vol = anchor_vol[nearest_idx[0], nearest_idx[1], nearest_idx[2]]
        
        # Apply distance cutoff so we don't extrapolate out to infinity
        max_dist_voxels = cfg.vnn_max_radius_mm / cfg.voxel_size_mm
        filled_mask = distances <= max_dist_voxels
        
        # Zero out voxels beyond the maximum radius
        vnn_vol[~filled_mask] = 0.0
        
        logger.info("  EDT completed in %.1f s", time.perf_counter() - t_edt)

        # ── Step 4: Gaussian smoothing ────────────────────────────────────────
        if cfg.gaussian_sigma > 0:
            logger.info("== Step 4/4  Gaussian smoothing (sigma=%.1f vox) ==",
                        cfg.gaussian_sigma)
            volume = self._smooth(vnn_vol, filled_mask, cfg.gaussian_sigma)
        else:
            logger.info("== Step 4/4  Gaussian smoothing skipped ==")
            volume = vnn_vol

        self._volume = volume
        self._mask   = filled_mask

        elapsed  = time.perf_counter() - t0
        pct_fill = 100.0 * filled_mask.sum() / filled_mask.size
        logger.info("Done %.1f s  |  %.1f%% filled", elapsed, pct_fill)
        return volume

    # ── Anchor filler ─────────────────────────────────────────────────────────

    def _bin_fill_anchors(self, sequences, vox_sum, vox_count, nz, ny, nx):
        """Map all pixels to their nearest voxel to seed the EDT."""
        cfg    = self.cfg
        origin = self._origin
        vs     = cfg.voxel_size_mm
        ds     = cfg.downsample_factor
        sz     = nz * ny * nx
        total  = sum(s.n_frames for s in sequences)
        done   = 0
        pts_cache: dict = {}

        for seq in sequences:
            h, w = seq.image_height, seq.image_width
            for frame in seq.frames:
                ps    = frame.pixel_spacing_mm
                T_eff = effective_transform(frame.probe_to_tracker, self.T_calib)

                key = (h, w, ps, ds)
                if key not in pts_cache:
                    pts_cache[key] = make_image_points(h, w, ps, downsample=ds)
                pts_img = pts_cache[key]

                pts_t = transform_to_tracker(pts_img, T_eff)
                exact = (pts_t - origin[:, None]) / vs

                pix_flat = frame.image[::ds, ::ds].ravel().astype(np.float64)

                # Pure nearest-voxel scatter (no Gaussian weighting needed for VNN seeds)
                idx = np.round(exact).astype(np.int32)
                ix, iy, iz = idx[0], idx[1], idx[2]
                
                valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
                if valid.any():
                    ix, iy, iz = ix[valid], iy[valid], iz[valid]
                    flat = iz.astype(np.int64) * ny * nx + iy * nx + ix
                    pv   = pix_flat[valid]
                    vox_sum   += np.bincount(flat, weights=pv, minlength=sz)
                    vox_count += np.bincount(flat, minlength=sz)

                done += 1
                if done % 100 == 0 or done == total:
                    logger.info("  Anchor init: %d/%d (%.0f%%)",
                                done, total, 100 * done / total)

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
