import logging
import time
import numpy as np
import scipy.ndimage as ndi
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from us3d_io.mhd_reader import SequenceData
from config import PipelineConfig
from core.geometry import (
    build_voxel_grid, compute_volume_bounds, effective_transform,
    make_image_points, transform_to_tracker,
)

logger = logging.getLogger(__name__)

class SplineReconstructor:
    """
    Advanced PyTorch-accelerated Spline Reconstructor.
    Uses Backward Mapping:
    1. Forward Splat to find the closest frame for every voxel (using Exact Distance Transform).
    2. For every voxel, project its 3D coordinate backwards onto its closest 2D frame.
    3. Sample the 2D frame at the sub-pixel coordinate using Bicubic Splines (grid_sample).
    """
    def __init__(self, config: PipelineConfig,
                 image_to_probe: Optional[np.ndarray] = None):
        self.cfg     = config
        self.T_calib = image_to_probe
        self._volume = self._mask = self._origin = self._shape = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAS_TORCH else None
        
        if HAS_TORCH:
            logger.info(f"SplineReconstructor initialized. PyTorch backend: {self.device.type.upper()}")
        else:
            logger.warning("PyTorch not found! Falling back to SciPy map_coordinates (CPU only).")

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

    def _compute_bounding_box(self, sequences):
        logger.info("== Step 1/4  Bounding-box ==")
        bb_min, bb_max = compute_volume_bounds(
            sequences, self.T_calib, downsample=8,
            margin_mm=self.cfg.bb_margin_mm,
        )
        self._origin, self._shape = build_voxel_grid(bb_min, bb_max, self.cfg.voxel_size_mm)
        nz, ny, nx = self._shape
        logger.info("Voxels %d×%d×%d = %s  (%.0f MB f32)",
                    nz, ny, nx, f"{nz*ny*nx:,}", nz*ny*nx*4/1e6)

    def reconstruct(self, sequences: List[SequenceData]) -> None:
        if not sequences:
            return

        t0 = time.time()
        # 1. Bounding box & Grid
        self._compute_bounding_box(sequences)
        if self.shape is None:
            return
            
        Z, Y, X = self.shape
        total_voxels = Z * Y * X

        # We need a way to assign each voxel to its nearest frame.
        # Create a grid to store the frame ID that hits each voxel.
        frame_id_vol = np.full(self.shape, -1, dtype=np.int32)
        
        # Flattened frame data
        all_frames = []
        all_T_tracker_to_image = [] # Inverse transforms
        all_ps = []
        
        frame_counter = 0
        
        logger.info("== Step 2/4  Spline Anchor Initialization (Forward) ==")
        for seq_idx, seq in enumerate(sequences):
            h, w = seq.image_height, seq.image_width
            for frame in seq.frames:
                img = frame.image
                T_p2t = frame.probe_to_tracker
                
                # Image -> Tracker
                T_i2t = T_p2t @ self.T_calib
                
                # Store inverse for backward mapping
                T_t2i = np.linalg.inv(T_i2t)
                all_frames.append(img)
                all_T_tracker_to_image.append(T_t2i)
                all_ps.append(frame.pixel_spacing_mm)
                
                # Forward Splat (just to get anchor points for the EDT)
                # Only sample a subset of pixels to be fast for anchoring
                step = 4
                uu, vv = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
                uu = uu.flatten()
                vv = vv.flatten()
                
                pts_img = np.stack([
                    uu * frame.pixel_spacing_mm,
                    vv * frame.pixel_spacing_mm,
                    np.zeros_like(uu),
                    np.ones_like(uu)
                ], axis=0) # (4, N)
                
                pts_tracker = T_i2t @ pts_img # (4, N)
                
                # To voxel indices
                vx = np.round((pts_tracker[0] - self.origin[0]) / self.cfg.voxel_size_mm).astype(np.int32)
                vy = np.round((pts_tracker[1] - self.origin[1]) / self.cfg.voxel_size_mm).astype(np.int32)
                vz = np.round((pts_tracker[2] - self.origin[2]) / self.cfg.voxel_size_mm).astype(np.int32)
                
                valid = (vx >= 0) & (vx < X) & (vy >= 0) & (vy < Y) & (vz >= 0) & (vz < Z)
                
                # Splat the frame ID
                frame_id_vol[vz[valid], vy[valid], vx[valid]] = frame_counter
                
                frame_counter += 1
                if frame_counter % 100 == 0:
                    logger.info(f"  Anchors initialized: {frame_counter} frames")

        logger.info(f"  Anchors complete. Frames loaded: {frame_counter}")

        # 2. Exact Distance Transform to find nearest frame ID for every voxel
        logger.info("== Step 3/4  Exact Distance Transform (EDT) ==")
        t_edt = time.time()
        mask = (frame_id_vol >= 0)
        
        # distance_transform_edt returns the indices of the closest background element if we invert the mask
        # wait, we want the closest True element for every False element.
        # So background is where mask is True (distance 0). We want distance from False to True.
        # In scipy, distance is to the closest *zero* value. 
        # So we want zeros at the anchors.
        edt_input = (~mask).astype(int) 
        
        # Get distances and indices
        distances, indices = ndi.distance_transform_edt(edt_input, return_distances=True, return_indices=True)
        
        # Max radius cutoff
        valid_fill = distances <= (self.cfg.vnn_max_radius_mm / self.cfg.voxel_size_mm)
        
        # The nearest frame for every voxel
        nearest_frame_vol = frame_id_vol[tuple(indices)]
        
        # Only keep voxels within the max radius
        nearest_frame_vol[~valid_fill] = -1
        
        logger.info(f"  EDT completed in {time.time() - t_edt:.1f} s")
        
        # 3. Backward Mapping with Splines
        logger.info("== Step 4/4  Bicubic Spline Backward Mapping ==")
        t_back = time.time()
        
        volume = np.zeros(self.shape, dtype=np.float32)
        count_vol = np.zeros(self.shape, dtype=np.uint8) # Just to mark filled
        
        # Precompute 3D physical coordinates for the entire grid
        zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing='ij')
        coords_x_mm = xx * self.cfg.voxel_size_mm + self.origin[0]
        coords_y_mm = yy * self.cfg.voxel_size_mm + self.origin[1]
        coords_z_mm = zz * self.cfg.voxel_size_mm + self.origin[2]
        
        # We will process frame by frame. For each frame, find which voxels assigned to it.
        if HAS_TORCH:
            with torch.no_grad():
                for f_id in range(frame_counter):
                    mask_f = (nearest_frame_vol == f_id)
                    if not np.any(mask_f):
                        continue
                        
                    # Get 3D coords of target voxels
                    X_tgt = coords_x_mm[mask_f]
                    Y_tgt = coords_y_mm[mask_f]
                    Z_tgt = coords_z_mm[mask_f]
                    N_pts = len(X_tgt)
                    
                    pts_3d = np.stack([X_tgt, Y_tgt, Z_tgt, np.ones_like(X_tgt)], axis=0) # (4, N)
                    
                    T_inv = all_T_tracker_to_image[f_id] # (4, 4)
                    pts_2d = T_inv @ pts_3d # (4, N)
                    ps = all_ps[f_id]
                    
                    # Pixel coordinates
                    u = pts_2d[0] / ps
                    v = pts_2d[1] / ps
                    
                    img = all_frames[f_id].astype(np.float32)
                    H, W = img.shape
                    
                    # Normalize for grid_sample [-1, 1]
                    norm_u = (u / (W - 1)) * 2.0 - 1.0
                    norm_v = (v / (H - 1)) * 2.0 - 1.0
                    
                    # To PyTorch
                    grid = torch.tensor(np.stack([norm_u, norm_v], axis=-1), dtype=torch.float32, device=self.device) # (N, 2)
                    grid = grid.view(1, 1, N_pts, 2) # (N, C, H_out, W_out, 2) -> (1, 1, N_pts, 2)
                    
                    img_tensor = torch.tensor(img, dtype=torch.float32, device=self.device)
                    img_tensor = img_tensor.view(1, 1, H, W)
                    
                    # Bicubic Spline interpolation!
                    sampled = F.grid_sample(img_tensor, grid, mode='bicubic', padding_mode='zeros', align_corners=True)
                    
                    # Extract back to numpy
                    sampled_np = sampled.view(-1).cpu().numpy()
                    
                    volume[mask_f] = sampled_np
                    count_vol[mask_f] = 1
                    
        else:
            # Fallback to SciPy
            for f_id in range(frame_counter):
                mask_f = (nearest_frame_vol == f_id)
                if not np.any(mask_f):
                    continue
                    
                X_tgt = coords_x_mm[mask_f]
                Y_tgt = coords_y_mm[mask_f]
                Z_tgt = coords_z_mm[mask_f]
                
                pts_3d = np.stack([X_tgt, Y_tgt, Z_tgt, np.ones_like(X_tgt)], axis=0)
                T_inv = all_T_tracker_to_image[f_id]
                pts_2d = T_inv @ pts_3d
                ps = all_ps[f_id]
                
                u = pts_2d[0] / ps
                v = pts_2d[1] / ps
                
                img = all_frames[f_id].astype(np.float32)
                
                # order=3 is cubic spline
                sampled_np = ndi.map_coordinates(img, [v, u], order=3, mode='constant', cval=0.0)
                
                volume[mask_f] = sampled_np
                count_vol[mask_f] = 1

        self._volume = volume
        self._mask = count_vol > 0
        
        filled = np.sum(self._mask)
        logger.info(f"  Backward mapping completed in {time.time() - t_back:.1f} s | {filled/total_voxels*100:.1f}% filled")

