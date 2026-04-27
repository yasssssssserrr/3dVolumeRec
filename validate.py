"""
validate.py – Leave-One-Out Cross Validation for 3D Ultrasound

This script drops a subset of 2D frames, reconstructs the 3D volume with the remaining frames,
and then extracts virtual 2D slices from the volume to compare against the dropped frames.

Metrics:
- MAE (Mean Absolute Error): Average absolute difference in pixel intensities (0-255).
- NCC (Normalized Cross-Correlation): Structural similarity measure (-1.0 to 1.0).
"""
import argparse
import logging
import sys
import copy
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from config import PipelineConfig
from us3d_io.calibration_reader import load_calibration
from us3d_io.mhd_reader import load_sequence
from core.pnn import PNNReconstructor
from core.vnn import VNNReconstructor
from core.spline import SplineReconstructor
from core.dl import DLReconstructor
from core.geometry import orthonormalize_rotation, effective_transform, make_image_points, transform_to_tracker

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Cross-Validation for 3D US Reconstruction")
    _root = Path(__file__).resolve().parent
    p.add_argument("--method",      choices=["pnn", "vnn", "spline", "dl"], default="vnn", help="Algorithm to validate")
    p.add_argument("--base-dir",    type=Path, default=_root / "tmp" / "dataset_3")
    p.add_argument("--calib-file",  type=Path, default=_root / "tmp" / "dataset_3" / "Calibration_matrices (1).txt")
    p.add_argument("--calib-type",  choices=["averaged", "interpolated"], default="averaged")
    p.add_argument("--voxel",       type=float, default=0.5)
    p.add_argument("--downsample",  type=int,   default=4)
    p.add_argument("--fill-radius", type=int,   default=10)
    p.add_argument("--smooth-sigma",type=float, default=1.5)
    p.add_argument("--vnn-radius",  type=float, default=5.0)
    p.add_argument("--phantom",     action="store_true", help="Validate on phantom data")
    p.add_argument("--val-split",   type=int,   default=20, help="Hold out every N-th frame for validation")
    return p.parse_args()

def compute_ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom < 1e-8:
        return np.nan
    return np.sum(a * b) / denom

def run_validation():
    args = parse_args()
    
    cfg = PipelineConfig(
        method=args.method,
        base_dir=args.base_dir,
        calibration_file=args.calib_file,
        calibration_type=args.calib_type,
        voxel_size_mm=args.voxel,
        downsample_factor=args.downsample,
        hole_fill_radius=args.fill_radius,
        gaussian_sigma=args.smooth_sigma,
        vnn_max_radius_mm=args.vnn_radius,
        use_phantom_data=args.phantom,
        export_mhd=False, export_nifti=False, export_figures=False
    )
    
    logger.info("== Loading sequences ==")
    seqs = []
    for src in cfg.sources:
        try:
            seq = load_sequence(src.mhd_path, label=src.label)
            if seq.n_frames > 0:
                seqs.append(seq)
        except Exception as e:
            logger.error("Failed to load %s: %s", src.label, e)
            
    if not seqs:
        return
        
    T_calib = load_calibration(cfg.calibration_file, cfg.calibration_type)
    T_calib = orthonormalize_rotation(T_calib)
    
    # Split data
    train_seqs = copy.deepcopy(seqs)
    val_frames = []
    
    for i, seq in enumerate(train_seqs):
        train_f = []
        for j, f in enumerate(seq.frames):
            if j % args.val_split == 0:
                val_frames.append((seq.image_height, seq.image_width, f))
            else:
                train_f.append(f)
        seq.frames = train_f
        
    logger.info("Split: %d train frames, %d validation frames (1 in %d)", 
                sum(s.n_frames for s in train_seqs), len(val_frames), args.val_split)
                
    # Reconstruct
    logger.info("== Reconstructing Volume (%s) ==", cfg.method.upper())
    if cfg.method == "vnn":
        rec = VNNReconstructor(cfg, image_to_probe=T_calib)
    elif cfg.method == "spline":
        rec = SplineReconstructor(cfg, image_to_probe=T_calib)
    elif cfg.method == "dl":
        rec = DLReconstructor(cfg, image_to_probe=T_calib)
    else:
        rec = PNNReconstructor(cfg, image_to_probe=T_calib)
    rec.reconstruct(train_seqs)
    
    vol = rec.volume
    mask = rec.mask
    origin = rec.origin
    vs = rec.voxel_size
    
    # Evaluate
    logger.info("== Evaluating Validation Frames ==")
    all_mae = []
    all_ncc = []
    
    for h, w, frame in val_frames:
        ps = frame.pixel_spacing_mm
        T_eff = effective_transform(frame.probe_to_tracker, T_calib)
        
        # Sample at ds=2 for fast but accurate evaluation
        ds = 2
        pts_img = make_image_points(h, w, ps, downsample=ds)
        pts_t = transform_to_tracker(pts_img, T_eff)
        
        # Convert to voxel coordinates
        coords = (pts_t - origin[:, None]) / vs
        
        # Sample the reconstructed volume (trilinear interpolation)
        pred_flat = map_coordinates(vol, [coords[2], coords[1], coords[0]], order=1, mode='constant', cval=0.0)
        
        # Sample the mask to only evaluate pixels that actually fall inside the reconstructed tissue
        valid_flat = map_coordinates(mask.astype(float), [coords[2], coords[1], coords[0]], order=0, mode='constant', cval=0.0) > 0.5
        
        if not valid_flat.any():
            continue
            
        true_flat = frame.image[::ds, ::ds].ravel().astype(np.float64)
        
        true_valid = true_flat[valid_flat]
        pred_valid = pred_flat[valid_flat]
        
        if len(true_valid) < 100:  # Skip frames that barely intersect the volume
            continue
            
        mae = float(np.mean(np.abs(true_valid - pred_valid)))
        ncc = compute_ncc(true_valid, pred_valid)
        
        if not np.isnan(ncc):
            all_mae.append(mae)
            all_ncc.append(ncc)
            
    final_mae = np.mean(all_mae)
    final_ncc = np.mean(all_ncc)
    
    logger.info("=====================================")
    logger.info("  CROSS-VALIDATION RESULTS (%s)", cfg.method.upper())
    logger.info("=====================================")
    logger.info("  Frames evaluated : %d / %d", len(all_mae), len(val_frames))
    logger.info("  Mean Abs Error   : %.2f (pixels 0-255)", final_mae)
    logger.info("  Norm Cross-Corr  : %.4f (-1 to 1, higher is better)", final_ncc)
    logger.info("=====================================")

if __name__ == "__main__":
    run_validation()
