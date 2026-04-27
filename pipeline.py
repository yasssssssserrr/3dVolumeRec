"""
pipeline.py
===========
Final complete 3-D freehand ultrasound PNN reconstruction pipeline.

Datasets used
-------------
  forearm_tracked trials 1–6  :  near-100% valid tracking
  L123_trial_7               :  92.3% valid, no tracked version

Calibration
-----------
  All XML files use placeholder Identity for Image→Probe.
  The real spatial calibration (from 13 N-wire phantom runs) is loaded
  from Calibration_matrices.txt and applied as:

    p_tracker = T_ProbeToTracker · T_ImageToProbe · [u·ps, v·ps, 0, 1]ᵀ

Usage
-----
  python pipeline.py                        # default full run
  python pipeline.py --voxel 0.3           # higher resolution
  python pipeline.py --no-calib            # identity (for comparison)
  python pipeline.py --calib-type interp  # use interpolated matrix
  python pipeline.py --downsample 1        # full pixel resolution
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path as _Path

# Ensure the project root (this file's directory) is always importable,
# regardless of how the interpreter is invoked.
sys.path.insert(0, str(_Path(__file__).resolve().parent))
from pathlib import Path
from typing import Optional

import numpy as np

from config             import PipelineConfig
from us3d_io.calibration_reader import load_calibration
from us3d_io.mhd_reader      import load_sequence, SequenceData
from core.pnn           import PNNReconstructor
from core.vnn import VNNReconstructor
from core.spline import SplineReconstructor
from core.dl import DLReconstructor
from core.geometry      import orthonormalize_rotation
from us3d_io.volume_writer   import write_mhd, write_nifti
from viz.slicer         import (
    plot_ortho_slices,
    plot_axial_montage,
    plot_coverage,
    plot_calibration_comparison,
    plot_mip,
)

# ── Logging ───────────────────────────────────────────────────────────────────
# Force UTF-8 output on Windows (CP1252 terminals choke on ══ / → characters)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt= "%H:%M:%S",
    stream = sys.stdout,
)
logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3-D Freehand US PNN Reconstruction – Final Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _root = _Path(__file__).resolve().parent
    p.add_argument("--method",      choices=["pnn", "vnn", "spline", "dl"], default="vnn",
                   help="Reconstruction algorithm to use")
    p.add_argument("--base-dir",    type=Path,
                   default=_root / "tmp" / "dataset_3",
                   help="Root dataset directory")
    p.add_argument("--calib-file",  type=Path,
                   default=_root / "tmp" / "dataset_3" / "Calibration_matrices (1).txt",
                   help="Path to Calibration_matrices.txt")
    p.add_argument("--calib-type",  choices=["averaged", "interpolated"],
                   default="averaged",
                   help="Which calibration matrix to use")
    p.add_argument("--no-calib",    action="store_true",
                   help="Disable calibration (Identity, for comparison)")
    p.add_argument("--voxel",       type=float, default=0.5, metavar="MM",
                   help="Isotropic voxel size (mm)")
    p.add_argument("--downsample",  type=int,   default=4,   metavar="N",
                   help="Process every N-th pixel column (1=full res)")
    p.add_argument("--fill-radius", type=int,   default=10,  metavar="R",
                   help="Hole-fill dilation passes")
    p.add_argument("--smooth-sigma",type=float, default=1.5, metavar="S",
                   help="Post-process Gaussian smoothing sigma (voxels, 0=off)")
    p.add_argument("--no-trilinear",action="store_true",
                   help="Use nearest-voxel scatter instead of trilinear (PNN only)")
    p.add_argument("--bb-margin",   type=float, default=5.0, metavar="MM",
                   help="Bounding-box padding (mm)")
    p.add_argument("--vnn-radius",  type=float, default=5.0, metavar="MM",
                   help="Maximum fill distance for VNN (mm)")
    p.add_argument("--phantom",     action="store_true",
                   help="Reconstruct the phantom data instead of the forearm")
    p.add_argument("--out-dir",     type=Path,
                   default=_root / "tmp" / "output",
                   help="Output directory")
    p.add_argument("--stem",        type=str,   default="recon_pnn_calibrated",
                   help="Base filename stem for outputs")
    p.add_argument("--compare",     action="store_true",
                   help="Also run without calibration and produce comparison figure")
    # ── Deep Learning flags ───────────────────────────────────────────────────────────────
    p.add_argument("--dl-epochs",   type=int,   default=20, metavar="E",
                   help="U-Net training epochs (only used when --method dl)")
    p.add_argument("--dl-lr",       type=float, default=1e-3, metavar="LR",
                   help="Adam learning rate for U-Net training")
    p.add_argument("--dl-features", type=int,   default=16, metavar="F",
                   help="U-Net base feature channels (16=compact, 32=better quality)")
    p.add_argument("--dl-skip-train",action="store_true",
                   help="Skip training and load existing checkpoint from --out-dir")
    return p.parse_args()


# ── Loader ────────────────────────────────────────────────────────────────────

def load_all_sequences(cfg: PipelineConfig) -> list[SequenceData]:
    seqs = []
    for src in cfg.sources:
        try:
            seq = load_sequence(src.mhd_path, label=src.label)
            if seq.n_frames == 0:
                logger.warning("[%s] No valid frames – skipped", src.label)
                continue
            seqs.append(seq)
        except Exception as exc:
            logger.error("[%s] Failed: %s", src.label, exc)
    if not seqs:
        logger.critical("No sequences loaded. Aborting.")
        sys.exit(1)
    return seqs


# ── Quality report ────────────────────────────────────────────────────────────

def print_report(
    rec:     PNNReconstructor,
    seqs:    list[SequenceData],
    elapsed: float,
    calib:   Optional[np.ndarray],
) -> None:
    vol     = rec.volume
    mask    = rec.mask
    vs      = rec.voxel_size
    nz, ny, nx = vol.shape
    filled  = vol[mask]

    border = "=" * 65
    print(f"\n{border}")
    print("  3-D PNN RECONSTRUCTION  —  QUALITY REPORT")
    print(border)

    print(f"  Calibration      : {'Applied (T_ImageToProbe)' if calib is not None else 'Identity (disabled)'}")
    if calib is not None:
        det = np.linalg.det(calib[:3, :3])
        t   = calib[:3, 3]
        print(f"    det(R)         : {det:.6f}")
        print(f"    translation    : [{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}] mm")

    print(f"\n  Volume shape     : {nz} × {ny} × {nx}  (Z × Y × X voxels)")
    print(f"  Voxel size       : {vs:.2f} mm  (isotropic)")
    print(f"  Physical extent  : {nz*vs:.1f} × {ny*vs:.1f} × {nx*vs:.1f} mm")
    print(f"  Total voxels     : {mask.size:,}")
    print(f"  Filled voxels    : {int(mask.sum()):,}  ({100*mask.mean():.1f}%)")
    if filled.size:
        print(f"  Intensity range  : [{filled.min():.1f},  {filled.max():.1f}]")
        print(f"  Intensity mean   : {filled.mean():.1f}  ±  {filled.std():.1f}")
    print(f"  Memory (fp32)    : {vol.nbytes / 1e6:.1f} MB")
    print(f"  Runtime          : {elapsed:.1f} s")

    print(f"\n  Sequences used ({len(seqs)}) :")
    for seq in seqs:
        pct = 100 * seq.n_frames / max(seq.n_total, 1)
        print(f"    {seq.label:<40s}  {seq.n_frames:4d}/{seq.n_total:4d} frames ({pct:.0f}%)")

    print(border + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace | None = None) -> PNNReconstructor:
    if args is None:
        args = parse_args()

    t_start = time.perf_counter()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = PipelineConfig(
        method           = args.method,
        base_dir         = args.base_dir,
        calibration_file = args.calib_file,
        calibration_type = args.calib_type,
        apply_calibration= not args.no_calib,
        voxel_size_mm    = args.voxel,
        downsample_factor= args.downsample,
        hole_fill_radius = args.fill_radius,
        gaussian_sigma   = args.smooth_sigma,
        use_trilinear    = not args.no_trilinear,
        bb_margin_mm     = args.bb_margin,
        vnn_max_radius_mm= args.vnn_radius,
        use_phantom_data = args.phantom,
        output_dir       = args.out_dir,
        output_stem      = args.stem,
        dl_epochs        = args.dl_epochs,
        dl_lr            = args.dl_lr,
        dl_base_features = args.dl_features,
        dl_skip_training = args.dl_skip_train,
    )

    logger.info(
        "Pipeline start  |  method=%s  calib=%s (%s)  voxel=%.2f mm",
        cfg.method.upper(), not args.no_calib, args.calib_type, args.voxel
    )

    # ── Load calibration ──────────────────────────────────────────────────────
    T_calib: Optional[np.ndarray] = None
    if cfg.apply_calibration:
        if cfg.calibration_file.exists():
            T_calib = load_calibration(cfg.calibration_file, cfg.calibration_type)
            # Orthonormalise R (corrects det=0.9826 -> 1.0000)
            T_calib = orthonormalize_rotation(T_calib)
        else:
            logger.warning(
                "Calibration file not found (%s) – falling back to Identity",
                cfg.calibration_file,
            )

    # ── Load sequences ────────────────────────────────────────────────────────
    logger.info("══ Loading sequences ══")
    seqs = load_all_sequences(cfg)
    logger.info(
        "Loaded %d sequences | %d valid frames total",
        len(seqs), sum(s.n_frames for s in seqs),
    )

    # ── Calibrated reconstruction ─────────────────────────────────────────────
    logger.info(f"Pipeline start  |  method={cfg.method.upper()}  calib={cfg.apply_calibration} ({cfg.calibration_type})  voxel={cfg.voxel_size_mm:.2f} mm")

    if cfg.method == "vnn":
        rec = VNNReconstructor(cfg, image_to_probe=T_calib)
    elif cfg.method == "spline":
        rec = SplineReconstructor(cfg, image_to_probe=T_calib)
    elif cfg.method == "dl":
        rec = DLReconstructor(
            cfg,
            image_to_probe  = T_calib,
            epochs          = cfg.dl_epochs,
            lr              = cfg.dl_lr,
            base_features   = cfg.dl_base_features,
            skip_training   = cfg.dl_skip_training,
        )
    else:
        rec = PNNReconstructor(cfg, image_to_probe=T_calib)
    rec.reconstruct(seqs)

    stem = args.stem
    vs   = rec.voxel_size
    vol  = rec.volume
    orig = rec.origin

    # ── Export ────────────────────────────────────────────────────────────────
    logger.info("══ Writing outputs ══")
    if cfg.export_mhd:
        write_mhd(vol, cfg.output_dir, stem, orig, vs)
    if cfg.export_nifti:
        write_nifti(vol, cfg.output_dir, stem, orig, vs)

    # ── Visualise calibrated ──────────────────────────────────────────────────
    if cfg.export_figures:
        logger.info("== Generating figures ==")
        plot_ortho_slices(
            vol, cfg.output_dir, stem, vs,
            mask=rec.mask,
            calibrated=(T_calib is not None),
            dpi=cfg.dpi, cmap=cfg.colormap,
        )
        plot_mip(
            vol, cfg.output_dir, stem, vs,
            dpi=cfg.dpi, cmap=cfg.colormap,
        )
        plot_axial_montage(
            vol, cfg.output_dir, stem, vs,
            mask=rec.mask,
            n_slices=25, dpi=cfg.dpi, cmap=cfg.colormap,
        )
        plot_coverage(
            rec.mask, cfg.output_dir, stem, vs,
            n_trials=len(seqs), dpi=cfg.dpi,
        )

    # ── Optional comparison (with vs without calibration) ─────────────────────
    if args.compare and T_calib is not None:
        logger.info("══ No-calibration comparison run ══")
        rec_nc = ReconstructorClass(cfg, image_to_probe=None)
        rec_nc.reconstruct(seqs)
        plot_calibration_comparison(
            vol_calib  = vol,
            vol_nocal  = rec_nc.volume,
            output_dir = cfg.output_dir,
            stem       = stem,
            voxel_size = vs,
            dpi        = cfg.dpi,
            cmap       = cfg.colormap,
        )

    # ── Report ────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print_report(rec, seqs, elapsed, T_calib)

    return rec


if __name__ == "__main__":
    run()
