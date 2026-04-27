"""
dl_compare.py
=============
Runs all four reconstruction methods (VNN, PNN, Spline, DL) on the same
forearm dataset, evaluates each with LOOCV, and produces:

  tmp/output/comparison_results.csv   – numeric metrics
  tmp/output/method_comparison.png    – side-by-side orthoview (4 × 3 grid)
  tmp/output/metric_bars.png          – bar chart of MAE and NCC

Usage
-----
  python dl_compare.py                         # full run (20 DL epochs)
  python dl_compare.py --epochs 5              # quick smoke-test
  python dl_compare.py --skip-training         # reuse saved checkpoint
  python dl_compare.py --voxel 0.5            # custom voxel size
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates

# ── UTF-8 on Windows ──────────────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from config import PipelineConfig
from us3d_io.calibration_reader import load_calibration
from us3d_io.mhd_reader import load_sequence
from core.geometry import (
    orthonormalize_rotation, effective_transform,
    make_image_points, transform_to_tracker,
)
from core.pnn    import PNNReconstructor
from core.vnn    import VNNReconstructor
from core.spline import SplineReconstructor
from core.dl     import DLReconstructor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare VNN / PNN / Spline / DL reconstruction methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-dir",      type=Path,
                   default=_ROOT / "tmp" / "dataset_3")
    p.add_argument("--calib-file",    type=Path,
                   default=_ROOT / "tmp" / "dataset_3" / "Calibration_matrices (1).txt")
    p.add_argument("--calib-type",    choices=["averaged", "interpolated"],
                   default="averaged")
    p.add_argument("--voxel",         type=float, default=0.5, metavar="MM")
    p.add_argument("--downsample",    type=int,   default=4,   metavar="N")
    p.add_argument("--fill-radius",   type=int,   default=10,  metavar="R")
    p.add_argument("--smooth-sigma",  type=float, default=1.5, metavar="S")
    p.add_argument("--vnn-radius",    type=float, default=5.0, metavar="MM")
    p.add_argument("--val-split",     type=int,   default=20,
                   help="Hold out every N-th frame for validation")
    p.add_argument("--out-dir",       type=Path,
                   default=_ROOT / "tmp" / "output")
    # DL-specific
    p.add_argument("--epochs",        type=int,   default=20,  metavar="E",
                   help="U-Net training epochs")
    p.add_argument("--dl-lr",         type=float, default=1e-3, metavar="LR")
    p.add_argument("--dl-features",   type=int,   default=16,  metavar="F")
    p.add_argument("--skip-training", action="store_true",
                   help="Load existing DL checkpoint instead of training")
    p.add_argument("--methods",       nargs="+",
                   choices=["vnn", "pnn", "spline", "dl"],
                   default=["vnn", "pnn", "spline", "dl"],
                   help="Which methods to run (subset to skip slow ones)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if denom < 1e-8:
        return float("nan")
    return float(np.sum(a * b) / denom)


def evaluate_loocv(
    rec,
    val_frames: List[Tuple[int, int, object]],
    val_split_ds: int = 2,
) -> Tuple[float, float]:
    """
    Sample the reconstructed volume at held-out frame locations.
    Returns (mean_mae, mean_ncc).
    """
    vol    = rec.volume
    mask   = rec.mask
    origin = rec.origin
    vs     = rec.voxel_size

    all_mae, all_ncc = [], []
    for h, w, frame in val_frames:
        ps    = frame.pixel_spacing_mm
        T_eff = effective_transform(frame.probe_to_tracker, rec.T_calib)
        pts   = make_image_points(h, w, ps, downsample=val_split_ds)
        pts_t = transform_to_tracker(pts, T_eff)
        coords = (pts_t - origin[:, None]) / vs

        pred = map_coordinates(
            vol, [coords[2], coords[1], coords[0]],
            order=1, mode="constant", cval=0.0,
        )
        in_mask = map_coordinates(
            mask.astype(float), [coords[2], coords[1], coords[0]],
            order=0, mode="constant", cval=0.0,
        ) > 0.5

        if not in_mask.any():
            continue

        true = frame.image[::val_split_ds, ::val_split_ds].ravel().astype(np.float64)
        t_v, p_v = true[in_mask], pred[in_mask]
        if len(t_v) < 100:
            continue

        all_mae.append(float(np.mean(np.abs(t_v - p_v))))
        ncc = compute_ncc(t_v, p_v)
        if not np.isnan(ncc):
            all_ncc.append(ncc)

    if not all_mae:
        return float("nan"), float("nan")
    return float(np.mean(all_mae)), float(np.mean(all_ncc))


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def _mid(v: np.ndarray, ax: int) -> int:
    return v.shape[ax] // 2


def plot_ortho_comparison(
    volumes: Dict[str, np.ndarray],
    metrics: Dict[str, Dict[str, float]],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """4-column × 3-row orthoview comparison figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        logger.warning("matplotlib not available – skipping comparison figure")
        return

    methods = list(volumes.keys())
    n_cols  = len(methods)
    rows    = ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"]
    n_rows  = len(rows)

    cmap = "gray"
    fig  = plt.figure(figsize=(5 * n_cols, 4.5 * n_rows), facecolor="#0d0d0d")
    gs   = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=0.08, wspace=0.04,
        left=0.04, right=0.96, top=0.93, bottom=0.04,
    )

    HIGHLIGHT = {
        "vnn":    "#4FC3F7",   # light blue
        "pnn":    "#A5D6A7",   # light green
        "spline": "#FFB74D",   # amber
        "dl":     "#EF9A9A",   # rose
    }

    for col, method in enumerate(methods):
        vol = volumes[method]
        nz, ny, nx = vol.shape
        vmin, vmax = np.percentile(vol[vol > 0], [2, 98]) if vol.any() else (0, 255)
        colour = HIGHLIGHT.get(method, "white")

        slices = [
            vol[_mid(vol, 0), :, :],   # axial
            vol[:, _mid(vol, 1), :],   # coronal
            vol[:, :, _mid(vol, 2)],   # sagittal
        ]

        for row, sl in enumerate(slices):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(sl, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            ax.axis("off")

            # Column header
            if row == 0:
                mae = metrics.get(method, {}).get("mae", float("nan"))
                ncc = metrics.get(method, {}).get("ncc", float("nan"))
                title = (
                    f"{method.upper()}\n"
                    f"MAE={mae:.2f}  NCC={ncc:.4f}"
                    if not np.isnan(mae) else method.upper()
                )
                ax.set_title(title, fontsize=11, fontweight="bold",
                             color=colour, pad=4)

            # Row label
            if col == 0:
                ax.set_ylabel(rows[row], fontsize=9, color="#cccccc",
                              rotation=90, labelpad=6)

            # Coloured border
            for spine in ax.spines.values():
                spine.set_edgecolor(colour)
                spine.set_linewidth(1.5)
                spine.set_visible(True)

    fig.suptitle(
        "3-D Freehand Ultrasound – Reconstruction Method Comparison",
        fontsize=14, fontweight="bold", color="white", y=0.97,
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Orthoview saved → %s", out_path)


def plot_metric_bars(
    metrics: Dict[str, Dict[str, float]],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Side-by-side bar chart of MAE (↓) and NCC (↑)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    methods = list(metrics.keys())
    mae_vals = [metrics[m].get("mae", float("nan")) for m in methods]
    ncc_vals = [metrics[m].get("ncc", float("nan")) for m in methods]

    COLOURS = {
        "vnn":    "#4FC3F7",
        "pnn":    "#A5D6A7",
        "spline": "#FFB74D",
        "dl":     "#EF9A9A",
    }
    bar_colours = [COLOURS.get(m, "#888888") for m in methods]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 4.5), facecolor="#0d0d0d"
    )
    fig.patch.set_facecolor("#0d0d0d")

    x = np.arange(len(methods))
    w = 0.5

    for ax, vals, ylabel, title in [
        (ax1, mae_vals, "MAE  (pixels, lower is better)", "Mean Absolute Error  ↓"),
        (ax2, ncc_vals, "NCC  (higher is better)",         "Normalised Cross-Corr  ↑"),
    ]:
        bars = ax.bar(x, vals, width=w, color=bar_colours,
                      edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_facecolor("#1a1a1a")
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in methods],
                           fontsize=11, color="white")
        ax.set_ylabel(ylabel, color="#cccccc", fontsize=9)
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")
        ax.tick_params(colors="#cccccc")
        ax.spines["bottom"].set_color("#444444")
        ax.spines["left"].set_color("#444444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#333333", zorder=0)

        # Value labels on bars
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=9, color="white", fontweight="bold",
                )

    fig.suptitle(
        "LOOCV Metrics by Reconstruction Method",
        fontsize=13, fontweight="bold", color="white", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Metric bar chart saved → %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_comparison() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load calibration ──────────────────────────────────────────────────────
    T_calib: Optional[np.ndarray] = None
    if args.calib_file.exists():
        T_calib = load_calibration(args.calib_file, args.calib_type)
        T_calib = orthonormalize_rotation(T_calib)
    else:
        logger.warning("Calibration file not found – using Identity")

    # ── Shared config (no export – just reconstruction + metrics) ─────────────
    base_cfg = dict(
        base_dir          = args.base_dir,
        calibration_file  = args.calib_file,
        calibration_type  = args.calib_type,
        voxel_size_mm     = args.voxel,
        downsample_factor = args.downsample,
        hole_fill_radius  = args.fill_radius,
        gaussian_sigma    = args.smooth_sigma,
        vnn_max_radius_mm = args.vnn_radius,
        export_mhd        = False,
        export_nifti      = False,
        export_figures    = False,
        output_dir        = args.out_dir,
    )

    # ── Load sequences + train/val split ──────────────────────────────────────
    logger.info("══ Loading sequences ══")
    cfg0 = PipelineConfig(method="vnn", **base_cfg)
    all_seqs = []
    for src in cfg0.sources:
        try:
            s = load_sequence(src.mhd_path, label=src.label)
            if s.n_frames > 0:
                all_seqs.append(s)
        except Exception as exc:
            logger.error("Failed to load %s: %s", src.label, exc)

    if not all_seqs:
        logger.critical("No sequences loaded – aborting.")
        sys.exit(1)

    logger.info(
        "Loaded %d sequences | %d frames total",
        len(all_seqs), sum(s.n_frames for s in all_seqs),
    )

    # Build train / val split (deepcopy so each method starts from same data)
    all_seqs_copy = copy.deepcopy(all_seqs)
    val_frames: List[Tuple[int, int, object]] = []
    for seq in all_seqs_copy:
        kept = []
        for j, f in enumerate(seq.frames):
            if j % args.val_split == 0:
                val_frames.append((seq.image_height, seq.image_width, f))
            else:
                kept.append(f)
        seq.frames = kept

    logger.info(
        "Split: %d train frames | %d val frames (1 in %d)",
        sum(s.n_frames for s in all_seqs_copy),
        len(val_frames), args.val_split,
    )

    # ── Run each method ───────────────────────────────────────────────────────
    METHOD_MAP = {
        "vnn":    lambda cfg: VNNReconstructor(cfg, image_to_probe=T_calib),
        "pnn":    lambda cfg: PNNReconstructor(cfg, image_to_probe=T_calib),
        "spline": lambda cfg: SplineReconstructor(cfg, image_to_probe=T_calib),
        "dl":     lambda cfg: DLReconstructor(
            cfg,
            image_to_probe  = T_calib,
            epochs          = args.epochs,
            lr              = args.dl_lr,
            base_features   = args.dl_features,
            skip_training   = args.skip_training,
        ),
    }

    volumes: Dict[str, np.ndarray] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    runtimes: Dict[str, float] = {}

    for method in args.methods:
        logger.info("\n══════════════════════════════════════")
        logger.info("  Running method : %s", method.upper())
        logger.info("══════════════════════════════════════")

        cfg = PipelineConfig(method=method, **base_cfg)
        train_seqs = copy.deepcopy(all_seqs_copy)

        rec = METHOD_MAP[method](cfg)
        # Expose T_calib so evaluate_loocv can use it
        rec.T_calib = T_calib

        t0 = time.perf_counter()
        rec.reconstruct(train_seqs)
        elapsed = time.perf_counter() - t0
        runtimes[method] = elapsed

        logger.info("  Reconstruction done in %.1f s", elapsed)
        logger.info("  Evaluating LOOCV metrics …")

        mae, ncc = evaluate_loocv(rec, val_frames)
        metrics[method] = {"mae": mae, "ncc": ncc, "time_s": elapsed}
        volumes[method] = rec.volume

        logger.info("  %s  MAE=%.2f  NCC=%.4f  time=%.1f s",
                    method.upper(), mae, ncc, elapsed)

    # ── Results table ─────────────────────────────────────────────────────────
    border = "═" * 65
    print(f"\n{border}")
    print("  RECONSTRUCTION METHOD COMPARISON")
    print(border)
    print(f"  {'Method':<12}  {'MAE':>8}  {'NCC':>8}  {'Time (s)':>10}")
    print("  " + "-" * 42)
    for m in args.methods:
        r = metrics[m]
        print(f"  {m.upper():<12}  {r['mae']:>8.2f}  {r['ncc']:>8.4f}  {r['time_s']:>10.1f}")
    print(border + "\n")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = args.out_dir / "comparison_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "MAE", "NCC", "Time_s"])
        for m in args.methods:
            r = metrics[m]
            w.writerow([m.upper(), f"{r['mae']:.4f}", f"{r['ncc']:.6f}",
                        f"{r['time_s']:.1f}"])
    logger.info("CSV results saved → %s", csv_path)

    # ── Figures ───────────────────────────────────────────────────────────────
    if len(volumes) > 0:
        plot_ortho_comparison(
            volumes, metrics,
            out_path=args.out_dir / "method_comparison.png",
        )
        plot_metric_bars(
            metrics,
            out_path=args.out_dir / "metric_bars.png",
        )

    logger.info("Done. Outputs written to %s", args.out_dir)


if __name__ == "__main__":
    run_comparison()
