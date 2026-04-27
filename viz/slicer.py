"""
viz/slicer.py v2 – Smart centroid-centred slices + Maximum Intensity Projections
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)


def _clim(arr: np.ndarray, lo=2.0, hi=98.0):
    nz = arr[arr > 0]
    return (0.0, 255.0) if nz.size == 0 else \
           (float(np.percentile(nz, lo)), float(np.percentile(nz, hi)))


def _save(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    logger.info("Figure: %s", path.name)


def _style(ax):
    ax.set_facecolor("black")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.tick_params(colors="white", labelsize=7)


def _tissue_centroid(mask: np.ndarray) -> Tuple[int,int,int]:
    """Return (cz,cy,cx) at the median position of all filled voxels."""
    iz,iy,ix = np.where(mask)
    if len(iz) == 0:
        nz,ny,nx = mask.shape
        return nz//2, ny//2, nx//2
    return int(np.median(iz)), int(np.median(iy)), int(np.median(ix))


def _scalebar(ax, vs, mm=10):
    xl, yl = ax.get_xlim(), ax.get_ylim()
    x0 = xl[0] + (xl[1]-xl[0])*0.05
    y0 = yl[0] + (yl[1]-yl[0])*0.93
    ax.plot([x0, x0+mm], [y0,y0], color="white", lw=2)
    ax.text(x0+mm/2, y0-(yl[1]-yl[0])*0.04, f"{mm} mm",
            color="white", fontsize=7, ha="center")


# ── 1. Orthogonal slices centred on tissue ────────────────────────────────────

def plot_ortho_slices(volume, output_dir, stem, voxel_size,
                      mask=None, calibrated=True, dpi=150, cmap="gray"):
    nz, ny, nx = volume.shape
    if mask is not None:
        cz, cy, cx = _tissue_centroid(mask)
    else:
        cz, cy, cx = nz//2, ny//2, nx//2

    logger.info("Ortho centre voxel: z=%d y=%d x=%d  (mm: %.1f %.1f %.1f)",
                cz, cy, cx, cz*voxel_size, cy*voxel_size, cx*voxel_size)

    axial    = volume[cz, :, :]
    coronal  = volume[:, cy, :]
    sagittal = volume[:, :, cx]

    vmin, vmax = _clim(volume)
    tag = "calibrated" if calibrated else "identity"

    fig, axes = plt.subplots(1, 3, figsize=(15,5), facecolor="black")
    fig.suptitle(
        f"3-D US PNN Reconstruction  ({tag}, voxel={voxel_size:.2f} mm)\n"
        f"Slices centred on tissue centroid  "
        f"z={cz*voxel_size:.0f} mm, y={cy*voxel_size:.0f} mm, x={cx*voxel_size:.0f} mm",
        color="white", fontsize=9,
    )

    panels = [
        (axial,    f"Axial",   "X (mm)","Y (mm)", nx, ny),
        (coronal,  f"Coronal", "X (mm)","Z (mm)", nx, nz),
        (sagittal, f"Sagittal","Y (mm)","Z (mm)", ny, nz),
    ]
    for ax, (img, title, xl, yl, nc, nr) in zip(axes, panels):
        _style(ax)
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper",
                       extent=[0, nc*voxel_size, nr*voxel_size, 0],
                       interpolation="bilinear")
        ax.set_title(title, color="white", fontsize=9, pad=3)
        ax.set_xlabel(xl, color="white", fontsize=8)
        ax.set_ylabel(yl, color="white", fontsize=8)
        _scalebar(ax, voxel_size)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors="white", labelsize=7)

    plt.tight_layout()
    out = output_dir / f"{stem}_ortho.png"
    _save(fig, out, dpi)
    return out


# ── 2. Maximum Intensity Projections ─────────────────────────────────────────

def plot_mip(volume, output_dir, stem, voxel_size, dpi=150, cmap="gray"):
    """Three-axis MIP – shows the 3D tissue extent clearly regardless of slice position."""
    mip_z = volume.max(axis=0)    # XY projection
    mip_y = volume.max(axis=1)    # XZ projection
    mip_x = volume.max(axis=2)    # YZ projection
    nz, ny, nx = volume.shape

    vmin, vmax = _clim(volume)

    fig, axes = plt.subplots(1, 3, figsize=(15,5), facecolor="black")
    fig.suptitle(f"Maximum Intensity Projection (MIP)  –  voxel={voxel_size:.2f} mm",
                 color="white", fontsize=10)

    panels = [
        (mip_z, "MIP axial  (Z projection)",   "X (mm)","Y (mm)", nx, ny),
        (mip_y, "MIP coronal  (Y projection)", "X (mm)","Z (mm)", nx, nz),
        (mip_x, "MIP sagittal  (X projection)","Y (mm)","Z (mm)", ny, nz),
    ]
    for ax, (img, title, xl, yl, nc, nr) in zip(axes, panels):
        _style(ax)
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper",
                       extent=[0, nc*voxel_size, nr*voxel_size, 0],
                       interpolation="bilinear")
        ax.set_title(title, color="white", fontsize=9, pad=3)
        ax.set_xlabel(xl, color="white", fontsize=8)
        ax.set_ylabel(yl, color="white", fontsize=8)
        _scalebar(ax, voxel_size)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors="white", labelsize=7)

    plt.tight_layout()
    out = output_dir / f"{stem}_mip.png"
    _save(fig, out, dpi)
    return out


# ── 3. Axial montage (centred on tissue) ──────────────────────────────────────

def plot_axial_montage(volume, output_dir, stem, voxel_size, mask=None,
                       n_slices=25, dpi=150, cmap="gray"):
    if mask is not None:
        iz,_,_ = np.where(mask)
        z_lo, z_hi = int(iz.min()), int(iz.max())
    else:
        z_lo, z_hi = 0, volume.shape[0]-1

    indices = np.linspace(z_lo, z_hi, n_slices, dtype=int)
    ncols = 5; nrows = int(np.ceil(n_slices/ncols))
    vmin, vmax = _clim(volume)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3), facecolor="black")
    fig.suptitle(f"Axial slice montage  (tissue extent, {n_slices} planes, voxel={voxel_size:.2f} mm)",
                 color="white", fontsize=10)

    for ax, iz in zip(axes.flat, indices):
        _style(ax)
        ax.imshow(volume[iz], cmap=cmap, vmin=vmin, vmax=vmax,
                  origin="upper", interpolation="bilinear")
        ax.set_title(f"z={iz*voxel_size:.1f} mm", color="white", fontsize=7, pad=2)
        ax.axis("off")
    for ax in list(axes.flat)[n_slices:]:
        ax.set_visible(False)

    plt.tight_layout(pad=0.4)
    out = output_dir / f"{stem}_montage.png"
    _save(fig, out, dpi)
    return out


# ── 4. Coverage ───────────────────────────────────────────────────────────────

def plot_coverage(mask, output_dir, stem, voxel_size, n_trials=0, dpi=150):
    nz, ny, nx = mask.shape
    iz,_,_ = np.where(mask)
    z_lo = int(iz.min()) if len(iz) else 0
    z_hi = int(iz.max()) if len(iz) else nz-1

    fill_z  = mask.mean(axis=(1,2))
    proj_xy = mask.sum(axis=0) / nz
    z_mm    = np.arange(nz) * voxel_size

    fig = plt.figure(figsize=(13,5), facecolor="black")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    _style(ax1)
    ax1.fill_between(z_mm, fill_z*100, alpha=0.4, color="#5DCAA5")
    ax1.plot(z_mm, fill_z*100, color="#5DCAA5", lw=1.5)
    ax1.axhline(fill_z.mean()*100, color="orange", lw=1, ls="--",
                label=f"mean {fill_z.mean()*100:.1f}%")
    ax1.axvline(z_lo*voxel_size, color="red",  lw=1, ls=":", alpha=0.7, label="tissue range")
    ax1.axvline(z_hi*voxel_size, color="red",  lw=1, ls=":", alpha=0.7)
    ax1.set_xlabel("Depth Z (mm)", color="white", fontsize=9)
    ax1.set_ylabel("Fill (%)", color="white", fontsize=9)
    ax1.set_ylim(0,105)
    title = f"Fill ratio per Z-plane" + (f"  ({n_trials} trials)" if n_trials else "")
    ax1.set_title(title, color="white", fontsize=9)
    ax1.legend(fontsize=7, labelcolor="white", facecolor="black")

    ax2 = fig.add_subplot(gs[1])
    _style(ax2)
    im = ax2.imshow(proj_xy, cmap="viridis", origin="upper",
                    extent=[0,nx*voxel_size, ny*voxel_size, 0])
    ax2.set_title("Fill projection X–Y", color="white", fontsize=9)
    ax2.set_xlabel("X (mm)", color="white", fontsize=9)
    ax2.set_ylabel("Y (mm)", color="white", fontsize=9)
    cb = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label("Fraction filled", color="white", fontsize=8)
    cb.ax.tick_params(colors="white", labelsize=7)

    out = output_dir / f"{stem}_coverage.png"
    _save(fig, out, dpi)
    return out


# ── 5. Calibration comparison ─────────────────────────────────────────────────

def plot_calibration_comparison(vol_calib, vol_nocal, output_dir, stem, voxel_size,
                                mask_calib=None, dpi=150, cmap="gray"):
    if mask_calib is not None:
        cz,cy,cx = _tissue_centroid(mask_calib)
    else:
        nz,ny,nx = vol_calib.shape; cz,cy,cx = nz//2, ny//2, nx//2

    vmin_c, vmax_c = _clim(vol_calib)
    vmin_n, vmax_n = _clim(vol_nocal)
    nz_n, ny_n, nx_n = vol_nocal.shape

    fig, axes = plt.subplots(2, 3, figsize=(15,10), facecolor="black")
    fig.suptitle("Calibration effect: with T_ImageToProbe  vs  Identity",
                 color="white", fontsize=11)

    def _row(row_axes, vol, cz_, cy_, cx_, vmin, vmax, tag):
        nz_, ny_, nx_ = vol.shape
        panels = [
            (vol[cz_,:,:], "Axial",    nx_, ny_),
            (vol[:,cy_,:], "Coronal",  nx_, nz_),
            (vol[:,:,cx_], "Sagittal", ny_, nz_),
        ]
        for ax,(img,title,nc,nr) in zip(row_axes, panels):
            _style(ax)
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper",
                           extent=[0,nc*voxel_size,nr*voxel_size,0],
                           interpolation="bilinear")
            ax.set_title(f"{title}  [{tag}]", color="white", fontsize=8, pad=2)
            ax.set_xlabel("mm", color="white", fontsize=7)
            ax.set_ylabel("mm", color="white", fontsize=7)
            _scalebar(ax, voxel_size)

    _row(axes[0], vol_calib, cz,           cy,           cx,
         vmin_c, vmax_c, "calibrated")
    cz_n,cy_n,cx_n = _tissue_centroid(vol_nocal > 0) if mask_calib is None else \
                     (nz_n//2, ny_n//2, nx_n//2)
    _row(axes[1], vol_nocal, cz_n, cy_n, cx_n, vmin_n, vmax_n, "identity")

    plt.tight_layout()
    out = output_dir / f"{stem}_calib_compare.png"
    _save(fig, out, dpi)
    return out
