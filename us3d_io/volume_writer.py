"""
io/volume_writer.py
===================
Exports the reconstructed float32 volume to:
  • MetaImage  (.mhd + .raw)  – 3D Slicer, ITK, PLUS
  • NIfTI-1    (.nii.gz)      – FSL, SPM, nibabel

The origin (tracker-space mm of voxel 0,0,0) is encoded into both formats
so that opening the files in a viewer preserves the physical coordinate system.
"""

from __future__ import annotations

import gzip
import io
import logging
import struct
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _to_uint8(volume: np.ndarray) -> np.ndarray:
    """
    Normalise a float32 volume to [0, 255] using its actual min/max so the
    full uint8 dynamic range is used regardless of the raw intensity scale.
    """
    v_min = float(volume.min())
    v_max = float(volume.max())
    if v_max <= v_min:
        return np.zeros_like(volume, dtype=np.uint8)
    scaled = (volume - v_min) * (255.0 / (v_max - v_min))
    return np.clip(scaled, 0, 255).astype(np.uint8)


# ── MetaImage ─────────────────────────────────────────────────────────────────

def write_mhd(
    volume:     np.ndarray,
    output_dir: Path | str,
    stem:       str,
    origin:     np.ndarray,
    voxel_size: float,
) -> Path:
    """
    Write (NZ, NY, NX) float32 volume as MetaImage (.mhd + .raw).

    The 3D Slicer convention is DimSize = NX NY NZ with C-order storage.
    """
    output_dir = Path(output_dir)
    mhd_path   = output_dir / f"{stem}.mhd"
    raw_path   = output_dir / f"{stem}.raw"

    nz, ny, nx = volume.shape
    vol_u8     = _to_uint8(volume)
    vol_u8.tofile(raw_path)

    spacing = f"{voxel_size:.6f} {voxel_size:.6f} {voxel_size:.6f}"
    offset  = " ".join(f"{o:.6f}" for o in origin)
    lines   = [
        "ObjectType = Image",
        "NDims = 3",
        "BinaryData = True",
        "BinaryDataByteOrderMSB = False",
        "CompressedData = False",
        "TransformMatrix = 1 0 0 0 1 0 0 0 1",
        f"Offset = {offset}",
        "CenterOfRotation = 0 0 0",
        "AnatomicalOrientation = RAI",
        f"ElementSpacing = {spacing}",
        f"DimSize = {nx} {ny} {nz}",
        "ElementNumberOfChannels = 1",
        "ElementType = MET_UCHAR",
        f"ElementDataFile = {raw_path.name}",
    ]
    mhd_path.write_text("\n".join(lines) + "\n")
    logger.info("MHD: %s  (%.1f MB)", mhd_path.name, vol_u8.nbytes / 1e6)
    return mhd_path


# ── NIfTI-1 ───────────────────────────────────────────────────────────────────

def write_nifti(
    volume:     np.ndarray,
    output_dir: Path | str,
    stem:       str,
    origin:     np.ndarray,
    voxel_size: float,
) -> Path:
    """Write (NZ, NY, NX) float32 volume as compressed NIfTI-1 (.nii.gz)."""
    output_dir = Path(output_dir)
    nii_path   = output_dir / f"{stem}.nii.gz"

    try:
        import nibabel as nib
        vol_u8 = _to_uint8(volume)
        affine = np.diag([voxel_size, voxel_size, voxel_size, 1.0])
        affine[:3, 3] = origin
        img = nib.Nifti1Image(vol_u8, affine)
        img.header.set_xyzt_units("mm")
        nib.save(img, str(nii_path))
        logger.info("NIfTI (nibabel): %s", nii_path.name)

    except ImportError:
        logger.warning("nibabel not available – writing minimal NIfTI-1 header manually")
        _write_nifti_manual(volume, nii_path, origin, voxel_size)

    return nii_path


def _write_nifti_manual(
    volume:     np.ndarray,
    path:       Path,
    origin:     np.ndarray,
    voxel_size: float,
) -> None:
    """Minimal NIfTI-1 writer that requires no external libraries."""
    nz, ny, nx = volume.shape
    vol_u8     = _to_uint8(volume)

    hdr = bytearray(348)
    def pk(fmt, off, *v): struct.pack_into(fmt, hdr, off, *v)

    pk("<i",  0,   348)           # sizeof_hdr
    pk("<h",  40,  3)             # ndim
    pk("<h",  42,  nx)            # dim[1]
    pk("<h",  44,  ny)            # dim[2]
    pk("<h",  46,  nz)            # dim[3]
    pk("<h",  70,  2)             # datatype = uint8
    pk("<h",  72,  8)             # bitpix
    pk("<f",  80,  voxel_size)    # pixdim[1]
    pk("<f",  84,  voxel_size)    # pixdim[2]
    pk("<f",  88,  voxel_size)    # pixdim[3]
    pk("<f", 108,  352.0)         # vox_offset
    pk("<f", 112,  1.0)           # scl_slope
    pk("<h", 252,  1)             # sform_code = scanner
    pk("<fff", 280, voxel_size, 0.0, 0.0)
    pk("<fff", 292, 0.0, voxel_size, 0.0)
    pk("<fff", 304, 0.0, 0.0, voxel_size)
    pk("<fff", 316, *origin[:3].tolist())   # srow offsets
    hdr[344:348] = b"n+1\x00"

    buf = io.BytesIO()
    buf.write(bytes(hdr))
    buf.write(b"\x00\x00\x00\x00")   # no extension
    buf.write(vol_u8.tobytes())
    buf.seek(0)

    with gzip.open(path, "wb") as gz:
        gz.write(buf.read())

    logger.info("NIfTI (manual): %s", path.name)
