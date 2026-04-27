"""
io/mhd_reader.py
================
Reads PLUS-format .mhd / .raw sequence files.

Each frame stores:
  - Seq_FrameNNNN_ProbeToTrackerTransform  : 4×4 row-major float matrix
  - Seq_FrameNNNN_ProbeToTrackerTransformStatus : "OK" | "MISSING"
  - Seq_FrameNNNN_ElementSpacing            : pixel spacing in mm
  - Seq_FrameNNNN_Timestamp                 : acquisition time in seconds
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "MET_UCHAR":  np.uint8,
    "MET_CHAR":   np.int8,
    "MET_USHORT": np.uint16,
    "MET_SHORT":  np.int16,
    "MET_FLOAT":  np.float32,
    "MET_DOUBLE": np.float64,
}

_FRAME_RE = re.compile(r'^Seq_Frame(\d+)_(\w+)\s*=\s*(.+)$')
_GLOBAL_RE = re.compile(r'^(\w+)\s*=\s*(.+)$')


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class FrameData:
    frame_id:         int
    timestamp:        float
    image:            np.ndarray   # (height, width) uint8
    probe_to_tracker: np.ndarray   # (4, 4) float64
    pixel_spacing_mm: float


@dataclass
class SequenceData:
    path:             Path
    dims:             List[int]    # [width, height, n_total]
    pixel_spacing_mm: float
    label:            str = ""
    frames:           List[FrameData] = field(default_factory=list)

    @property
    def n_frames(self)      -> int: return len(self.frames)
    @property
    def image_width(self)   -> int: return self.dims[0]
    @property
    def image_height(self)  -> int: return self.dims[1]
    @property
    def n_total(self)       -> int: return self.dims[2]


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_sequence(
    mhd_path: Path | str,
    label: str = "",
) -> SequenceData:
    """
    Load a PLUS .mhd sequence and its binary .raw pixel data.

    Returns
    -------
    SequenceData  with only OK-tracked frames populated in .frames
    """
    mhd_path = Path(mhd_path)
    if not mhd_path.exists():
        raise FileNotFoundError(f"MHD not found: {mhd_path}")

    with open(mhd_path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()

    # ── Global header ──────────────────────────────────────────────────────────
    global_hdr: Dict[str, str] = {}
    for ln in lines:
        m = _GLOBAL_RE.match(ln.strip())
        if m and not m.group(1).startswith("Seq_Frame"):
            global_hdr[m.group(1)] = m.group(2).strip()

    dims_str = global_hdr.get("DimSize", "")
    dims = [int(x) for x in dims_str.split() if x.strip()]
    if len(dims) < 3:
        raise ValueError(f"DimSize malformed: {dims_str!r}")

    dtype = DTYPE_MAP.get(global_hdr.get("ElementType", "MET_UCHAR"), np.uint8)

    raw_filename = global_hdr.get("ElementDataFile", "")
    raw_path = mhd_path.parent / raw_filename
    if not raw_path.exists():
        raise FileNotFoundError(f"RAW not found: {raw_path}")

    default_ps = 0.0884652
    sp_str = global_hdr.get("ElementSpacing", "1 1 1")
    try:
        sp_vals = [float(x) for x in sp_str.split()]
        if sp_vals and sp_vals[0] > 0:
            default_ps = sp_vals[0]
    except ValueError:
        pass

    width, height, n_total = dims[0], dims[1], dims[2]
    frame_px = width * height

    logger.info("%-40s %d×%d × %d frames", mhd_path.name, width, height, n_total)

    # ── Per-frame metadata ─────────────────────────────────────────────────────
    frame_meta: Dict[int, Dict[str, str]] = {}
    for ln in lines:
        m = _FRAME_RE.match(ln.strip())
        if m:
            fid, key, val = int(m.group(1)), m.group(2), m.group(3).strip()
            if fid not in frame_meta:
                frame_meta[fid] = {}
            frame_meta[fid][key] = val

    # ── Memory-map binary data ─────────────────────────────────────────────────
    raw = np.memmap(raw_path, dtype=dtype, mode="r")
    if raw.size < frame_px * n_total:
        raise ValueError(
            f"RAW too small: need {frame_px*n_total} bytes, got {raw.size}"
        )

    seq = SequenceData(
        path=mhd_path,
        dims=dims,
        pixel_spacing_mm=default_ps,
        label=label or mhd_path.stem,
    )

    n_ok = n_miss = n_err = 0
    for fid in sorted(frame_meta.keys()):
        fd = frame_meta[fid]

        if fd.get("ProbeToTrackerTransformStatus") != "OK":
            n_miss += 1
            continue

        tf_str = fd.get("ProbeToTrackerTransform", "")
        T = _parse_4x4(tf_str)
        if T is None:
            n_err += 1
            continue

        ps = default_ps
        sp = fd.get("ElementSpacing", "")
        if sp:
            try:
                sv = [float(x) for x in sp.split()]
                if sv and sv[0] > 0:
                    ps = sv[0]
            except ValueError:
                pass

        ts = float(fd.get("Timestamp", 0.0))

        offset = fid * frame_px
        if offset + frame_px > raw.size:
            n_err += 1
            continue

        img = raw[offset: offset + frame_px].reshape(height, width).copy()
        seq.frames.append(FrameData(fid, ts, img, T, ps))
        n_ok += 1

    logger.info(
        "  → %d valid | %d missing | %d errors   (%.1f%% OK)",
        n_ok, n_miss, n_err, 100 * n_ok / max(n_ok + n_miss + n_err, 1),
    )
    return seq


def _parse_4x4(s: str) -> Optional[np.ndarray]:
    try:
        nums = [float(x) for x in s.split()]
        return np.array(nums, dtype=np.float64).reshape(4, 4) if len(nums) == 16 else None
    except ValueError:
        return None
