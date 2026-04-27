"""
io/calibration_reader.py
========================
Parses the external spatial calibration file produced by the PLUS N-wire
phantom calibration workflow.

The calibration matrix T_ImageToProbe maps from image-plane coordinates
(in mm: u*ps, v*ps, 0) to the physical probe coordinate frame (where the
NDI optical marker is attached).

Full transform chain
--------------------
  p_image_mm  = [u * pixel_spacing, v * pixel_spacing, 0, 1]^T
  p_probe     = T_ImageToProbe  @ p_image_mm
  p_tracker   = T_ProbeToTracker @ p_probe
  ⟹
  p_tracker   = (T_ProbeToTracker @ T_ImageToProbe) @ p_image_mm

Both an "averaged" and an "interpolated" matrix are provided from 13
calibration runs.  The averaged matrix has det(R) ≈ 0.983 (closer to the
ideal 1.0 than the interpolated value of ≈ 0.971) and is therefore the
recommended choice.

File format (plain-text, space-separated 4×4 matrices)
-----------
  Averaged Calibration matrix of 13 calibrations
  <4 rows of 4 space-separated floats>

  Interpolated Calibration matrix of 13 calibrations
  <4 rows of 4 space-separated floats>
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MatrixType = Literal["averaged", "interpolated"]


def load_calibration(
    path: Path | str,
    matrix_type: MatrixType = "averaged",
) -> np.ndarray:
    """
    Load a 4×4 spatial calibration matrix from the PLUS calibration text file.

    Parameters
    ----------
    path        : path to Calibration_matrices.txt
    matrix_type : "averaged" (default, det≈0.983) or "interpolated" (det≈0.971)

    Returns
    -------
    T : (4, 4) float64 ndarray – Image-to-Probe homogeneous transform
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()

    matrices = _parse_all_matrices(text)

    key = matrix_type.lower()
    if key not in matrices:
        available = list(matrices.keys())
        raise KeyError(
            f"Matrix type '{matrix_type}' not found in {path.name}. "
            f"Available: {available}"
        )

    T = matrices[key]
    _validate_calibration_matrix(T, matrix_type)
    logger.info(
        "Loaded %s calibration matrix  |  det(R)=%.6f  |  translation=%s mm",
        matrix_type,
        np.linalg.det(T[:3, :3]),
        T[:3, 3].round(2),
    )
    return T


def load_both_calibrations(
    path: Path | str,
) -> Dict[MatrixType, np.ndarray]:
    """Return both calibration matrices keyed by type."""
    path = Path(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    return _parse_all_matrices(text)


# ── Private helpers ────────────────────────────────────────────────────────────

def _parse_all_matrices(text: str) -> Dict[str, np.ndarray]:
    """
    Extract all 4×4 matrices from the calibration text.

    The file structure is:
      <label line containing "Averaged" or "Interpolated">
      <4 rows of 4 floats>
      <blank line or next label>
    """
    matrices: Dict[str, np.ndarray] = {}

    # Split on label lines; detect "Averaged" and "Interpolated"
    blocks = re.split(r'\n(?=\s*(?:Averaged|Interpolated))', text.strip(), flags=re.IGNORECASE)

    for block in blocks:
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue

        # Identify matrix type from label line
        label_line = lines[0].lower()
        if "averaged" in label_line:
            key = "averaged"
        elif "interpolated" in label_line:
            key = "interpolated"
        else:
            continue

        # Extract numeric rows (lines that contain ≥4 numbers)
        numeric_rows = []
        for ln in lines[1:]:
            nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', ln)
            if len(nums) >= 4:
                numeric_rows.append([float(n) for n in nums[:4]])

        if len(numeric_rows) < 4:
            logger.warning("Block '%s': expected 4 numeric rows, found %d", key, len(numeric_rows))
            continue

        T = np.array(numeric_rows[:4], dtype=np.float64)
        matrices[key] = T

    return matrices


def _validate_calibration_matrix(T: np.ndarray, label: str) -> None:
    """Sanity-check a calibration matrix and log warnings if suspect."""
    if T.shape != (4, 4):
        raise ValueError(f"Calibration matrix must be 4×4, got {T.shape}")

    # Bottom row must be [0, 0, 0, 1]
    bottom_ok = np.allclose(T[3], [0, 0, 0, 1], atol=1e-6)
    if not bottom_ok:
        logger.warning("[%s] Bottom row %s is not [0,0,0,1]", label, T[3])

    # Rotation part should be close to orthogonal
    R = T[:3, :3]
    det = np.linalg.det(R)
    if abs(abs(det) - 1.0) > 0.05:
        logger.warning("[%s] det(R) = %.4f – rotation matrix may be poorly conditioned", label, det)

    # Translation should be physically plausible (< 500 mm)
    t = T[:3, 3]
    if np.linalg.norm(t) > 500:
        logger.warning("[%s] Translation norm = %.1f mm – seems large", label, np.linalg.norm(t))

    logger.debug("[%s] det(R)=%.6f  t=%s", label, det, t.round(2))
