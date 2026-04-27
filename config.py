"""
config.py – Pipeline configuration (v3: trilinear interp, IDW weights, no bias)

Trial 7 (L123_trial_7) is EXCLUDED from reconstruction:
  - Its image-plane centroid is ~70 mm from the forearm trials in tracker space
  - It was recorded in a different session / body position
  - Mixing it with trials 1-6 creates a geometrically disconnected volume
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Tuple

_HERE = Path(__file__).resolve().parent


@dataclass
class DataSource:
    trial_id: int
    mhd_path: Path
    label: str = ""
    def __post_init__(self):
        self.mhd_path = Path(self.mhd_path)
        if not self.label:
            self.label = self.mhd_path.stem


@dataclass
class PipelineConfig:
    method:            Literal["pnn", "vnn", "spline", "dl"] = "spline"
    base_dir:          Path  = field(default_factory=lambda: _HERE / "tmp" / "dataset_3")
    calibration_file:  Path  = field(default_factory=lambda: _HERE / "tmp" / "dataset_3" / "Calibration_matrices (1).txt")
    calibration_type:  Literal["averaged","interpolated"] = "averaged"
    apply_calibration: bool  = True
    pixel_spacing_mm:  float = 0.0884652
    voxel_size_mm:     float = 0.5
    hole_fill_radius:  int   = 10          # more passes → better gap closure
    downsample_factor: int   = 4
    bb_margin_mm:      float = 5.0         # bounding-box padding (mm)
    gaussian_sigma_mm: float = 1.0         # physical σ for IDW Gaussian weights
    use_trilinear:     bool  = True        # trilinear scatter (vs nearest-voxel)
    vnn_max_radius_mm: float = 5.0         # max fill distance for VNN (mm)
    use_phantom_data:  bool  = False       # if True, load phantom wire scans
    # ── Deep Learning hyper-parameters ────────────────────────────────────────
    dl_epochs:         int   = 20           # U-Net training epochs
    dl_lr:             float = 1e-3         # Adam learning rate
    dl_base_features:  int   = 16           # U-Net base channel width
    dl_skip_training:  bool  = False        # reuse checkpoint if it exists
    output_dir:        Path  = field(default_factory=lambda: _HERE / "tmp" / "output")
    output_stem:       str   = "recon_pnn_calibrated"
    export_mhd:        bool  = True
    export_nifti:      bool  = True
    export_figures:    bool  = True
    gaussian_sigma:    float = 1.5         # post-process smoothing (voxels)
    dpi:               int   = 150
    colormap:          str   = "gray"
    sources: Tuple[DataSource,...] = field(default_factory=tuple)

    def __post_init__(self):
        self.base_dir         = Path(self.base_dir)
        self.calibration_file = Path(self.calibration_file)
        self.output_dir       = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.sources:
            self.sources = self._build_sources()

    def _build_sources(self) -> Tuple[DataSource,...]:
        srcs = []
        if self.use_phantom_data:
            for tid in (1, 2, 3, 4, 5, 6, 7):
                p = self.base_dir / f"L123_trial_{tid}.mhd"
                if p.exists():
                    srcs.append(DataSource(tid, p, f"phantom_trial_{tid}"))
        else:
            tracked = self.base_dir / "tracked_probe_data"
            for tid in (1, 2, 3, 4, 5, 6):
                p = tracked / f"forearm_tracked_L123_trial_{tid}.mhd"
                if p.exists():
                    srcs.append(DataSource(tid, p, f"tracked_trial_{tid}"))
        return tuple(srcs)
