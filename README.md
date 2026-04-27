# 3D Freehand Ultrasound Reconstruction

This repository contains a high-performance Python pipeline for reconstructing 3D volumetric medical images from 2D freehand ultrasound sweeps tracked via spatial sensors.

## Features
- **Voxel Nearest Neighbor (VNN)**: High-performance dense reconstruction using SciPy's Exact Distance Transform (EDT), yielding perfectly solid tissue volumes.
- **Probabilistic Neural Network (PNN)**: Classical trilinear interpolation based on inverse distance weighting (IDW) with Gaussian decay.
- **Bicubic Spline (Spline)**: Advanced backward mapping using PyTorch-accelerated bicubic interpolation for superior surface smoothness.
- **Deep Learning (DL)**: Self-supervised 3D U-Net refinement that learns to denoise and complete sparse VNN initializations.
- **Leave-One-Out Cross-Validation (LOOCV)**: An objective, mathematical validation framework (`validate.py`) capable of computing Mean Absolute Error (MAE) and Normalized Cross-Correlation (NCC) against hidden physical frames.
- **Comparison Framework**: Automated benchmarking of all methods (`dl_compare.py`) with side-by-side visualization and metric tables.
- **Visualization**: Built-in volume plotting (Maximum Intensity Projection, Orthogonal Slices, Axial Montages, Coverage maps).
- **Export**: Generates standard medical formats (`.mhd`, `.nii.gz`) compatible with clinical viewing software like 3D Slicer.

## Setup & Requirements

1. Install dependencies:
   ```bash
   pip install numpy scipy matplotlib simpleitk torch
   ```

2. **Data Structure**:
   By default, the pipeline expects data in a `tmp/dataset_3/tracked_probe_data` directory. 
   *(Note: Datasets are excluded from this repository via `.gitignore` due to size).*

## Usage

**1. Run 3D Reconstruction (any method):**
```bash
python pipeline.py --method dl --dl-epochs 20
```
*(Methods: `vnn`, `pnn`, `spline`, `dl`)*

**2. Multi-Method Benchmarking:**
Compare VNN, PNN, and DL back-to-back:
```bash
python dl_compare.py --epochs 20
```
This generates a comparison table and side-by-side figures in `tmp/output/`.

**3. Validate Reconstruction Accuracy:**
```bash
python validate.py --method dl
```

## Architecture

* `core/vnn.py` - Core logic for VNN dense hole-filling.
* `core/pnn.py` - Core logic for PNN trilinear interpolation.
* `core/dl.py` - 3D U-Net architecture and self-supervised training logic.
* `core/spline.py` - Bicubic spline backward mapping.
* `pipeline.py` - The main CLI runner orchestrating data loading, reconstruction, and visualization.
* `dl_compare.py` - Benchmarking script for comparing all reconstruction algorithms.
* `validate.py` - LOOCV framework for algorithm tuning.
* `config.py` - Central configuration definitions.
