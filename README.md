# 3D Freehand Ultrasound Reconstruction

This repository contains a high-performance Python pipeline for reconstructing 3D volumetric medical images from 2D freehand ultrasound sweeps tracked via spatial sensors.

## Features
- **Voxel Nearest Neighbor (VNN)**: High-performance dense reconstruction using SciPy's Exact Distance Transform (EDT), yielding perfectly solid tissue volumes.
- **Probabilistic Neural Network (PNN)**: Classical trilinear interpolation based on inverse distance weighting (IDW) with Gaussian decay.
- **Leave-One-Out Cross-Validation (LOOCV)**: An objective, mathematical validation framework (`validate.py`) capable of computing Mean Absolute Error (MAE) and Normalized Cross-Correlation (NCC) against hidden physical frames.
- **Visualization**: Built-in volume plotting (Maximum Intensity Projection, Orthogonal Slices, Axial Montages, Coverage maps).
- **Export**: Generates standard medical formats (`.mhd`, `.nii.gz`) compatible with clinical viewing software like 3D Slicer.

## Setup & Requirements

1. Install dependencies:
   ```bash
   pip install numpy scipy matplotlib simpleitk
   ```

2. **Data Structure**:
   By default, the pipeline expects data in a `tmp/dataset_3/tracked_probe_data` directory. 
   *(Note: Datasets are excluded from this repository via `.gitignore` due to size).*

## Usage

**1. Run 3D Reconstruction (VNN):**
```bash
python pipeline.py --method vnn
```

**2. Validate Reconstruction Accuracy:**
```bash
python validate.py --method vnn
```

**3. Phantom Calibration Test:**
```bash
python pipeline.py --method vnn --phantom
```

## Architecture

* `core/vnn.py` - Core logic for VNN dense hole-filling.
* `core/pnn.py` - Core logic for PNN trilinear interpolation.
* `pipeline.py` - The main CLI runner orchestrating data loading, reconstruction, and visualization.
* `validate.py` - LOOCV framework for algorithm tuning.
* `config.py` - Central configuration definitions.
