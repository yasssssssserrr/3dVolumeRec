"""
tests/run_tests.py
==================
Complete test suite for the final calibration-aware pipeline.
Run with:  python tests/run_tests.py
"""

import sys, unittest, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from us3d_io.calibration_reader import load_calibration, _parse_all_matrices
from core.geometry import (
    make_image_points, transform_to_tracker, effective_transform,
    build_voxel_grid, tracker_to_voxel,
)
from us3d_io.mhd_reader  import SequenceData, FrameData
from core.pnn       import PNNReconstructor
from config         import PipelineConfig

CALIB_TEXT = """
Averaged Calibration matrix of 13 calibrations
 -0.9372349829593   0.3512557135458  -0.0819525252683   215.136514579976
 -0.1879419702486  -0.6825645724011  -0.6884870238461    -4.325898744288
 -0.3033577938485  -0.6346362914727   0.6978688330490    -0.3370001472492
  0                 0                 0                  1

Interpolated Calibration matrix of 13 calibrations
 -0.8558302911452   0.4769784363453  -0.1069906573362   207.7339081643368
 -0.2941287028958  -0.6729739651303  -0.6667265546581     4.2462900767277
 -0.3900161164101  -0.5391357581231   0.7375830126545    19.6995075731916
  0                 0                 0                  1
"""


def _synth_seq(n_frames=5, h=20, w=20, ps=1.0, fill=128, z_step=1.0):
    seq = SequenceData(path=Path("synth.mhd"), dims=[w, h, n_frames], pixel_spacing_mm=ps)
    for i in range(n_frames):
        T = np.eye(4, dtype=np.float64); T[2, 3] = i * z_step
        img = np.full((h, w), fill, dtype=np.uint8)
        seq.frames.append(FrameData(i, float(i), img, T, ps))
    return seq


def _cfg(tmp):
    cfg = PipelineConfig.__new__(PipelineConfig)
    cfg.voxel_size_mm     = 1.0
    cfg.downsample_factor = 1
    cfg.hole_fill_radius  = 1
    cfg.intensity_offset  = 0.03
    cfg.output_dir        = Path(tmp)
    cfg.export_mhd        = False
    cfg.export_nifti      = False
    cfg.export_figures    = False
    cfg.output_stem       = "test"
    return cfg


# ── Calibration reader ────────────────────────────────────────────────────────

class TestCalibrationReader(unittest.TestCase):

    def test_parse_both_matrices(self):
        mats = _parse_all_matrices(CALIB_TEXT)
        self.assertIn("averaged",     mats)
        self.assertIn("interpolated", mats)

    def test_matrix_shape(self):
        mats = _parse_all_matrices(CALIB_TEXT)
        self.assertEqual(mats["averaged"].shape,     (4, 4))
        self.assertEqual(mats["interpolated"].shape, (4, 4))

    def test_bottom_row_identity(self):
        mats = _parse_all_matrices(CALIB_TEXT)
        for key in ("averaged", "interpolated"):
            np.testing.assert_allclose(mats[key][3], [0, 0, 0, 1], atol=1e-9)

    def test_translation_values(self):
        mats = _parse_all_matrices(CALIB_TEXT)
        t = mats["averaged"][:3, 3]
        np.testing.assert_allclose(t[0], 215.136514579976, rtol=1e-6)

    def test_load_from_file(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
            fh.write(CALIB_TEXT)
            tmp = fh.name
        T = load_calibration(tmp, "averaged")
        self.assertEqual(T.shape, (4, 4))
        os.unlink(tmp)

    def test_det_averaged_close_to_one(self):
        mats = _parse_all_matrices(CALIB_TEXT)
        det = abs(np.linalg.det(mats["averaged"][:3, :3]))
        self.assertAlmostEqual(det, 1.0, delta=0.05)


# ── Geometry ──────────────────────────────────────────────────────────────────

class TestGeometry(unittest.TestCase):

    def test_image_points_shape(self):
        pts = make_image_points(10, 8, 0.5, downsample=1)
        self.assertEqual(pts.shape, (4, 80))

    def test_downsample_reduces_columns(self):
        pts = make_image_points(10, 8, 0.5, downsample=2)
        self.assertEqual(pts.shape, (4, 40))

    def test_homogeneous_row_all_ones(self):
        pts = make_image_points(5, 5, 0.1)
        np.testing.assert_array_equal(pts[3], np.ones(pts.shape[1]))

    def test_z_row_all_zeros(self):
        pts = make_image_points(5, 5, 0.1)
        np.testing.assert_array_equal(pts[2], np.zeros(pts.shape[1]))

    def test_pixel_spacing_applied_correctly(self):
        ps  = 0.0884652
        pts = make_image_points(1, 3, ps, downsample=1)
        np.testing.assert_allclose(pts[0], [0, ps, 2 * ps], atol=1e-9)

    def test_effective_transform_identity_calibration(self):
        T_frame = np.eye(4); T_frame[0, 3] = 5.0
        T_eff   = effective_transform(T_frame, None)
        np.testing.assert_array_equal(T_eff, T_frame)

    def test_effective_transform_with_calibration(self):
        mats    = _parse_all_matrices(CALIB_TEXT)
        T_calib = mats["averaged"]
        T_frame = np.eye(4)
        T_eff   = effective_transform(T_frame, T_calib)
        np.testing.assert_allclose(T_eff, T_calib, atol=1e-12)

    def test_transform_to_tracker_identity(self):
        pts = make_image_points(3, 3, 1.0, downsample=1)
        res = transform_to_tracker(pts, np.eye(4))
        np.testing.assert_allclose(res, pts[:3], atol=1e-9)

    def test_transform_translation_only(self):
        T = np.eye(4); T[:3, 3] = [10, 20, 30]
        pts = np.array([[0.], [0.], [0.], [1.]])
        res = transform_to_tracker(pts, T)
        np.testing.assert_allclose(res, [[10.], [20.], [30.]], atol=1e-9)

    def test_voxel_grid_origin_equals_bbmin(self):
        bb_min = np.array([5., -3., 100.])
        bb_max = np.array([10., 0., 110.])
        origin, _ = build_voxel_grid(bb_min, bb_max, 0.5)
        np.testing.assert_allclose(origin, bb_min)

    def test_voxel_index_at_origin(self):
        origin = np.zeros(3)
        pts    = np.zeros((3, 1))
        idx    = tracker_to_voxel(pts, origin, 1.0)
        np.testing.assert_array_equal(idx, [[0], [0], [0]])

    def test_voxel_index_with_offset(self):
        origin = np.array([10., 0., 0.])
        pts    = np.array([[11.], [0.], [0.]])
        idx    = tracker_to_voxel(pts, origin, 1.0)
        self.assertEqual(idx[0, 0], 1)


# ── PNN reconstructor ─────────────────────────────────────────────────────────

class TestPNNReconstructor(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_reconstruct_runs(self):
        rec = PNNReconstructor(_cfg(self.tmp))
        vol = rec.reconstruct([_synth_seq()])
        self.assertEqual(vol.ndim, 3)

    def test_volume_not_all_zero(self):
        rec = PNNReconstructor(_cfg(self.tmp))
        vol = rec.reconstruct([_synth_seq()])
        self.assertGreater(vol.max(), 0)

    def test_values_in_valid_range(self):
        rec = PNNReconstructor(_cfg(self.tmp))
        vol = rec.reconstruct([_synth_seq()])
        self.assertGreaterEqual(float(vol.min()), 0.0)
        self.assertLessEqual(float(vol.max()), 255.0)

    def test_uniform_image_intensity_preserved(self):
        seq = _synth_seq(n_frames=4, h=5, w=5, fill=200)
        rec = PNNReconstructor(_cfg(self.tmp))
        vol = rec.reconstruct([seq])
        filled = vol[rec.mask]
        self.assertAlmostEqual(float(filled.mean()), 200.0, delta=5.0)

    def test_mask_not_empty(self):
        rec = PNNReconstructor(_cfg(self.tmp))
        rec.reconstruct([_synth_seq(n_frames=10, h=10, w=10)])
        self.assertGreater(int(rec.mask.sum()), 0)

    def test_with_averaged_calibration(self):
        mats = _parse_all_matrices(CALIB_TEXT)
        T_calib = mats["averaged"]
        rec = PNNReconstructor(_cfg(self.tmp), image_to_probe=T_calib)
        vol = rec.reconstruct([_synth_seq(n_frames=5, h=10, w=10)])
        self.assertGreater(vol.max(), 0)

    def test_multiple_sequences(self):
        seqs = [_synth_seq(n_frames=3, h=8, w=8, z_step=i+1) for i in range(3)]
        rec  = PNNReconstructor(_cfg(self.tmp))
        vol  = rec.reconstruct(seqs)
        self.assertGreater(vol.max(), 0)


# ── I/O ───────────────────────────────────────────────────────────────────────

class TestVolumeWriter(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_mhd_files_exist(self):
        from us3d_io.volume_writer import write_mhd
        vol = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        write_mhd(vol, self.tmp, "test", np.zeros(3), 1.0)
        self.assertTrue((self.tmp / "test.mhd").exists())
        self.assertTrue((self.tmp / "test.raw").exists())

    def test_mhd_roundtrip_values(self):
        from us3d_io.volume_writer import write_mhd
        vol = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        write_mhd(vol, self.tmp, "rt", np.zeros(3), 1.0)
        loaded = np.fromfile(self.tmp / "rt.raw", dtype=np.uint8).reshape(2, 3, 4)
        np.testing.assert_array_equal(loaded, vol.astype(np.uint8))

    def test_nifti_created_and_nonempty(self):
        from us3d_io.volume_writer import write_nifti
        vol = np.zeros((4, 4, 4), dtype=np.float32)
        p   = write_nifti(vol, self.tmp, "nii", np.zeros(3), 0.5)
        self.assertTrue(p.exists())
        self.assertGreater(p.stat().st_size, 0)

    def test_mhd_header_contains_dimsize(self):
        from us3d_io.volume_writer import write_mhd
        vol = np.zeros((5, 6, 7), dtype=np.float32)
        p   = write_mhd(vol, self.tmp, "hdr", np.zeros(3), 0.5)
        content = p.read_text()
        self.assertIn("DimSize = 7 6 5", content)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestCalibrationReader, TestGeometry,
        TestPNNReconstructor,  TestVolumeWriter,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
