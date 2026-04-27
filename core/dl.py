"""
core/dl.py – Deep Learning Reconstructor for 3-D Freehand Ultrasound
=====================================================================

Architecture: Lightweight 3-D U-Net
  Input : 2-channel volume  [VNN intensity (0-1), filled-mask (0/1)]
  Output: 1-channel refined intensity volume (sigmoid × 255)

Training strategy: self-supervised Leave-One-Trial-Out
  • Hold out 1 trial, VNN-reconstruct with the other N-1  → input
  • VNN-reconstruct with all N trials                     → target
  • Loss = MSE + 0.1 × (1 – SSIM3D)

Inference (DLReconstructor.reconstruct):
  1. Run VNN to obtain the initial sparse volume + mask.
  2. Optionally train (or load a saved checkpoint).
  3. Forward-pass the U-Net → refined volume.

Public API mirrors VNNReconstructor:
  .volume, .mask, .origin, .voxel_size, .shape
  .reconstruct(sequences) → np.ndarray
"""

from __future__ import annotations

import copy
import logging
import sys
import time
from pathlib import Path as _Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from config import PipelineConfig
from core.geometry import (
    build_voxel_grid, compute_volume_bounds, effective_transform,
    make_image_points, transform_to_tracker,
)
from core.vnn import VNNReconstructor
from us3d_io.mhd_reader import SequenceData

logger = logging.getLogger(__name__)

# ── Optional PyTorch ──────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ═════════════════════════════════════════════════════════════════════════════
# 3-D U-Net building blocks
# ═════════════════════════════════════════════════════════════════════════════

if HAS_TORCH:

    class _ConvBlock3D(nn.Module):
        """Two successive (Conv3d → BN → ReLU) layers."""
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class UNet3D(nn.Module):
        """
        Compact 3-D U-Net for volumetric refinement.

        Parameters
        ----------
        in_channels  : 2  (VNN intensity + mask)
        out_channels : 1  (refined intensity)
        base_features: base channel count (default 16 → memory-friendly)
        depth        : number of encoder / decoder levels (default 3)
        """
        def __init__(self, in_channels: int = 2, out_channels: int = 1,
                     base_features: int = 16, depth: int = 3):
            super().__init__()
            f = base_features

            # ── Encoder ──────────────────────────────────────────────────────
            self.encoders = nn.ModuleList()
            self.pools    = nn.ModuleList()
            ch = in_channels
            self.enc_channels = []
            for d in range(depth):
                out = f * (2 ** d)
                self.encoders.append(_ConvBlock3D(ch, out))
                self.pools.append(nn.MaxPool3d(2))
                ch = out
                self.enc_channels.append(out)

            # ── Bottleneck ────────────────────────────────────────────────────
            bottleneck_ch = f * (2 ** depth)
            self.bottleneck = _ConvBlock3D(ch, bottleneck_ch)
            ch = bottleneck_ch

            # ── Decoder ───────────────────────────────────────────────────────
            self.upconvs  = nn.ModuleList()
            self.decoders = nn.ModuleList()
            for d in reversed(range(depth)):
                skip_ch = self.enc_channels[d]
                out = f * (2 ** d)
                self.upconvs.append(
                    nn.ConvTranspose3d(ch, out, kernel_size=2, stride=2)
                )
                self.decoders.append(_ConvBlock3D(out + skip_ch, out))
                ch = out

            # ── Head ──────────────────────────────────────────────────────────
            self.head = nn.Conv3d(ch, out_channels, kernel_size=1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            skips = []
            for enc, pool in zip(self.encoders, self.pools):
                x = enc(x)
                skips.append(x)
                x = pool(x)

            x = self.bottleneck(x)

            for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
                x = up(x)
                # Pad if spatial dims mismatch (odd input sizes)
                if x.shape != skip.shape:
                    diff = [s - x.shape[i+2] for i, s in enumerate(skip.shape[2:])]
                    x = F.pad(x, [0, diff[2], 0, diff[1], 0, diff[0]])
                x = torch.cat([skip, x], dim=1)
                x = dec(x)

            return torch.sigmoid(self.head(x))  # → [0, 1]


    # ── SSIM-based loss component ─────────────────────────────────────────────
    def _ssim3d_loss(pred: "torch.Tensor", target: "torch.Tensor",
                     window_size: int = 7) -> "torch.Tensor":
        """
        Simplified 3-D SSIM loss term (1 – SSIM), computed with a uniform
        window for efficiency.
        """
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        w = window_size
        pad = w // 2
        kernel = torch.ones(1, 1, w, w, w, device=pred.device) / (w ** 3)

        mu1 = F.conv3d(pred,   kernel, padding=pad)
        mu2 = F.conv3d(target, kernel, padding=pad)
        mu1_sq  = mu1 ** 2
        mu2_sq  = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv3d(pred   ** 2, kernel, padding=pad) - mu1_sq
        sigma2_sq = F.conv3d(target ** 2, kernel, padding=pad) - mu2_sq
        sigma12   = F.conv3d(pred * target, kernel, padding=pad) - mu1_mu2

        ssim_map = (
            (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        ) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return 1.0 - ssim_map.mean()


# ═════════════════════════════════════════════════════════════════════════════
# DL Reconstructor
# ═════════════════════════════════════════════════════════════════════════════

class DLReconstructor:
    """
    Deep-Learning 3-D volume reconstructor.

    Workflow
    --------
    1. Build the full VNN volume (used as both DL input and pseudo-GT target).
    2. Train the U-Net: for each trial, reconstruct without that trial (VNN)
       → input; full VNN → target.
    3. Refine the full VNN volume with the trained U-Net → final output.

    Parameters
    ----------
    config         : PipelineConfig
    image_to_probe : 4×4 calibration matrix (None = Identity)
    epochs         : training epochs per LOTO fold (default 20)
    lr             : Adam learning rate (default 1e-3)
    checkpoint_path: where to save / load the U-Net weights
    skip_training  : if True and checkpoint exists, skip training
    base_features  : U-Net base channel count (default 16)
    """

    def __init__(
        self,
        config: PipelineConfig,
        image_to_probe: Optional[np.ndarray] = None,
        epochs: int = 20,
        lr: float = 1e-3,
        checkpoint_path: Optional[_Path] = None,
        skip_training: bool = False,
        base_features: int = 16,
    ):
        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch is required for DLReconstructor. "
                "Install it with: pip install torch"
            )

        self.cfg             = config
        self.T_calib         = image_to_probe
        self.epochs          = epochs
        self.lr              = lr
        self.skip_training   = skip_training
        self.base_features   = base_features

        # Checkpoint path defaults to output_dir / "dl_unet.pt"
        self.checkpoint_path = (
            checkpoint_path
            if checkpoint_path is not None
            else config.output_dir / "dl_unet.pt"
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "DLReconstructor initialised | device=%s | epochs=%d | base_features=%d",
            self.device.type.upper(), epochs, base_features,
        )

        # Internal state
        self._volume = self._mask = self._origin = self._shape = None
        self._model: Optional[UNet3D] = None

    # ── Public properties (same API as VNNReconstructor) ──────────────────────

    @property
    def volume(self):
        if self._volume is None:
            raise RuntimeError("Call reconstruct() first.")
        return self._volume

    @property
    def mask(self):
        if self._mask is None:
            raise RuntimeError("Call reconstruct() first.")
        return self._mask

    @property
    def origin(self):
        return self._origin

    @property
    def voxel_size(self):
        return self.cfg.voxel_size_mm

    @property
    def shape(self):
        return self._shape

    # ── Main entry point ──────────────────────────────────────────────────────

    def reconstruct(self, sequences: List[SequenceData]) -> np.ndarray:
        """
        Full pipeline: VNN init → (train or load) U-Net → refine.
        """
        t0 = time.perf_counter()

        # ── Step 1: Full-data VNN (pseudo ground truth + shared bounding box) ─
        logger.info("═══ DL Step 1/4  Full-data VNN initialisation ═══")
        vnn_full = self._run_vnn(sequences)
        vol_full  = vnn_full.volume          # (NZ, NY, NX) float32
        mask_full = vnn_full.mask            # (NZ, NY, NX) bool
        self._origin = vnn_full.origin
        self._shape  = vnn_full.shape

        # ── Step 2: Train U-Net (or load checkpoint) ──────────────────────────
        logger.info("═══ DL Step 2/4  U-Net training ═══")
        model = UNet3D(in_channels=2, out_channels=1,
                       base_features=self.base_features).to(self.device)

        loaded = False
        if self.skip_training and self.checkpoint_path.exists():
            logger.info("  Skipping training – loading checkpoint: %s",
                        self.checkpoint_path)
            state = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(state)
            loaded = True

        if not loaded:
            self._train(model, sequences, vol_full, mask_full)
            torch.save(model.state_dict(), self.checkpoint_path)
            logger.info("  Checkpoint saved → %s", self.checkpoint_path)

        self._model = model

        # ── Step 3: Refine the full VNN volume ────────────────────────────────
        logger.info("═══ DL Step 3/4  U-Net inference (refinement) ═══")
        refined = self._infer(model, vol_full, mask_full)

        # ── Step 4: Optional Gaussian smoothing ───────────────────────────────
        if self.cfg.gaussian_sigma > 0:
            logger.info("═══ DL Step 4/4  Gaussian smoothing (sigma=%.1f vox) ═══",
                        self.cfg.gaussian_sigma)
            refined = _mask_smooth(refined, mask_full, self.cfg.gaussian_sigma)
        else:
            logger.info("═══ DL Step 4/4  Smoothing skipped ═══")

        self._volume = np.clip(refined, 0.0, 255.0).astype(np.float32)
        self._mask   = mask_full

        elapsed = time.perf_counter() - t0
        pct_fill = 100.0 * mask_full.sum() / mask_full.size
        logger.info("DL done in %.1f s  |  %.1f%% filled", elapsed, pct_fill)
        return self._volume

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _run_vnn(self, sequences: List[SequenceData]) -> VNNReconstructor:
        """Run VNN on the given sequences and return the reconstructor."""
        vnn = VNNReconstructor(self.cfg, image_to_probe=self.T_calib)
        vnn.reconstruct(sequences)
        return vnn

    def _prepare_tensor(
        self, volume: np.ndarray, mask: np.ndarray
    ) -> "torch.Tensor":
        """
        Stack normalised VNN intensity and binary mask into a 2-channel tensor:
          shape (1, 2, D, H, W) on self.device
        """
        vol_n  = (volume / 255.0).astype(np.float32)
        mask_f = mask.astype(np.float32)
        x = np.stack([vol_n, mask_f], axis=0)             # (2, D, H, W)
        return torch.tensor(x, dtype=torch.float32,
                            device=self.device).unsqueeze(0)  # (1, 2, D, H, W)

    def _infer(self, model: "UNet3D",
               volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Run a single forward pass and return a float32 numpy array (0-255)."""
        model.eval()
        with torch.no_grad():
            x   = self._prepare_tensor(volume, mask)
            out = model(x)                           # (1, 1, D, H, W) ∈ [0,1]
        refined = out.squeeze().cpu().numpy() * 255.0
        # Preserve background zeros outside the mask
        refined[~mask] = 0.0
        return refined.astype(np.float32)

    def _train(
        self,
        model: "UNet3D",
        sequences: List[SequenceData],
        vol_full: np.ndarray,
        mask_full: np.ndarray,
    ) -> None:
        """
        Self-supervised training using Leave-One-Trial-Out:
          For each trial i → reconstruct WITHOUT trial i (VNN) → input
          Full VNN volume → target
        Training patches are processed on GPU to avoid OOM.
        """
        logger.info(
            "  Building LOTO training pairs (%d trials × %d epochs) …",
            len(sequences), self.epochs,
        )

        # Collect (input_vol, target_vol) pairs
        pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        for i, held_out in enumerate(sequences):
            remaining = [s for j, s in enumerate(sequences) if j != i]
            if not remaining:
                continue
            logger.info("  LOTO fold %d/%d – excluding trial '%s'",
                        i + 1, len(sequences), held_out.label)
            vnn_partial = self._run_vnn(remaining)

            # Resize to match full volume shape if bounding box differs slightly
            inp = _resize_to_match(vnn_partial.volume, vol_full.shape)
            msk = _resize_to_match(
                vnn_partial.mask.astype(np.float32), vol_full.shape
            ) > 0.5
            pairs.append((inp, msk))

        if not pairs:
            logger.warning("  No training pairs – skipping training.")
            return

        optimizer = Adam(model.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        # Pre-build the fixed target tensor (full-data VNN)
        tgt_t_full = self._prepare_tensor(vol_full, mask_full)
        tgt_intensity_full = tgt_t_full[:, :1]          # only intensity channel
        tgt_msk_t = torch.tensor(
            mask_full.astype(np.float32), dtype=torch.float32, device=self.device
        ).unsqueeze(0).unsqueeze(0)

        model.train()
        t_train = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for inp_vol, inp_msk in pairs:
                inp_t = self._prepare_tensor(inp_vol, inp_msk)

                optimizer.zero_grad()
                pred = model(inp_t)             # (1,1,D,H,W) ∈ [0,1]

                mse  = F.mse_loss(pred * tgt_msk_t, tgt_intensity_full * tgt_msk_t)
                ssim = _ssim3d_loss(pred * tgt_msk_t, tgt_intensity_full * tgt_msk_t)
                loss = mse + 0.1 * ssim

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg = epoch_loss / max(len(pairs), 1)

            if epoch == 1 or epoch % 5 == 0 or epoch == self.epochs:
                elapsed = time.perf_counter() - t_train
                logger.info(
                    "  Epoch %3d/%d  loss=%.5f  lr=%.2e  elapsed=%.1f s",
                    epoch, self.epochs,
                    avg, scheduler.get_last_lr()[0], elapsed,
                )

        logger.info(
            "  Training complete in %.1f s", time.perf_counter() - t_train
        )


# ═════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═════════════════════════════════════════════════════════════════════════════

def _mask_smooth(volume: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smooth only within the filled region, preserving background."""
    sm = gaussian_filter(volume * mask.astype(np.float32), sigma=sigma)
    wt = gaussian_filter(mask.astype(np.float32),           sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(wt > 1e-6, sm / wt, volume).astype(np.float32)
    out[~mask] = 0.0
    return np.clip(out, 0.0, 255.0)


def _resize_to_match(vol: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Resize `vol` to `target_shape` using scipy zoom.
    Used to align LOTO partial VNN volumes to the full-data bounding box.
    """
    if vol.shape == target_shape:
        return vol
    from scipy.ndimage import zoom
    factors = tuple(t / s for t, s in zip(target_shape, vol.shape))
    return zoom(vol, factors, order=1).astype(vol.dtype)
