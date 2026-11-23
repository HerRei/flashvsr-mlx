# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hermes Reisner. Part of the MLX port of FlashVSR (https://github.com/OpenImagingLab/FlashVSR).

import gc
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import mlx_flashvsr as mf

_ROOT = Path(__file__).resolve().parents[1]
_MODEL_DIR = _ROOT / "models" / "FlashVSR-v1.1"
_RESULTS_DIR = _ROOT / "results_mlx"

_REQUIRED_WEIGHTS = (
    _MODEL_DIR / "diffusion_pytorch_model_streaming_dmd.safetensors",
    _MODEL_DIR / "LQ_proj_in.ckpt",
    _MODEL_DIR / "TCDecoder.ckpt",
)
_REQUIRED_REFS = (
    _RESULTS_DIR / "lq_np.npy",
    _RESULTS_DIR / "noise_np.npy",
    _RESULTS_DIR / "gt_frames.npy",
    _RESULTS_DIR / "ref_frames.npy",
    _RESULTS_DIR / "ref_latents.npy",
)


def _psnr_255(actual: np.ndarray, expected: np.ndarray) -> float:
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    mse = float(np.mean(diff * diff))
    return 10.0 * math.log10((255.0 ** 2) / mse) if mse > 0 else float("inf")


def _cosine(actual: np.ndarray, expected: np.ndarray) -> float:
    a = actual.reshape(-1).astype(np.float64)
    b = expected.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 1.0


def test_real_weights_flashvsr_v1_1_parity():
    """Real FlashVSR-v1.1 bfloat16 parity against precomputed AMD ROCm PyTorch reference."""
    missing = [p for p in (*_REQUIRED_WEIGHTS, *_REQUIRED_REFS) if not p.exists()]
    if missing:
        pytest.skip(f"Missing real weights or reference files: {missing[0]}")

    lq_np = np.load(_RESULTS_DIR / "lq_np.npy")
    noise_np = np.load(_RESULTS_DIR / "noise_np.npy")
    gt_frames = np.load(_RESULTS_DIR / "gt_frames.npy")
    ref_frames = np.load(_RESULTS_DIR / "ref_frames.npy")
    ref_latents = np.load(_RESULTS_DIR / "ref_latents.npy")

    height, width, num_frames = 256, 512, 41
    topk_ratio = 2.0 * 768 * 1280 / (height * width)

    pipe = mf.FlashVSRTinyPipeline.from_pretrained(_MODEL_DIR, dtype="bfloat16")
    lq_bf16 = mx.array(lq_np).astype(mx.bfloat16)
    noise_bf16 = mx.array(noise_np).astype(mx.bfloat16)

    latents = pipe(
        height=height,
        width=width,
        num_frames=num_frames,
        LQ_video=lq_bf16,
        noise=noise_bf16,
        is_full_block=False,
        if_buffer=True,
        topk_ratio=topk_ratio,
        kv_ratio=3.0,
        local_range=11,
        color_fix=True,
        return_latents=True,
    )
    mx.eval(latents)

    mlx_latents_np = np.asarray(latents.transpose(0, 2, 1, 3, 4).astype(mx.float32))
    latent_cos = _cosine(mlx_latents_np, ref_latents)
    latent_max_abs = float(np.max(np.abs(mlx_latents_np - ref_latents)))

    pipe.TCDecoder.clean_mem()
    cond = lq_bf16[:, :, : pipe.last_lq_cur_idx, :, :]
    decoded = pipe.TCDecoder.decode_video(
        latents.transpose(0, 2, 1, 3, 4),
        parallel=False,
        show_progress_bar=False,
        cond=cond,
    )
    frames = decoded.transpose(0, 2, 1, 3, 4) * 2.0 - 1.0
    frames = pipe.ColorCorrector(
        frames,
        lq_bf16[:, :, : frames.shape[2], :, :],
        clip_range=(-1.0, 1.0),
        chunk_size=16,
        method="adain",
    )
    mx.eval(frames)

    mlx_frames_np = np.asarray(frames[0].astype(mx.float32))
    assert mlx_frames_np.shape == ref_frames.shape

    mlx_255 = (np.transpose(mlx_frames_np, (1, 2, 3, 0)) + 1.0) * 127.5
    ref_255 = (np.transpose(ref_frames, (1, 2, 3, 0)) + 1.0) * 127.5
    gt_37 = gt_frames[:37]

    psnr_vs_ref = _psnr_255(mlx_255, ref_255)
    frame_cos = _cosine(mlx_frames_np, ref_frames)
    mlx_psnr_gt = _psnr_255(mlx_255, gt_37)
    ref_psnr_gt = _psnr_255(ref_255, gt_37)

    print(
        f"[Real v1.1 41f 512x256 bf16] latent_cos={latent_cos:.6f} "
        f"latent_max_abs={latent_max_abs:.4f} psnr_vs_ref={psnr_vs_ref:.2f}dB "
        f"frame_cos={frame_cos:.6f} mlx_vs_gt={mlx_psnr_gt:.2f}dB "
        f"ref_vs_gt={ref_psnr_gt:.2f}dB"
    )

    # Measured: latent_cos=0.999706, latent_max_abs=0.7891, psnr_vs_ref=44.05 dB,
    # frame_cos=0.999619, mlx_vs_gt=23.68 dB vs ref_vs_gt=23.67 dB.
    # Tolerances allow cross-GPU bfloat16 rounding differences (M1 Pro MLX vs AMD ROCm).
    assert latent_cos >= 0.999
    assert latent_max_abs <= 1.5
    assert psnr_vs_ref >= 40.0
    assert frame_cos >= 0.999
    assert abs(mlx_psnr_gt - ref_psnr_gt) <= 0.25

    del pipe, lq_bf16, noise_bf16, latents, cond, decoded, frames
    gc.collect()
    mx.clear_cache()
