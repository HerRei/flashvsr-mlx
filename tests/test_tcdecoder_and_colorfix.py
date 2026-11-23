# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hermes Reisner. Part of the MLX port of FlashVSR (https://github.com/OpenImagingLab/FlashVSR).

import gc
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import numpy as np
import pytest
import torch
import torch.nn as tnn

import mlx_flashvsr as mf
from mlx_flashvsr import tcdecoder as mf_tcd

_CKPT_KEYS_PATH = Path(__file__).resolve().parent / "checkpoint_keys_v1.1.json"


def _to_np(x: mx.array) -> np.ndarray:
    mx.eval(x)
    if x.dtype == mx.bfloat16:
        return np.asarray(x.astype(mx.float32))
    return np.asarray(x)


def _parity_metrics(actual: np.ndarray, expected: np.ndarray):
    diff = np.abs(actual - expected)
    max_abs = float(np.max(diff))
    out_scale = max(float(np.max(np.abs(expected))), 1e-8)
    rel_max = max_abs / out_scale
    a_flat = actual.reshape(-1).astype(np.float64)
    e_flat = expected.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(a_flat) * np.linalg.norm(e_flat))
    cosine = float(np.dot(a_flat, e_flat) / denom) if denom > 0 else 1.0
    return max_abs, rel_max, cosine


def test_tcdecoder_checkpoint_keys_v1_1_coverage(pt_tcdecoder_mod):
    """Verify all 66 TCDecoder.ckpt keys/shapes map 1-to-1 without materializing weights."""
    data = json.loads(_CKPT_KEYS_PATH.read_text())
    tcd_spec = data["TCDecoder.ckpt"]
    assert len(tcd_spec) == 66

    # Verify upstream PyTorch TCDecoder matches TCDecoder.ckpt with zero missing/unexpected keys.
    pt_tcd = pt_tcdecoder_mod.build_tcdecoder(
        new_channels=[512, 256, 128, 128],
        device="cpu",
        dtype=torch.float32,
        new_latent_channels=16 + 768,
    )
    pt_sd = pt_tcd.state_dict()
    assert set(pt_sd.keys()) == set(tcd_spec.keys())
    for k, (shape, _) in tcd_spec.items():
        assert list(pt_sd[k].shape) == shape
    del pt_tcd, pt_sd
    gc.collect()

    mx_tcd = mf.build_tcdecoder(
        new_channels=[512, 256, 128, 128],
        dtype="float32",
        new_latent_channels=16 + 768,
    )
    mx_params = dict(tree_flatten(mx_tcd.parameters()))
    assert len(mx_params) == len(tcd_spec)

    mapped = {mf.map_pytorch_key(k): (k, tuple(shape)) for k, (shape, _) in tcd_spec.items()}
    assert set(mapped.keys()) == set(mx_params.keys())
    for mlx_key, (pt_key, pt_shape) in mapped.items():
        expected_shape = mf.map_pytorch_shape(pt_key, pt_shape)
        assert tuple(mx_params[mlx_key].shape) == expected_shape, (
            f"Shape mismatch for {pt_key} -> {mlx_key}: "
            f"got {tuple(mx_params[mlx_key].shape)}, expected {expected_shape}"
        )

    placeholders = {
        k: mx.broadcast_to(mx.zeros((1,), dtype=mx.float32), tuple(shape))
        for k, (shape, _) in tcd_spec.items()
    }
    mf.load_torch_state_dict(mx_tcd, placeholders, dtype="float32", strict=True)
    # Also test strict=False with an unexpected key (upstream infer script passes strict=False).
    placeholders_extra = dict(placeholders)
    placeholders_extra["encoder.unused.weight"] = mx.zeros((4, 4), dtype=mx.float32)
    mf.load_torch_state_dict(mx_tcd, placeholders_extra, dtype="float32", strict=False)

    assert mx.get_active_memory() < 50 * 1024 * 1024


def test_tpool_tgrow_memblock_apply_parity(pt_tcdecoder_mod):
    """Verify TPool, TGrow, MemBlock, and apply_model_with_memblocks (parallel=True and False)."""
    torch.manual_seed(701)
    c = 8
    pt_seq = tnn.Sequential(
        pt_tcdecoder_mod.MemBlock(c, c),
        pt_tcdecoder_mod.TPool(c, stride=2),
        pt_tcdecoder_mod.MemBlock(c, c * 2),
        pt_tcdecoder_mod.TGrow(c * 2, stride=2),
        pt_tcdecoder_mod.MemBlock(c * 2, c * 2),
    ).eval()
    with torch.no_grad():
        for p in pt_seq.parameters():
            p.normal_(mean=0.0, std=0.2)

    mx_seq = nn.Sequential(
        mf.MemBlock(c, c),
        mf.TPool(c, stride=2),
        mf.MemBlock(c, c * 2),
        mf.TGrow(c * 2, stride=2),
        mf.MemBlock(c * 2, c * 2),
    )
    pt_sd = {f"layers.{k}" if False else k: v for k, v in pt_seq.state_dict().items()}
    # Map leading `<idx>.` of top-level Sequential into `layers.<idx>.`
    remapped_sd = {}
    for k, v in pt_sd.items():
        head, rest = k.split(".", 1)
        remapped_sd[f"layers.{head}.{rest}"] = v
    mf.load_torch_state_dict(mx_seq, remapped_sd, dtype="float32", strict=True)

    x_pt = torch.randn(2, 4, c, 8, 8, dtype=torch.float32)
    x_mx = mx.array(x_pt.numpy().copy())
    with torch.no_grad():
        y_pt_par, _ = pt_tcdecoder_mod.apply_model_with_memblocks(
            pt_seq, x_pt.clone(), parallel=True, show_progress_bar=False, mem=None
        )
        y_pt_seq, mem_pt = pt_tcdecoder_mod.apply_model_with_memblocks(
            pt_seq, x_pt.clone(), parallel=False, show_progress_bar=False, mem=[None] * len(pt_seq)
        )
    assert torch.allclose(y_pt_par, y_pt_seq, atol=1e-5)

    y_mx_par, _ = mf.apply_model_with_memblocks(mx_seq, x_mx, parallel=True)
    y_mx_seq, mem_mx = mf.apply_model_with_memblocks(mx_seq, x_mx, parallel=False)

    max_abs_p, rel_p, cos_p = _parity_metrics(_to_np(y_mx_par), y_pt_par.numpy())
    max_abs_s, rel_s, cos_s = _parity_metrics(_to_np(y_mx_seq), y_pt_seq.numpy())
    print(f"[TPool/TGrow/MemBlock parallel fp32] max_abs={max_abs_p:.3e} rel={rel_p:.3e} cos={cos_p:.6f}")
    print(f"[TPool/TGrow/MemBlock seq fp32] max_abs={max_abs_s:.3e} rel={rel_s:.3e} cos={cos_s:.6f}")
    assert rel_p <= 1e-4
    assert rel_s <= 1e-4
    assert len(mem_mx) == len(mem_pt)


def test_tcdecoder_streaming_and_cond_parity(pt_tcdecoder_mod):
    """Verify full-width TCDecoder parity across two consecutive decode_video calls (MemBlock state + cond)."""
    torch.manual_seed(702)
    channels = [512, 256, 128, 128]
    latent_ch = 16 + 768

    pt_tcd = pt_tcdecoder_mod.build_tcdecoder(
        new_channels=channels,
        device="cpu",
        dtype=torch.float32,
        new_latent_channels=latent_ch,
    ).eval()
    with torch.no_grad():
        for name, p in pt_tcd.named_parameters():
            if "bias" in name:
                p.normal_(mean=0.0, std=0.05)
            else:
                p.normal_(mean=0.0, std=0.02)

    sd = {k: v.clone() for k, v in pt_tcd.state_dict().items()}

    # Chunk 1: 4 latent frames -> cond has 13 frames (13 % 4 == 1, exercises PixelShuffle3d padding)
    lat1_pt = torch.randn(1, 4, 16, 8, 8, dtype=torch.float32)
    cond1_pt = torch.randn(1, 3, 13, 64, 64, dtype=torch.float32)
    # Chunk 2: 2 latent frames without clean_mem() -> exercises persistent self.mem and trim_flag=False
    lat2_pt = torch.randn(1, 2, 16, 8, 8, dtype=torch.float32)
    cond2_pt = torch.randn(1, 3, 8, 64, 64, dtype=torch.float32)

    pt_tcd.clean_mem()
    with torch.no_grad():
        out1_pt = pt_tcd.decode_video(lat1_pt, parallel=False, show_progress_bar=False, cond=cond1_pt)
        out2_pt = pt_tcd.decode_video(lat2_pt, parallel=False, show_progress_bar=False, cond=cond2_pt)

    assert tuple(out1_pt.shape) == (1, 13, 3, 64, 64)
    assert tuple(out2_pt.shape) == (1, 8, 3, 64, 64)
    out1_np = out1_pt.numpy().copy()
    out2_np = out2_pt.numpy().copy()
    lat1_np = lat1_pt.numpy().copy()
    cond1_np = cond1_pt.numpy().copy()
    lat2_np = lat2_pt.numpy().copy()
    cond2_np = cond2_pt.numpy().copy()

    del pt_tcd, lat1_pt, cond1_pt, lat2_pt, cond2_pt, out1_pt, out2_pt
    gc.collect()

    # MLX float32
    mx_tcd_fp32 = mf.build_tcdecoder(
        new_channels=channels,
        dtype="float32",
        new_latent_channels=latent_ch,
    )
    mf.load_torch_state_dict(mx_tcd_fp32, sd, dtype="float32", strict=True)
    mx_tcd_fp32.clean_mem()
    out1_mx = mx_tcd_fp32.decode_video(
        mx.array(lat1_np), parallel=False, cond=mx.array(cond1_np)
    )
    out2_mx = mx_tcd_fp32.decode_video(
        mx.array(lat2_np), parallel=False, cond=mx.array(cond2_np)
    )

    m1_abs, m1_rel, m1_cos = _parity_metrics(_to_np(out1_mx), out1_np)
    m2_abs, m2_rel, m2_cos = _parity_metrics(_to_np(out2_mx), out2_np)
    print(f"[TCDecoder chunk1 fp32] max_abs={m1_abs:.3e} rel={m1_rel:.3e} cos={m1_cos:.6f}")
    print(f"[TCDecoder chunk2 fp32] max_abs={m2_abs:.3e} rel={m2_rel:.3e} cos={m2_cos:.6f}")
    assert m1_rel <= 1e-4
    assert m2_rel <= 1e-4

    # Verify parallel=True on chunk 1 matches parallel=False
    mx_tcd_fp32.clean_mem()
    out1_par = mx_tcd_fp32.decode_video(
        mx.array(lat1_np), parallel=True, cond=mx.array(cond1_np)
    )
    _, rel_par, _ = _parity_metrics(_to_np(out1_par), out1_np)
    assert rel_par <= 1e-4

    del mx_tcd_fp32, out1_mx, out2_mx, out1_par
    gc.collect()
    mx.clear_cache()

    # MLX bfloat16
    mx_tcd_bf16 = mf.build_tcdecoder(
        new_channels=channels,
        dtype="bfloat16",
        new_latent_channels=latent_ch,
    )
    mf.load_torch_state_dict(mx_tcd_bf16, sd, dtype="bfloat16", strict=True)
    del sd
    gc.collect()

    mx_tcd_bf16.clean_mem()
    out1_bf = mx_tcd_bf16.decode_video(
        mx.array(lat1_np).astype(mx.bfloat16),
        parallel=False,
        cond=mx.array(cond1_np).astype(mx.bfloat16),
    )
    out2_bf = mx_tcd_bf16.decode_video(
        mx.array(lat2_np).astype(mx.bfloat16),
        parallel=False,
        cond=mx.array(cond2_np).astype(mx.bfloat16),
    )

    _, _, cos1_bf = _parity_metrics(_to_np(out1_bf), out1_np)
    _, _, cos2_bf = _parity_metrics(_to_np(out2_bf), out2_np)
    print(f"[TCDecoder chunk1 bf16] cos={cos1_bf:.6f}")
    print(f"[TCDecoder chunk2 bf16] cos={cos2_bf:.6f}")
    assert cos1_bf >= 0.999
    assert cos2_bf >= 0.999

    del mx_tcd_bf16, out1_bf, out2_bf
    gc.collect()
    mx.clear_cache()


@pytest.mark.parametrize("method", ["wavelet", "adain"])
@pytest.mark.parametrize("chunk_size", [None, 4])
def test_colorfix_parity(flashvsr_tiny, method, chunk_size):
    """Verify ColorCorrectorWavelet parity for wavelet and adain methods with and without chunking."""
    torch.manual_seed(703)
    hq_pt = torch.randn(1, 3, 10, 64, 64, dtype=torch.float32) * 0.6
    lq_pt = torch.randn(1, 3, 10, 64, 64, dtype=torch.float32) * 0.6

    pt_cc = flashvsr_tiny.TorchColorCorrectorWavelet(levels=5)
    with torch.no_grad():
        out_pt = pt_cc(hq_pt, lq_pt, clip_range=(-1.0, 1.0), method=method, chunk_size=chunk_size)
    out_np = out_pt.numpy()

    mx_cc = mf.ColorCorrectorWavelet(levels=5)
    out_mx_fp32 = mx_cc(
        mx.array(hq_pt.numpy()),
        mx.array(lq_pt.numpy()),
        clip_range=(-1.0, 1.0),
        method=method,
        chunk_size=chunk_size,
    )
    max_abs, rel_max, cos = _parity_metrics(_to_np(out_mx_fp32), out_np)
    print(f"[ColorFix {method} chunk={chunk_size} fp32] max_abs={max_abs:.3e} rel={rel_max:.3e} cos={cos:.6f}")
    assert rel_max <= 1e-4

    out_mx_bf16 = mx_cc(
        mx.array(hq_pt.numpy()).astype(mx.bfloat16),
        mx.array(lq_pt.numpy()).astype(mx.bfloat16),
        clip_range=(-1.0, 1.0),
        method=method,
        chunk_size=chunk_size,
    )
    _, _, cos_bf = _parity_metrics(_to_np(out_mx_bf16), out_np)
    print(f"[ColorFix {method} chunk={chunk_size} bf16] cos={cos_bf:.6f}")
    assert cos_bf >= 0.999
