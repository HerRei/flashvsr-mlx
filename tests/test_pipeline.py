# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hermes Reisner. Part of the MLX port of FlashVSR (https://github.com/OpenImagingLab/FlashVSR).

import gc
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
import torch

import mlx_flashvsr as mf


def _pop_pt_to_mlx_state(pt_sd: dict, dtype: str) -> dict:
    converted = {}
    for k in list(pt_sd.keys()):
        t = pt_sd.pop(k)
        arr = mf.torch_to_mlx(t, dtype=dtype)
        if arr.ndim == 5:
            arr = arr.transpose(0, 2, 3, 4, 1)
        mx.eval(arr)
        converted[mf.map_pytorch_key(k)] = arr
        del t
    gc.collect()
    return converted


def _cast_module_bf16(src_mod, dst_mod):
    dst_keys = {k for k, _ in tree_flatten(dst_mod.parameters())}
    flat = dict(tree_flatten(src_mod.parameters()))
    bf_items = []
    for k in list(flat.keys()):
        arr = flat.pop(k)
        if k in dst_keys:
            v = arr.astype(mx.bfloat16)
            mx.eval(v)
            bf_items.append((k, v))
    dst_mod.update(tree_unflatten(bf_items))

_PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "FlashVSR"
    / "examples"
    / "WanVSR"
    / "prompt_tensor"
    / "posi_prompt.pth"
)


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


class _TCDecoderCapture:
    """Test-only stub capturing denoised latents passed by upstream FlashVSRTinyPipeline."""

    def __init__(self):
        self.captured_latents = None
        self.captured_cond_frames = None

    def clean_mem(self):
        pass

    def decode_video(self, latents_btchw, **kwargs):
        self.captured_latents = latents_btchw.transpose(1, 2).clone()
        cond = kwargs.get("cond")
        if cond is not None:
            self.captured_cond_frames = int(cond.shape[2])
        return latents_btchw


def test_pipeline_streaming_parity(wan_dit, flashvsr_tiny, pt_lq_proj_cls):
    torch.manual_seed(606)
    cfg = mf.WAN_1_3B_CONFIG
    num_layers = 2
    height, width = 128, 128  # smallest multiple of 128
    num_frames = 41  # (41 - 1) // 8 - 2 = 3 streaming steps (f=6, 2, 2 -> 10 latent frames)

    model_kwargs = {
        "dim": cfg["dim"],
        "in_dim": cfg["in_dim"],
        "ffn_dim": cfg["ffn_dim"],
        "out_dim": cfg["out_dim"],
        "text_dim": cfg["text_dim"],
        "freq_dim": cfg["freq_dim"],
        "eps": cfg["eps"],
        "patch_size": cfg["patch_size"],
        "num_heads": cfg["num_heads"],
        "num_layers": num_layers,
    }

    pt_dit = wan_dit.WanModel(**model_kwargs).eval()
    pt_lq = pt_lq_proj_cls(in_dim=3, out_dim=cfg["dim"], layer_num=1).eval()
    with torch.no_grad():
        pt_lq.norm1.gamma.normal_(mean=1.0, std=0.1)
        pt_lq.norm2.gamma.normal_(mean=1.0, std=0.1)
    pt_dit.LQ_proj_in = pt_lq

    dit_sd = {k: v for k, v in pt_dit.state_dict().items() if not k.startswith("LQ_proj_in.")}
    lq_sd = pt_lq.state_dict()

    pt_pipe = flashvsr_tiny.FlashVSRTinyPipeline(device="cpu", torch_dtype=torch.float32)
    pt_pipe.dit = pt_dit
    capture = _TCDecoderCapture()
    pt_pipe.TCDecoder = capture

    ctx_pt = torch.load(str(_PROMPT_PATH), map_location="cpu", weights_only=True).to(torch.float32)
    pt_pipe.init_cross_kv(context_tensor=ctx_pt)

    lq_video_pt = torch.randn(1, 3, num_frames, height, width, dtype=torch.float32)
    noise_pt = torch.randn(1, 16, (num_frames - 1) // 4, height // 8, width // 8, dtype=torch.float32)

    # Inject fixed noise so both frameworks denoise identical initial latents.
    pt_pipe.generate_noise = lambda shape, seed=None, device="cpu", dtype=torch.float32: noise_pt

    topk_ratio = 2.0 * 768 * 1280 / (height * width)
    kv_ratio = 3.0
    local_range = 11

    with torch.no_grad():
        pt_pipe(
            prompt="",
            negative_prompt="",
            cfg_scale=1.0,
            num_inference_steps=1,
            seed=0,
            LQ_video=lq_video_pt,
            num_frames=num_frames,
            height=height,
            width=width,
            is_full_block=False,
            if_buffer=True,
            topk_ratio=topk_ratio,
            kv_ratio=kv_ratio,
            local_range=local_range,
            color_fix=False,
        )
    pt_latents = capture.captured_latents
    pt_cond_frames = capture.captured_cond_frames
    assert pt_latents is not None
    assert pt_cond_frames == 37
    assert tuple(pt_latents.shape) == (1, 16, 10, height // 8, width // 8)
    pt_latents_np = pt_latents.numpy().copy()

    lq_video_np = lq_video_pt.numpy().copy()
    noise_np = noise_pt.numpy().copy()
    del pt_pipe, pt_dit, pt_lq, capture, ctx_pt, pt_latents, lq_video_pt, noise_pt
    gc.collect()

    dit_mx_sd = _pop_pt_to_mlx_state(dit_sd, dtype="float32")
    lq_mx_sd = _pop_pt_to_mlx_state(lq_sd, dtype="float32")
    del dit_sd, lq_sd
    gc.collect()

    # MLX float32 pipeline pass (loading posi_prompt.pth via default path)
    mx_dit_fp32 = mf.WanModel(**model_kwargs)
    mx_dit_fp32.update(tree_unflatten(list(dit_mx_sd.items())))
    del dit_mx_sd
    mx_lq_fp32 = mf.Causal_LQ4x_Proj(in_dim=3, out_dim=cfg["dim"], layer_num=1)
    mx_lq_fp32.update(tree_unflatten(list(lq_mx_sd.items())))
    del lq_mx_sd
    mx_dit_fp32.LQ_proj_in = mx_lq_fp32

    mx_pipe_fp32 = mf.FlashVSRTinyPipeline(dit=mx_dit_fp32, dtype="float32")
    mx_pipe_fp32.init_cross_kv()
    mx_latents_fp32 = mx_pipe_fp32(
        height=height,
        width=width,
        num_frames=num_frames,
        LQ_video=mx.array(lq_video_np),
        is_full_block=False,
        if_buffer=True,
        topk_ratio=topk_ratio,
        kv_ratio=kv_ratio,
        local_range=local_range,
        noise=mx.array(noise_np),
    )
    assert mx_pipe_fp32.last_lq_cur_idx == pt_cond_frames

    max_abs, rel_max, cos = _parity_metrics(_to_np(mx_latents_fp32), pt_latents_np)
    print(
        f"[Pipeline 41f 128x128 fp32] max_abs={max_abs:.3e} rel={rel_max:.3e} cos={cos:.6f}"
    )
    assert rel_max <= 1e-4

    # MLX bfloat16 pipeline pass (cast weights from fp32 module then free fp32 copy)
    mx_dit_bf16 = mf.WanModel(**model_kwargs)
    _cast_module_bf16(mx_dit_fp32, mx_dit_bf16)
    mx_lq_bf16 = mf.Causal_LQ4x_Proj(in_dim=3, out_dim=cfg["dim"], layer_num=1)
    _cast_module_bf16(mx_lq_fp32, mx_lq_bf16)
    del mx_pipe_fp32, mx_dit_fp32, mx_lq_fp32, mx_latents_fp32
    gc.collect()
    mx.clear_cache()

    mx_dit_bf16.LQ_proj_in = mx_lq_bf16

    mx_pipe_bf16 = mf.FlashVSRTinyPipeline(dit=mx_dit_bf16, dtype="bfloat16")
    mx_pipe_bf16.init_cross_kv()
    mx_latents_bf16 = mx_pipe_bf16(
        height=height,
        width=width,
        num_frames=num_frames,
        LQ_video=mx.array(lq_video_np).astype(mx.bfloat16),
        is_full_block=False,
        if_buffer=True,
        topk_ratio=topk_ratio,
        kv_ratio=kv_ratio,
        local_range=local_range,
        noise=mx.array(noise_np).astype(mx.bfloat16),
    )

    _, _, cos_bf = _parity_metrics(_to_np(mx_latents_bf16), pt_latents_np)
    print(f"[Pipeline 41f 128x128 bf16] cos={cos_bf:.6f}")
    assert cos_bf >= 0.999

    # Verify seed-based noise generation and rounding (height=120 -> 128, num_frames=38 -> 41)
    lat_seeded = mx_pipe_bf16(
        height=120,
        width=120,
        num_frames=38,
        seed=0,
        LQ_video=mx.array(lq_video_np).astype(mx.bfloat16),
        if_buffer=False,
        topk_ratio=topk_ratio,
        kv_ratio=kv_ratio,
        local_range=local_range,
    )
    mx.eval(lat_seeded)
    assert tuple(lat_seeded.shape) == (1, 16, 10, 16, 16)

    del mx_pipe_bf16, mx_dit_bf16, mx_lq_bf16, mx_latents_bf16, lat_seeded
    gc.collect()
    mx.clear_cache()
