# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hermes Reisner. Part of the MLX port of FlashVSR (https://github.com/OpenImagingLab/FlashVSR).

import gc
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
import torch

import mlx_flashvsr as mf

_PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "FlashVSR"
    / "examples"
    / "WanVSR"
    / "prompt_tensor"
    / "posi_prompt.pth"
)


def _pop_pt_to_mlx_state(pt_sd: dict, dtype: str) -> dict:
    converted = {}
    for k in list(pt_sd.keys()):
        t = pt_sd.pop(k)
        arr = mf.torch_to_mlx(t, dtype=dtype)
        if arr.ndim == 5:
            arr = arr.transpose(0, 2, 3, 4, 1)
        elif arr.ndim == 4 and k.endswith(".weight"):
            arr = arr.transpose(0, 2, 3, 1)
        mx.eval(arr)
        converted[mf.map_pytorch_key(k)] = arr
        del t
    gc.collect()
    return converted


def _cast_module_inplace_bf16(mod):
    flat = dict(tree_flatten(mod.parameters()))
    for k in list(flat.keys()):
        arr = flat.pop(k)
        v = arr.astype(mx.bfloat16)
        mx.eval(v)
        mod.update(tree_unflatten([(k, v)]))
        del arr
    gc.collect()
    mx.clear_cache()


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


def test_end_to_end_tiny_pipeline_pixels_parity(
    wan_dit, flashvsr_tiny, pt_lq_proj_cls, pt_tcdecoder_mod
):
    """End-to-end LQ_video -> output RGB frames parity (reduced-layer DiT + full TCDecoder + ColorFix)."""
    torch.manual_seed(808)
    cfg = mf.WAN_1_3B_CONFIG
    num_layers = 2
    height, width = 128, 128
    num_frames = 41  # 3 streaming chunks -> 10 latent frames -> 37 output frames

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

    tcd_channels = [512, 256, 128, 128]
    pt_tcd = pt_tcdecoder_mod.build_tcdecoder(
        new_channels=tcd_channels,
        device="cpu",
        dtype=torch.float32,
        new_latent_channels=16 + 768,
    ).eval()
    with torch.no_grad():
        for name, p in pt_tcd.named_parameters():
            if "bias" in name:
                p.normal_(mean=0.0, std=0.05)
            else:
                p.normal_(mean=0.0, std=0.02)
        # Scale the final RGB head so decoded frames span [0, 1] across each channel.
        pt_tcd.decoder[-1].weight.normal_(mean=0.0, std=1.2)
        pt_tcd.decoder[-1].bias.fill_(0.5)

    dit_sd = {k: v for k, v in pt_dit.state_dict().items() if not k.startswith("LQ_proj_in.")}
    lq_sd = pt_lq.state_dict()
    tcd_sd = pt_tcd.state_dict()

    pt_pipe = flashvsr_tiny.FlashVSRTinyPipeline(device="cpu", torch_dtype=torch.float32)
    pt_pipe.dit = pt_dit
    pt_pipe.TCDecoder = pt_tcd

    ctx_pt = torch.load(str(_PROMPT_PATH), map_location="cpu", weights_only=True).to(torch.float32)
    pt_pipe.init_cross_kv(context_tensor=ctx_pt)

    lq_video_pt = torch.randn(1, 3, num_frames, height, width, dtype=torch.float32).clamp(-1.0, 1.0)
    noise_pt = torch.randn(1, 16, (num_frames - 1) // 4, height // 8, width // 8, dtype=torch.float32)
    pt_pipe.generate_noise = lambda shape, seed=None, device="cpu", dtype=torch.float32: noise_pt

    topk_ratio = 2.0 * 768 * 1280 / (height * width)
    kv_ratio = 3.0
    local_range = 11

    with torch.no_grad():
        pt_frames_cf = pt_pipe(
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
            color_fix=True,
        )

    assert tuple(pt_frames_cf.shape) == (3, 37, height, width)
    pt_cf_np = pt_frames_cf.numpy().copy()
    lq_video_np = lq_video_pt.numpy().copy()
    noise_np = noise_pt.numpy().copy()

    del pt_pipe, pt_dit, pt_lq, pt_tcd, ctx_pt, pt_frames_cf, lq_video_pt, noise_pt
    gc.collect()

    dit_mx_sd = _pop_pt_to_mlx_state(dit_sd, dtype="float32")
    lq_mx_sd = _pop_pt_to_mlx_state(lq_sd, dtype="float32")
    tcd_mx_sd = _pop_pt_to_mlx_state(tcd_sd, dtype="float32")
    del dit_sd, lq_sd, tcd_sd
    gc.collect()

    # MLX float32 end-to-end pass
    mx_dit_fp32 = mf.WanModel(**model_kwargs)
    mx_dit_fp32.update(tree_unflatten(list(dit_mx_sd.items())))
    del dit_mx_sd
    mx_lq_fp32 = mf.Causal_LQ4x_Proj(in_dim=3, out_dim=cfg["dim"], layer_num=1)
    mx_lq_fp32.update(tree_unflatten(list(lq_mx_sd.items())))
    del lq_mx_sd
    mx_dit_fp32.LQ_proj_in = mx_lq_fp32
    mx_tcd_fp32 = mf.build_tcdecoder(
        new_channels=tcd_channels,
        dtype="float32",
        new_latent_channels=16 + 768,
    )
    mx_tcd_fp32.update(tree_unflatten(list(tcd_mx_sd.items())))
    del tcd_mx_sd
    gc.collect()

    mx_pipe_fp32 = mf.FlashVSRTinyPipeline(
        dit=mx_dit_fp32, tcdecoder=mx_tcd_fp32, dtype="float32"
    )
    mx_pipe_fp32.init_cross_kv()

    mx_cf_fp32 = mx_pipe_fp32(
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
        color_fix=True,
    )

    cf_abs, cf_rel, cf_cos = _parity_metrics(_to_np(mx_cf_fp32), pt_cf_np)
    print(f"[E2E 41f 128x128 color_fix=True fp32] max_abs={cf_abs:.3e} rel={cf_rel:.3e} cos={cf_cos:.6f}")
    assert cf_rel <= 1e-4

    # Cast modules to bfloat16 in-place and clear float32 buffers before running bfloat16 inference
    del mx_cf_fp32, mx_pipe_fp32
    gc.collect()
    mx.clear_cache()

    _cast_module_inplace_bf16(mx_dit_fp32)
    _cast_module_inplace_bf16(mx_tcd_fp32)
    mx_dit_bf16 = mx_dit_fp32
    mx_tcd_bf16 = mx_tcd_fp32

    mx_pipe_bf16 = mf.FlashVSRTinyPipeline(
        dit=mx_dit_bf16, tcdecoder=mx_tcd_bf16, dtype="bfloat16"
    )
    mx_pipe_bf16.init_cross_kv()

    mx_cf_bf16 = mx_pipe_bf16(
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
        color_fix=True,
    )

    _, _, cf_cos_bf = _parity_metrics(_to_np(mx_cf_bf16), pt_cf_np)
    print(f"[E2E 41f 128x128 color_fix=True bf16] cos={cf_cos_bf:.6f}")
    assert cf_cos_bf >= 0.999

    del mx_pipe_bf16, mx_dit_bf16, mx_lq_fp32, mx_tcd_bf16, mx_cf_bf16
    gc.collect()
    mx.clear_cache()
