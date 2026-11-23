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


def test_causal_lq4x_proj_streaming_and_forward(pt_lq_proj_cls):
    torch.manual_seed(404)
    in_dim, out_dim, layer_num = 3, 1536, 2
    h, w = 128, 128

    pt_proj = pt_lq_proj_cls(in_dim=in_dim, out_dim=out_dim, layer_num=layer_num).eval()
    with torch.no_grad():
        pt_proj.norm1.gamma.normal_(mean=1.0, std=0.1)
        pt_proj.norm2.gamma.normal_(mean=1.0, std=0.1)
    sd = pt_proj.state_dict()

    # Build 9-frame video: clip 0 gets 1 frame, clips 1..2 get 4 frames each; plus extra 4-frame clips.
    clips_pt = [torch.randn(1, in_dim, 1, h, w, dtype=torch.float32)] + [
        torch.randn(1, in_dim, 4, h, w, dtype=torch.float32) for _ in range(4)
    ]

    pt_proj.clear_cache()
    pt_outs = []
    pt_caches = []
    with torch.no_grad():
        for clip in clips_pt:
            out = pt_proj.stream_forward(clip)
            pt_outs.append([t.numpy().copy() for t in out] if out is not None else None)
            pt_caches.append(
                (pt_proj.cache["conv1"].numpy().copy(), pt_proj.cache["conv2"].numpy().copy())
            )
        full_video_pt = torch.cat(clips_pt[:3], dim=2)
        full_pt = [t.numpy().copy() for t in pt_proj(full_video_pt)]

    clips_np = [c.numpy().copy() for c in clips_pt]
    full_video_np = full_video_pt.numpy().copy()
    pt_proj.clear_cache()
    del pt_proj, clips_pt, full_video_pt
    gc.collect()

    sd_mx = _pop_pt_to_mlx_state(sd, dtype="float32")
    del sd
    gc.collect()

    # MLX float32 pass
    mx_proj_fp32 = mf.Causal_LQ4x_Proj(in_dim=in_dim, out_dim=out_dim, layer_num=layer_num)
    mx_proj_fp32.update(tree_unflatten(list(sd_mx.items())))
    del sd_mx
    mx_proj_fp32.clear_cache()

    for idx, clip_np in enumerate(clips_np):
        out_mx = mx_proj_fp32.stream_forward(mx.array(clip_np))
        if idx == 0:
            assert pt_outs[idx] is None and out_mx is None
        else:
            assert out_mx is not None and len(out_mx) == layer_num
            for layer_i in range(layer_num):
                max_abs, rel_max, cos = _parity_metrics(
                    _to_np(out_mx[layer_i]), pt_outs[idx][layer_i]
                )
                print(
                    f"[Causal_LQ4x_Proj stream clip {idx} layer {layer_i} fp32] "
                    f"max_abs={max_abs:.3e} rel={rel_max:.3e} cos={cos:.6f}"
                )
                assert rel_max <= 1e-4

        for cache_idx, cache_name in enumerate(("conv1", "conv2")):
            max_abs, rel_max, cos = _parity_metrics(
                _to_np(mx_proj_fp32.cache[cache_name]),
                pt_caches[idx][cache_idx],
            )
            print(
                f"[Causal_LQ4x_Proj cache {cache_name} clip {idx} fp32] "
                f"max_abs={max_abs:.3e} rel={rel_max:.3e}"
            )
            assert rel_max <= 1e-4

    # Non-streaming forward on 9 frames (1 + 4 + 4)
    full_mx = mx_proj_fp32(mx.array(full_video_np))
    for layer_i in range(layer_num):
        max_abs, rel_max, cos = _parity_metrics(
            _to_np(full_mx[layer_i]), full_pt[layer_i]
        )
        print(
            f"[Causal_LQ4x_Proj full layer {layer_i} fp32] "
            f"max_abs={max_abs:.3e} rel={rel_max:.3e} cos={cos:.6f}"
        )
        assert rel_max <= 1e-4

    mx_proj_bf16 = mf.Causal_LQ4x_Proj(in_dim=in_dim, out_dim=out_dim, layer_num=layer_num)
    _cast_module_bf16(mx_proj_fp32, mx_proj_bf16)
    del mx_proj_fp32, out_mx, full_mx
    gc.collect()
    mx.clear_cache()

    # MLX bfloat16 pass
    mx_proj_bf16.clear_cache()

    for idx, clip_np in enumerate(clips_np):
        out_bf = mx_proj_bf16.stream_forward(mx.array(clip_np).astype(mx.bfloat16))
        if idx > 0:
            for layer_i in range(layer_num):
                _, _, cos_bf = _parity_metrics(
                    _to_np(out_bf[layer_i]), pt_outs[idx][layer_i]
                )
                print(f"[Causal_LQ4x_Proj stream clip {idx} layer {layer_i} bf16] cos={cos_bf:.6f}")
                assert cos_bf >= 0.999
        for cache_idx, cache_name in enumerate(("conv1", "conv2")):
            _, _, cos_c = _parity_metrics(
                _to_np(mx_proj_bf16.cache[cache_name]),
                pt_caches[idx][cache_idx],
            )
            assert cos_c >= 0.999

    del mx_proj_bf16, out_bf
    gc.collect()
    mx.clear_cache()


def test_wan_model_streaming_three_steps(wan_dit, flashvsr_tiny):
    torch.manual_seed(505)
    cfg = mf.WAN_1_3B_CONFIG
    num_layers = 2
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
    sd = pt_dit.state_dict()

    ctx_pt = torch.load(str(_PROMPT_PATH), map_location="cpu", weights_only=True).to(torch.float32)
    timestep_pt = torch.tensor([1000.0], dtype=torch.float32)
    with torch.no_grad():
        pt_dit.reinit_cross_kv(ctx_pt)
        t_pt = pt_dit.time_embedding(wan_dit.sinusoidal_embedding_1d(pt_dit.freq_dim, timestep_pt))
        t_mod_pt = pt_dit.time_projection(t_pt).unflatten(1, (6, pt_dit.dim))

    # Latent spatial size 64x64 -> patchified 32x32 -> 4x4 block grid; local_range=3 exercises local & topk pruning.
    lat_h, lat_w = 64, 64
    p_h, p_w = lat_h // 2, lat_w // 2
    local_range = 3
    kv_ratio = 3.0
    topk_ratio = 2.0

    steps = [(0, 6), (1, 2), (2, 2)]
    inputs_pt = [
        torch.randn(1, cfg["in_dim"], f, lat_h, lat_w, dtype=torch.float32)
        for _, f in steps
    ]
    lq_latents_pt = [
        [torch.randn(1, f * p_h * p_w, cfg["dim"], dtype=torch.float32) for _ in range(num_layers)]
        for _, f in steps
    ]

    pt_ck = [None] * num_layers
    pt_cv = [None] * num_layers
    pt_outputs = []
    pt_cache_snaps = []
    with torch.no_grad():
        for step_i, (proc_idx, _) in enumerate(steps):
            out_pt, pt_ck, pt_cv = flashvsr_tiny.model_fn_wan_video(
                pt_dit,
                x=inputs_pt[step_i],
                timestep=timestep_pt,
                context=None,
                LQ_latents=lq_latents_pt[step_i],
                is_full_block=False,
                is_stream=True,
                pre_cache_k=pt_ck,
                pre_cache_v=pt_cv,
                topk_ratio=topk_ratio,
                kv_ratio=kv_ratio,
                cur_process_idx=proc_idx,
                t_mod=t_mod_pt,
                t=t_pt,
                local_range=local_range,
            )
            pt_outputs.append(out_pt.numpy().copy())
            pt_cache_snaps.append(
                ([c.numpy().copy() for c in pt_ck], [c.numpy().copy() for c in pt_cv])
            )

    inputs_np = [x.numpy().copy() for x in inputs_pt]
    lq_latents_np = [[t.numpy().copy() for t in step_lq] for step_lq in lq_latents_pt]
    ctx_np = ctx_pt.numpy().copy()
    pt_dit.clear_cross_kv()
    del pt_dit, pt_ck, pt_cv, inputs_pt, lq_latents_pt, ctx_pt, t_pt, t_mod_pt, timestep_pt
    gc.collect()

    sd_mx = _pop_pt_to_mlx_state(sd, dtype="float32")
    del sd
    gc.collect()

    # MLX float32 pass
    mx_dit_fp32 = mf.WanModel(**model_kwargs)
    mx_dit_fp32.update(tree_unflatten(list(sd_mx.items())))
    del sd_mx
    ctx_mx_fp32 = mx.array(ctx_np)
    mx_dit_fp32.reinit_cross_kv(ctx_mx_fp32)
    timestep_mx_fp32 = mx.array([1000.0], dtype=mx.float32)
    t_mx_fp32 = mx_dit_fp32.time_embedding(
        mf.sinusoidal_embedding_1d(mx_dit_fp32.freq_dim, timestep_mx_fp32)
    )
    t_mod_mx_fp32 = mx_dit_fp32.time_projection(t_mx_fp32).reshape(
        t_mx_fp32.shape[0], 6, mx_dit_fp32.dim
    )

    mx_ck = [None] * num_layers
    mx_cv = [None] * num_layers
    for step_i, (proc_idx, f) in enumerate(steps):
        out_mx, mx_ck, mx_cv = mf.model_fn_wan_video(
            mx_dit_fp32,
            x=mx.array(inputs_np[step_i]),
            timestep=timestep_mx_fp32,
            context=None,
            LQ_latents=[mx.array(t) for t in lq_latents_np[step_i]],
            is_full_block=False,
            is_stream=True,
            pre_cache_k=mx_ck,
            pre_cache_v=mx_cv,
            topk_ratio=topk_ratio,
            kv_ratio=kv_ratio,
            cur_process_idx=proc_idx,
            t_mod=t_mod_mx_fp32,
            t=t_mx_fp32,
            local_range=local_range,
        )
        max_abs, rel_max, cos = _parity_metrics(_to_np(out_mx), pt_outputs[step_i])
        print(
            f"[WanModel step {proc_idx} f={f} fp32] "
            f"out(max={max_abs:.3e}, rel={rel_max:.3e}, cos={cos:.6f})"
        )
        assert rel_max <= 1e-4

        for layer_i in range(num_layers):
            _, rel_ck, _ = _parity_metrics(
                _to_np(mx_ck[layer_i]), pt_cache_snaps[step_i][0][layer_i]
            )
            _, rel_cv, _ = _parity_metrics(
                _to_np(mx_cv[layer_i]), pt_cache_snaps[step_i][1][layer_i]
            )
            print(
                f"  layer {layer_i} cache: ck_rel={rel_ck:.3e} cv_rel={rel_cv:.3e}"
            )
            assert rel_ck <= 1e-4
            assert rel_cv <= 1e-4

    mx_dit_bf16 = mf.WanModel(**model_kwargs)
    _cast_module_bf16(mx_dit_fp32, mx_dit_bf16)
    mx_dit_fp32.clear_cross_kv()
    del mx_dit_fp32, mx_ck, mx_cv, out_mx, t_mx_fp32, t_mod_mx_fp32, ctx_mx_fp32
    gc.collect()
    mx.clear_cache()

    # MLX bfloat16 pass
    ctx_mx_bf16 = mx.array(ctx_np).astype(mx.bfloat16)
    mx_dit_bf16.reinit_cross_kv(ctx_mx_bf16)
    timestep_mx_bf16 = mx.array([1000.0], dtype=mx.bfloat16)
    t_mx_bf16 = mx_dit_bf16.time_embedding(
        mf.sinusoidal_embedding_1d(mx_dit_bf16.freq_dim, timestep_mx_bf16)
    )
    t_mod_mx_bf16 = mx_dit_bf16.time_projection(t_mx_bf16).reshape(
        t_mx_bf16.shape[0], 6, mx_dit_bf16.dim
    )

    bf_ck = [None] * num_layers
    bf_cv = [None] * num_layers
    for step_i, (proc_idx, f) in enumerate(steps):
        out_bf, bf_ck, bf_cv = mf.model_fn_wan_video(
            mx_dit_bf16,
            x=mx.array(inputs_np[step_i]).astype(mx.bfloat16),
            timestep=timestep_mx_bf16,
            context=None,
            LQ_latents=[mx.array(t).astype(mx.bfloat16) for t in lq_latents_np[step_i]],
            is_full_block=False,
            is_stream=True,
            pre_cache_k=bf_ck,
            pre_cache_v=bf_cv,
            topk_ratio=topk_ratio,
            kv_ratio=kv_ratio,
            cur_process_idx=proc_idx,
            t_mod=t_mod_mx_bf16,
            t=t_mx_bf16,
            local_range=local_range,
        )
        _, _, cos_bf = _parity_metrics(_to_np(out_bf), pt_outputs[step_i])
        cos_ck = min(
            _parity_metrics(_to_np(bf_ck[i]), pt_cache_snaps[step_i][0][i])[2]
            for i in range(num_layers)
        )
        cos_cv = min(
            _parity_metrics(_to_np(bf_cv[i]), pt_cache_snaps[step_i][1][i])[2]
            for i in range(num_layers)
        )
        print(
            f"[WanModel step {proc_idx} f={f} bf16] "
            f"out_cos={cos_bf:.6f} min_ck_cos={cos_ck:.6f} min_cv_cos={cos_cv:.6f}"
        )
        assert cos_bf >= 0.999
        assert cos_ck >= 0.999
        assert cos_cv >= 0.999

    mx_dit_bf16.clear_cross_kv()
    del mx_dit_bf16, bf_ck, bf_cv, out_bf
    gc.collect()
    mx.clear_cache()
