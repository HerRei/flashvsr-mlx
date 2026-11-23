# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hermes Reisner. Part of the MLX port of FlashVSR (https://github.com/OpenImagingLab/FlashVSR).

import mlx.core as mx
import numpy as np
import torch

import mlx_flashvsr as mf


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


def _build_chunk_freqs(pt_freqs_3d, mx_freqs_3d, f: int, h: int, w: int, f_start: int):
    pt_freqs = torch.cat(
        [
            pt_freqs_3d[0][f_start : f_start + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            pt_freqs_3d[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            pt_freqs_3d[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1)
    mx_freqs = mf.build_freqs_3d(mx_freqs_3d, f, h, w, f_start=f_start)
    return pt_freqs, mx_freqs


def test_self_attention_streaming_three_chunks(wan_dit):
    # h = w = 96 -> 12x12 block grid with local_range = 9 (window strictly smaller than grid)
    torch.manual_seed(101)
    dim, num_heads, head_dim = 256, 2, 128
    h, w = 96, 96
    local_range = 9
    kv_len = 3
    one_len = (h // 8) * (w // 8)  # 144 blocks per 2-frame chunk
    topk = int((one_len * one_len) * 2.0) - 1  # official flashvsr_tiny topk_ratio=2.0

    pt_sa = wan_dit.SelfAttention(dim, num_heads, eps=1e-6).eval()
    mx_sa_fp32 = mf.SelfAttention(dim, num_heads, eps=1e-6)
    mx_sa_bf16 = mf.SelfAttention(dim, num_heads, eps=1e-6)
    mf.load_torch_state_dict(mx_sa_fp32, pt_sa.state_dict(), dtype="float32")
    mf.load_torch_state_dict(mx_sa_bf16, pt_sa.state_dict(), dtype="bfloat16")

    pt_freqs_3d = wan_dit.precompute_freqs_cis_3d(head_dim, end=1024)
    mx_freqs_3d = mf.precompute_freqs_cis_3d(head_dim, end=1024)

    pt_ck = pt_cv = None
    mx_ck = mx_cv = None
    bf_ck = bf_cv = None

    chunks = [(6, 0), (2, 6), (2, 8)]
    expected_cache_chunks = [3, 3, 3]  # 3 -> (3+1 trimmed to 3) -> (3+1 trimmed to 3)

    for idx, ((f, f_start), exp_chunks) in enumerate(zip(chunks, expected_cache_chunks)):
        x_pt = torch.randn(1, f * h * w, dim, dtype=torch.float32)
        pt_freqs, mx_freqs = _build_chunk_freqs(pt_freqs_3d, mx_freqs_3d, f, h, w, f_start)

        with torch.no_grad():
            y_pt, pt_ck, pt_cv = pt_sa(
                x_pt,
                pt_freqs,
                f=f,
                h=h,
                w=w,
                topk=topk,
                kv_len=kv_len,
                is_stream=True,
                pre_cache_k=pt_ck,
                pre_cache_v=pt_cv,
                local_range=local_range,
            )

        y_mx, mx_ck, mx_cv = mx_sa_fp32(
            mx.array(x_pt.numpy()),
            mx_freqs,
            f=f,
            h=h,
            w=w,
            topk=topk,
            kv_len=kv_len,
            is_stream=True,
            pre_cache_k=mx_ck,
            pre_cache_v=mx_cv,
            local_range=local_range,
        )

        y_bf, bf_ck, bf_cv = mx_sa_bf16(
            mx.array(x_pt.numpy()).astype(mx.bfloat16),
            mx_freqs,
            f=f,
            h=h,
            w=w,
            topk=topk,
            kv_len=kv_len,
            is_stream=True,
            pre_cache_k=bf_ck,
            pre_cache_v=bf_cv,
            local_range=local_range,
        )

        assert pt_ck.shape[0] == exp_chunks * one_len
        assert mx_ck.shape[0] == exp_chunks * one_len

        max_abs, rel_max, cos = _parity_metrics(_to_np(y_mx), y_pt.numpy())
        max_abs_ck, rel_ck, _ = _parity_metrics(_to_np(mx_ck), pt_ck.numpy())
        max_abs_cv, rel_cv, _ = _parity_metrics(_to_np(mx_cv), pt_cv.numpy())
        _, _, cos_bf16 = _parity_metrics(_to_np(y_bf), y_pt.numpy())
        _, _, cos_bf16_ck = _parity_metrics(_to_np(bf_ck), pt_ck.numpy())
        _, _, cos_bf16_cv = _parity_metrics(_to_np(bf_cv), pt_cv.numpy())
        print(
            f"[SelfAttention chunk {idx} f={f}] "
            f"out(max={max_abs:.3e}, rel={rel_max:.3e}, cos={cos:.6f}) "
            f"ck(rel={rel_ck:.3e}) cv(rel={rel_cv:.3e}) "
            f"bf16_cos(out={cos_bf16:.6f}, ck={cos_bf16_ck:.6f}, cv={cos_bf16_cv:.6f})"
        )
        assert rel_max <= 1e-4
        assert rel_ck <= 1e-4
        assert rel_cv <= 1e-4
        assert cos_bf16 >= 0.999
        assert cos_bf16_ck >= 0.999
        assert cos_bf16_cv >= 0.999


def test_dit_block_streaming_small_dims(wan_dit):
    # 5x5 block grid (h=w=40) with local_range=3 so local window cuts
    torch.manual_seed(202)
    dim, num_heads, head_dim, ffn_dim = 256, 2, 128, 1024
    h, w = 40, 40
    local_range = 3
    kv_len = 3
    one_len = (h // 8) * (w // 8)
    topk = int((one_len * one_len) * 2.0) - 1  # official flashvsr_tiny topk_ratio=2.0

    pt_blk = wan_dit.DiTBlock(dim, num_heads, ffn_dim, eps=1e-6).eval()
    mx_blk_fp32 = mf.DiTBlock(dim, num_heads, ffn_dim, eps=1e-6)
    mx_blk_bf16 = mf.DiTBlock(dim, num_heads, ffn_dim, eps=1e-6)
    mf.load_torch_state_dict(mx_blk_fp32, pt_blk.state_dict(), dtype="float32")
    mf.load_torch_state_dict(mx_blk_bf16, pt_blk.state_dict(), dtype="bfloat16")

    ctx_pt = torch.randn(1, 32, dim, dtype=torch.float32)
    t_mod_pt = torch.randn(1, 6, dim, dtype=torch.float32)
    with torch.no_grad():
        pt_blk.cross_attn.init_cache(ctx_pt)
    mx_blk_fp32.cross_attn.init_cache(mx.array(ctx_pt.numpy()))
    mx_blk_bf16.cross_attn.init_cache(mx.array(ctx_pt.numpy()).astype(mx.bfloat16))

    pt_freqs_3d = wan_dit.precompute_freqs_cis_3d(head_dim, end=1024)
    mx_freqs_3d = mf.precompute_freqs_cis_3d(head_dim, end=1024)

    pt_ck = pt_cv = None
    mx_ck = mx_cv = None
    bf_ck = bf_cv = None

    for idx, (f, f_start) in enumerate(((6, 0), (2, 6), (2, 8))):
        x_pt = torch.randn(1, f * h * w, dim, dtype=torch.float32)
        pt_freqs, mx_freqs = _build_chunk_freqs(pt_freqs_3d, mx_freqs_3d, f, h, w, f_start)

        with torch.no_grad():
            y_pt, pt_ck, pt_cv = pt_blk(
                x_pt,
                None,
                t_mod_pt,
                pt_freqs,
                f,
                h,
                w,
                topk=topk,
                kv_len=kv_len,
                is_stream=True,
                pre_cache_k=pt_ck,
                pre_cache_v=pt_cv,
                local_range=local_range,
            )

        y_mx, mx_ck, mx_cv = mx_blk_fp32(
            mx.array(x_pt.numpy()),
            None,
            mx.array(t_mod_pt.numpy()),
            mx_freqs,
            f,
            h,
            w,
            topk=topk,
            kv_len=kv_len,
            is_stream=True,
            pre_cache_k=mx_ck,
            pre_cache_v=mx_cv,
            local_range=local_range,
        )

        y_bf, bf_ck, bf_cv = mx_blk_bf16(
            mx.array(x_pt.numpy()).astype(mx.bfloat16),
            None,
            mx.array(t_mod_pt.numpy()).astype(mx.bfloat16),
            mx_freqs,
            f,
            h,
            w,
            topk=topk,
            kv_len=kv_len,
            is_stream=True,
            pre_cache_k=bf_ck,
            pre_cache_v=bf_cv,
            local_range=local_range,
        )

        max_abs, rel_max, cos = _parity_metrics(_to_np(y_mx), y_pt.numpy())
        _, rel_ck, _ = _parity_metrics(_to_np(mx_ck), pt_ck.numpy())
        _, rel_cv, _ = _parity_metrics(_to_np(mx_cv), pt_cv.numpy())
        _, _, cos_bf16 = _parity_metrics(_to_np(y_bf), y_pt.numpy())
        print(
            f"[DiTBlock-small chunk {idx} f={f}] "
            f"out(max={max_abs:.3e}, rel={rel_max:.3e}, cos={cos:.6f}) "
            f"ck(rel={rel_ck:.3e}) cv(rel={rel_cv:.3e}) bf16_cos={cos_bf16:.6f}"
        )
        assert rel_max <= 1e-4
        assert rel_ck <= 1e-4
        assert rel_cv <= 1e-4
        assert cos_bf16 >= 0.999


def test_dit_block_real_1_3b_dims(wan_dit):
    # Real Wan 2.1 1.3B block dims at a small spatial size (4x4 block grid, local_range=3)
    torch.manual_seed(303)
    cfg = mf.WAN_1_3B_CONFIG
    dim = cfg["dim"]  # 1536
    num_heads = cfg["num_heads"]  # 12
    head_dim = cfg["head_dim"]  # 128
    ffn_dim = cfg["ffn_dim"]  # 8960
    eps = cfg["eps"]  # 1e-6

    f, h, w = 6, 32, 32
    local_range = 3
    kv_len = 3
    one_len = (h // 8) * (w // 8)  # 16
    topk = int((one_len * one_len) * 2.0) - 1  # official flashvsr_tiny topk_ratio=2.0

    pt_blk = wan_dit.DiTBlock(dim, num_heads, ffn_dim, eps=eps).eval()
    mx_blk_fp32 = mf.DiTBlock(dim, num_heads, ffn_dim, eps=eps)
    mx_blk_bf16 = mf.DiTBlock(dim, num_heads, ffn_dim, eps=eps)
    mf.load_torch_state_dict(mx_blk_fp32, pt_blk.state_dict(), dtype="float32")
    mf.load_torch_state_dict(mx_blk_bf16, pt_blk.state_dict(), dtype="bfloat16")

    ctx_pt = torch.randn(1, 32, dim, dtype=torch.float32)
    t_mod_pt = torch.randn(1, 6, dim, dtype=torch.float32)
    x_pt = torch.randn(1, f * h * w, dim, dtype=torch.float32)
    with torch.no_grad():
        pt_blk.cross_attn.init_cache(ctx_pt)
    mx_blk_fp32.cross_attn.init_cache(mx.array(ctx_pt.numpy()))
    mx_blk_bf16.cross_attn.init_cache(mx.array(ctx_pt.numpy()).astype(mx.bfloat16))

    pt_freqs_3d = wan_dit.precompute_freqs_cis_3d(head_dim, end=64)
    mx_freqs_3d = mf.precompute_freqs_cis_3d(head_dim, end=64)
    pt_freqs, mx_freqs = _build_chunk_freqs(pt_freqs_3d, mx_freqs_3d, f, h, w, 0)

    with torch.no_grad():
        y_pt, pt_ck, pt_cv = pt_blk(
            x_pt,
            None,
            t_mod_pt,
            pt_freqs,
            f,
            h,
            w,
            topk=topk,
            kv_len=kv_len,
            is_stream=True,
            local_range=local_range,
        )

    y_mx, mx_ck, mx_cv = mx_blk_fp32(
        mx.array(x_pt.numpy()),
        None,
        mx.array(t_mod_pt.numpy()),
        mx_freqs,
        f,
        h,
        w,
        topk=topk,
        kv_len=kv_len,
        is_stream=True,
        local_range=local_range,
    )

    y_bf, bf_ck, bf_cv = mx_blk_bf16(
        mx.array(x_pt.numpy()).astype(mx.bfloat16),
        None,
        mx.array(t_mod_pt.numpy()).astype(mx.bfloat16),
        mx_freqs,
        f,
        h,
        w,
        topk=topk,
        kv_len=kv_len,
        is_stream=True,
        local_range=local_range,
    )

    max_abs, rel_max, cos = _parity_metrics(_to_np(y_mx), y_pt.numpy())
    _, rel_ck, _ = _parity_metrics(_to_np(mx_ck), pt_ck.numpy())
    _, rel_cv, _ = _parity_metrics(_to_np(mx_cv), pt_cv.numpy())
    _, _, cos_bf16 = _parity_metrics(_to_np(y_bf), y_pt.numpy())
    print(
        f"[DiTBlock-1.3B fp32/bf16] "
        f"out(max={max_abs:.3e}, rel={rel_max:.3e}, cos={cos:.6f}) "
        f"ck(rel={rel_ck:.3e}) cv(rel={rel_cv:.3e}) bf16_cos={cos_bf16:.6f}"
    )
    assert rel_max <= 1e-4
    assert rel_ck <= 1e-4
    assert rel_cv <= 1e-4
    assert cos_bf16 >= 0.999
