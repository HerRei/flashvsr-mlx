# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hermes Reisner. Part of the MLX port of FlashVSR (https://github.com/OpenImagingLab/FlashVSR).

import math

import mlx.core as mx
import numpy as np
import pytest
import torch

import mlx_flashvsr as mf
from mlx_flashvsr import attention as mf_attn


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


def _torch_freqs_3d_to_mlx(freqs_torch: torch.Tensor) -> mx.array:
    # freqs_torch: complex tensor of shape (S, 1, D//2)
    real = freqs_torch.real.to(torch.float32).cpu().numpy()
    imag = freqs_torch.imag.to(torch.float32).cpu().numpy()
    return mx.array(np.stack([real, imag], axis=-1), dtype=mx.float32)


def test_rope_3d_and_apply(wan_dit):
    torch.manual_seed(0)
    head_dim = 128
    num_heads = 2
    dim = head_dim * num_heads
    f, h, w = 6, 12, 16

    pt_freqs_3d = wan_dit.precompute_freqs_cis_3d(head_dim, end=64)
    mx_freqs_3d = mf.precompute_freqs_cis_3d(head_dim, end=64)

    for idx in range(3):
        pt_c = np.stack(
            [pt_freqs_3d[idx].real.numpy(), pt_freqs_3d[idx].imag.numpy()], axis=-1
        ).astype(np.float32)
        mx_c = _to_np(mx_freqs_3d[idx])
        max_abs, rel_max, _ = _parity_metrics(mx_c, pt_c)
        assert rel_max <= 1e-4, f"precompute_freqs_cis[{idx}] rel_max={rel_max}"

    pt_freqs = torch.cat(
        [
            pt_freqs_3d[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            pt_freqs_3d[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            pt_freqs_3d[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1)
    mx_freqs = mf.build_freqs_3d(mx_freqs_3d, f, h, w)

    x_pt = torch.randn(1, f * h * w, dim, dtype=torch.float32)
    y_pt = wan_dit.rope_apply(x_pt, pt_freqs, num_heads)
    y_mx = mf.rope_apply(mx.array(x_pt.numpy()), mx_freqs, num_heads)

    max_abs, rel_max, cos = _parity_metrics(_to_np(y_mx), y_pt.numpy())
    print(f"[RoPE fp32] max_abs={max_abs:.3e} rel={rel_max:.3e} cos={cos:.6f}")
    assert rel_max <= 1e-4

    y_mx_bf16 = mf.rope_apply(mx.array(x_pt.numpy()).astype(mx.bfloat16), mx_freqs, num_heads)
    _, _, cos_bf16 = _parity_metrics(_to_np(y_mx_bf16), y_pt.numpy())
    print(f"[RoPE bf16] cos={cos_bf16:.6f}")
    assert cos_bf16 >= 0.999


def test_window_partition_3d(wan_dit):
    torch.manual_seed(1)
    win = (2, 8, 8)
    x_pt = torch.randn(1, 6, 32, 48, 256, dtype=torch.float32)
    w_pt = wan_dit.WindowPartition3D.partition(x_pt, win)
    r_pt = wan_dit.WindowPartition3D.reverse(w_pt, win, (6, 32, 48))

    x_mx = mx.array(x_pt.numpy())
    w_mx = mf.WindowPartition3D.partition(x_mx, win)
    r_mx = mf.WindowPartition3D.reverse(w_mx, win, (6, 32, 48))

    assert np.array_equal(_to_np(w_mx), w_pt.numpy())
    assert np.array_equal(_to_np(r_mx), r_pt.numpy())
    assert np.array_equal(_to_np(r_mx), x_pt.numpy())


@pytest.mark.parametrize("bh,bw,win_h,win_w,include_self", [
    (12, 12, 9, 9, True),   # h=w=96 tokens -> 12x12 block grid with local_range=9
    (12, 12, 9, 9, False),
    (4, 5, 3, 3, True),
])
def test_local_block_masks(wan_dit, bh, bw, win_h, win_w, include_self):
    m_pt = wan_dit.build_local_block_mask_shifted_vec_normal_slide(
        bh, bw, win_h, win_w, include_self=include_self
    )
    m_mx = mf.build_local_block_mask_shifted_vec_normal_slide(
        bh, bw, win_h, win_w, include_self=include_self
    )
    assert np.array_equal(_to_np(m_mx), m_pt.numpy())

    # Also verify non-sliding clipped variant when grid >= win
    if bh >= win_h and bw >= win_w:
        m2_pt = wan_dit.build_local_block_mask_shifted_vec(
            bh, bw, win_h, win_w, include_self=include_self
        )
        m2_mx = mf.build_local_block_mask_shifted_vec(
            bh, bw, win_h, win_w, include_self=include_self
        )
        assert np.array_equal(_to_np(m2_mx), m2_pt.numpy())


@pytest.mark.parametrize("force_grouped", [False, True])
@pytest.mark.parametrize("ties", [False, True])
def test_generate_draft_block_mask(wan_dit, force_grouped, ties):
    # h = w = 96 -> 12x12 block grid, local_range = 9 (local window strictly smaller than grid)
    bh, bw, local_range = 12, 12, 9
    f_q, f_kv = 6, 8  # seqlen=3 for Q, 4 chunks for KV
    seqlen = f_q // 2
    n_q_blocks = (f_q // 2) * bh * bw
    n_kv_blocks = (f_kv // 2) * bh * bw
    nheads = 2
    dim = 256

    torch.manual_seed(42)
    if ties:
        q_w = torch.ones(n_q_blocks, 128, dim, dtype=torch.float32)
        k_w = torch.ones(n_kv_blocks, 128, dim, dtype=torch.float32)
    else:
        q_w = torch.randn(n_q_blocks, 128, dim, dtype=torch.float32)
        k_w = torch.randn(n_kv_blocks, 128, dim, dtype=torch.float32)

    local_pt = wan_dit.build_local_block_mask_shifted_vec_normal_slide(
        bh, bw, local_range, local_range, include_self=True
    )
    local_mx = mf.build_local_block_mask_shifted_vec_normal_slide(
        bh, bw, local_range, local_range, include_self=True
    )

    topk = (bh * bw) * 6  # actively prunes within the local window
    mask_pt = wan_dit.generate_draft_block_mask(
        1,
        nheads,
        seqlen,
        q_w,
        k_w,
        topk=topk,
        local_attn_mask=local_pt,
        _force_grouped=force_grouped,
    )
    mask_mx = mf.generate_draft_block_mask(
        1,
        nheads,
        seqlen,
        mx.array(q_w.numpy()),
        mx.array(k_w.numpy()),
        topk=topk,
        local_attn_mask=local_mx,
        _force_grouped=force_grouped,
    )

    assert _to_np(mask_mx).dtype == np.bool_
    assert np.array_equal(_to_np(mask_mx), mask_pt.numpy())
    if not ties:
        # Confirm both locality and topk actually pruned blocks.
        assert 0 < int(mask_pt.sum().item()) < mask_pt.numel()


def test_block_sparse_attention_and_empty_rows(rocm_attn, monkeypatch):
    torch.manual_seed(7)
    for nq, nk in ((256, 384), (384, 384), (129, 257)):
        q_pt = torch.randn(1, 2, nq, 128, dtype=torch.float32)
        k_pt = torch.randn(1, 2, nk, 128, dtype=torch.float32)
        v_pt = torch.randn(1, 2, nk, 128, dtype=torch.float32)
        mask_pt = torch.rand(1, 2, (nq + 127) // 128, (nk + 127) // 128) > 0.4
        mask_pt[:, 0, 0] = False  # force an all-False block row

        out_pt = rocm_attn.reference_attention(q_pt, k_pt, v_pt, mask_pt)
        out_mx = mf.block_sparse_attention(
            mx.array(q_pt.numpy()),
            mx.array(k_pt.numpy()),
            mx.array(v_pt.numpy()),
            mx.array(mask_pt.numpy()),
        )
        out_mx_np = _to_np(out_mx)
        assert np.all(np.isfinite(out_mx_np))
        assert np.max(np.abs(out_mx_np[:, 0, :128])) == 0.0
        max_abs, rel_max, cos = _parity_metrics(out_mx_np, out_pt.numpy())
        print(f"[BlockSparseAttn ({nq},{nk}) fp32] max_abs={max_abs:.3e} rel={rel_max:.3e} cos={cos:.6f}")
        assert rel_max <= 1e-4

        out_mx_bf16 = mf.block_sparse_attention(
            mx.array(q_pt.numpy()).astype(mx.bfloat16),
            mx.array(k_pt.numpy()).astype(mx.bfloat16),
            mx.array(v_pt.numpy()).astype(mx.bfloat16),
            mx.array(mask_pt.numpy()),
        )
        out_bf16_np = _to_np(out_mx_bf16)
        assert np.all(np.isfinite(out_bf16_np))
        assert np.max(np.abs(out_bf16_np[:, 0, :128])) == 0.0
        _, _, cos_bf16 = _parity_metrics(out_bf16_np, out_pt.numpy())
        assert cos_bf16 >= 0.999

    # Force head-group + q-block chunking path and verify exact parity.
    monkeypatch.setattr(mf_attn, "_MAX_MASK_ELEMENTS", 128 * 384)
    q_pt = torch.randn(1, 4, 384, 128, dtype=torch.float32)
    k_pt = torch.randn(1, 4, 384, 128, dtype=torch.float32)
    v_pt = torch.randn(1, 4, 384, 128, dtype=torch.float32)
    mask_pt = torch.rand(1, 4, 3, 3) > 0.5
    mask_pt[:, 2, 1] = False
    out_pt = rocm_attn.reference_attention(q_pt, k_pt, v_pt, mask_pt)
    out_mx = mf.block_sparse_attention(
        mx.array(q_pt.numpy()),
        mx.array(k_pt.numpy()),
        mx.array(v_pt.numpy()),
        mx.array(mask_pt.numpy()),
    )
    max_abs, rel_max, _ = _parity_metrics(_to_np(out_mx), out_pt.numpy())
    print(f"[BlockSparseAttn chunked fp32] max_abs={max_abs:.3e} rel={rel_max:.3e}")
    assert rel_max <= 1e-4


def test_rmsnorm_and_gatemodule(wan_dit):
    torch.manual_seed(11)
    dim = 256
    pt_norm = wan_dit.RMSNorm(dim, eps=1e-6).eval()
    with torch.no_grad():
        pt_norm.weight.normal_()
    mx_norm = mf.RMSNorm(dim, eps=1e-6)
    mf.load_torch_state_dict(mx_norm, pt_norm.state_dict(), dtype="float32")

    x_pt = torch.randn(1, 64, dim, dtype=torch.float32)
    with torch.no_grad():
        y_pt = pt_norm(x_pt)
    y_mx = mx_norm(mx.array(x_pt.numpy()))
    max_abs, rel_max, cos = _parity_metrics(_to_np(y_mx), y_pt.numpy())
    print(f"[RMSNorm fp32] max_abs={max_abs:.3e} rel={rel_max:.3e} cos={cos:.6f}")
    assert rel_max <= 1e-4

    mx_norm_bf16 = mf.RMSNorm(dim, eps=1e-6)
    mf.load_torch_state_dict(mx_norm_bf16, pt_norm.state_dict(), dtype="bfloat16")
    y_mx_bf16 = mx_norm_bf16(mx.array(x_pt.numpy()).astype(mx.bfloat16))
    _, _, cos_bf16 = _parity_metrics(_to_np(y_mx_bf16), y_pt.numpy())
    print(f"[RMSNorm bf16] cos={cos_bf16:.6f}")
    assert cos_bf16 >= 0.999

    pt_gate = wan_dit.GateModule()
    mx_gate = mf.GateModule()
    gate_pt = torch.randn(1, 1, dim)
    res_pt = torch.randn(1, 64, dim)
    g_pt = pt_gate(x_pt, gate_pt, res_pt)
    g_mx = mx_gate(mx.array(x_pt.numpy()), mx.array(gate_pt.numpy()), mx.array(res_pt.numpy()))
    max_abs, rel_max, _ = _parity_metrics(_to_np(g_mx), g_pt.numpy())
    assert rel_max <= 1e-4


@pytest.mark.parametrize("has_pos_emb", [False, True])
def test_mlp(wan_dit, has_pos_emb):
    torch.manual_seed(13)
    in_dim, out_dim = (1280, 256) if has_pos_emb else (256, 512)
    pt_mlp = wan_dit.MLP(in_dim, out_dim, has_pos_emb=has_pos_emb).eval()
    if has_pos_emb:
        with torch.no_grad():
            pt_mlp.emb_pos.normal_(std=0.05)

    mx_mlp = mf.MLP(in_dim, out_dim, has_pos_emb=has_pos_emb)
    mf.load_torch_state_dict(mx_mlp, pt_mlp.state_dict(), dtype="float32")

    seq = 514 if has_pos_emb else 32
    x_pt = torch.randn(1, seq, in_dim, dtype=torch.float32)
    with torch.no_grad():
        y_pt = pt_mlp(x_pt)
    y_mx = mx_mlp(mx.array(x_pt.numpy()))
    max_abs, rel_max, cos = _parity_metrics(_to_np(y_mx), y_pt.numpy())
    print(f"[MLP pos={has_pos_emb} fp32] max_abs={max_abs:.3e} rel={rel_max:.3e} cos={cos:.6f}")
    assert rel_max <= 1e-4

    mx_mlp_bf16 = mf.MLP(in_dim, out_dim, has_pos_emb=has_pos_emb)
    mf.load_torch_state_dict(mx_mlp_bf16, pt_mlp.state_dict(), dtype="bfloat16")
    y_mx_bf16 = mx_mlp_bf16(mx.array(x_pt.numpy()).astype(mx.bfloat16))
    _, _, cos_bf16 = _parity_metrics(_to_np(y_mx_bf16), y_pt.numpy())
    print(f"[MLP pos={has_pos_emb} bf16] cos={cos_bf16:.6f}")
    assert cos_bf16 >= 0.999


def test_head(wan_dit):
    torch.manual_seed(17)
    dim, out_dim, patch_size = 256, 16, (1, 2, 2)
    pt_head = wan_dit.Head(dim, out_dim, patch_size, eps=1e-6).eval()
    mx_head = mf.Head(dim, out_dim, patch_size, eps=1e-6)
    mf.load_torch_state_dict(mx_head, pt_head.state_dict(), dtype="float32")

    x_pt = torch.randn(1, 64, dim, dtype=torch.float32)
    t_pt = torch.randn(1, dim, dtype=torch.float32)
    with torch.no_grad():
        y_pt = pt_head(x_pt, t_pt)
    y_mx = mx_head(mx.array(x_pt.numpy()), mx.array(t_pt.numpy()))
    max_abs, rel_max, cos = _parity_metrics(_to_np(y_mx), y_pt.numpy())
    print(f"[Head fp32] max_abs={max_abs:.3e} rel={rel_max:.3e} cos={cos:.6f}")
    assert rel_max <= 1e-4

    mx_head_bf16 = mf.Head(dim, out_dim, patch_size, eps=1e-6)
    mf.load_torch_state_dict(mx_head_bf16, pt_head.state_dict(), dtype="bfloat16")
    y_mx_bf16 = mx_head_bf16(
        mx.array(x_pt.numpy()).astype(mx.bfloat16),
        mx.array(t_pt.numpy()).astype(mx.bfloat16),
    )
    _, _, cos_bf16 = _parity_metrics(_to_np(y_mx_bf16), y_pt.numpy())
    print(f"[Head bf16] cos={cos_bf16:.6f}")
    assert cos_bf16 >= 0.999


def test_cross_attention(wan_dit):
    torch.manual_seed(19)
    dim, num_heads = 256, 2
    pt_ca = wan_dit.CrossAttention(dim, num_heads, eps=1e-6).eval()
    mx_ca = mf.CrossAttention(dim, num_heads, eps=1e-6)
    mf.load_torch_state_dict(mx_ca, pt_ca.state_dict(), dtype="float32")

    ctx_pt = torch.randn(1, 77, dim, dtype=torch.float32)
    x_pt = torch.randn(1, 128, dim, dtype=torch.float32)
    with torch.no_grad():
        pt_ca.init_cache(ctx_pt)
        y_pt = pt_ca(x_pt, None, is_stream=True)

    mx_ca.init_cache(mx.array(ctx_pt.numpy()))
    y_mx = mx_ca(mx.array(x_pt.numpy()), None, is_stream=True)
    max_abs, rel_max, cos = _parity_metrics(_to_np(y_mx), y_pt.numpy())
    print(f"[CrossAttention fp32] max_abs={max_abs:.3e} rel={rel_max:.3e} cos={cos:.6f}")
    assert rel_max <= 1e-4

    mx_ca.clear_cache()
    assert mx_ca.cache_k is None and mx_ca.cache_v is None

    mx_ca_bf16 = mf.CrossAttention(dim, num_heads, eps=1e-6)
    mf.load_torch_state_dict(mx_ca_bf16, pt_ca.state_dict(), dtype="bfloat16")
    mx_ca_bf16.init_cache(mx.array(ctx_pt.numpy()).astype(mx.bfloat16))
    y_mx_bf16 = mx_ca_bf16(mx.array(x_pt.numpy()).astype(mx.bfloat16), None, is_stream=True)
    _, _, cos_bf16 = _parity_metrics(_to_np(y_mx_bf16), y_pt.numpy())
    print(f"[CrossAttention bf16] cos={cos_bf16:.6f}")
    assert cos_bf16 >= 0.999


def test_converter_dtypes_and_strictness(wan_dit):
    torch.manual_seed(23)
    pt_block = wan_dit.DiTBlock(256, 2, 512, eps=1e-6)
    sd_fp32 = pt_block.state_dict()
    sd_bf16 = {k: v.to(torch.bfloat16) for k, v in sd_fp32.items()}

    for target_dtype, mx_dtype in (
        ("float32", mx.float32),
        ("float16", mx.float16),
        ("bfloat16", mx.bfloat16),
    ):
        blk = mf.DiTBlock(256, 2, 512, eps=1e-6)
        mf.load_torch_state_dict(blk, sd_fp32, dtype=target_dtype, strict=True)
        conv = mf.convert_state_dict(sd_bf16, dtype=target_dtype)
        assert all(v.dtype == mx_dtype for v in conv.values())

    bad_sd = dict(sd_fp32)
    del bad_sd["self_attn.q.weight"]
    with pytest.raises(KeyError):
        mf.load_torch_state_dict(mf.DiTBlock(256, 2, 512), bad_sd, strict=True)
