# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hermes Reisner. Part of the MLX port of FlashVSR (https://github.com/OpenImagingLab/FlashVSR).

import gc
import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_flashvsr import attention as mf_attn
from mlx_flashvsr.block_sparse_attn import (
    block_sparse_attention_metal,
    compact_block_mask,
)
from mlx_flashvsr.masks import (
    build_local_block_mask_shifted_vec_normal_slide,
    generate_draft_block_mask,
)


@pytest.fixture(autouse=True)
def _cleanup_mlx_cache():
    yield
    gc.collect()
    mx.clear_cache()


def _to_np_f32(x: mx.array) -> np.ndarray:
    mx.eval(x)
    return np.asarray(x.astype(mx.float32))


def _metrics(actual: np.ndarray, expected: np.ndarray):
    diff = np.abs(actual - expected)
    max_abs = float(np.max(diff))
    out_scale = max(float(np.max(np.abs(expected))), 1e-8)
    rel_max = max_abs / out_scale
    a = actual.reshape(-1).astype(np.float64)
    e = expected.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(e))
    cosine = float(np.dot(a, e) / denom) if denom > 0 else 1.0
    return max_abs, rel_max, cosine


def _numpy_block_sparse_ref(
    q_np: np.ndarray,
    k_np: np.ndarray,
    v_np: np.ndarray,
    mask_np: np.ndarray,
) -> np.ndarray:
    b, h, l_q, d = q_np.shape
    l_kv = k_np.shape[2]
    n_qb, n_kb = mask_np.shape[2], mask_np.shape[3]
    tok_mask = np.broadcast_to(
        mask_np[:, :, :, None, :, None], (b, h, n_qb, 128, n_kb, 128)
    ).reshape(b, h, l_q, l_kv)
    scores = np.matmul(q_np, np.swapaxes(k_np, -1, -2)) / math.sqrt(d)
    scores = np.where(tok_mask, scores, -np.inf)
    row_max = np.max(scores, axis=-1, keepdims=True)
    row_has = np.any(tok_mask, axis=-1, keepdims=True)
    safe_max = np.where(row_has, row_max, 0.0)
    exp_s = np.where(tok_mask, np.exp(scores - safe_max), 0.0)
    denom = np.sum(exp_s, axis=-1, keepdims=True)
    probs = np.where(row_has, exp_s / np.maximum(denom, 1e-30), 0.0)
    return np.matmul(probs, v_np).astype(np.float32)


def test_compact_block_mask():
    rng = np.random.default_rng(123)
    m_np = rng.random((2, 3, 5, 7)) > 0.5
    m_np[0, 1, 2, :] = False
    m_np[1, 0, 0, :] = True
    sel, cnt = compact_block_mask(mx.array(m_np))
    mx.eval(sel, cnt)
    sel_np = np.asarray(sel)
    cnt_np = np.asarray(cnt)
    assert np.array_equal(cnt_np, m_np.sum(axis=-1))
    for b in range(2):
        for h in range(3):
            for qb in range(5):
                c = int(cnt_np[b, h, qb])
                expected = np.where(m_np[b, h, qb])[0]
                assert np.array_equal(sel_np[b, h, qb, :c], expected)


def test_kernel_input_assertions():
    q = mx.zeros((1, 2, 128, 128), dtype=mx.bfloat16)
    k = mx.zeros((1, 2, 128, 128), dtype=mx.bfloat16)
    v = mx.zeros((1, 2, 128, 128), dtype=mx.bfloat16)
    m = mx.ones((1, 2, 1, 1), dtype=mx.bool_)

    with pytest.raises(AssertionError):
        block_sparse_attention_metal(mx.zeros((1, 2, 129, 128), dtype=mx.bfloat16), k, v, mx.ones((1, 2, 2, 1), dtype=mx.bool_))
    with pytest.raises(AssertionError):
        block_sparse_attention_metal(q, mx.zeros((1, 2, 130, 128), dtype=mx.bfloat16), mx.zeros((1, 2, 130, 128), dtype=mx.bfloat16), mx.ones((1, 2, 1, 2), dtype=mx.bool_))
    with pytest.raises(AssertionError):
        block_sparse_attention_metal(mx.zeros((1, 2, 128, 64), dtype=mx.bfloat16), k, v, m)
    with pytest.raises(AssertionError):
        block_sparse_attention_metal(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), m)
    with pytest.raises(AssertionError):
        block_sparse_attention_metal(q, k.astype(mx.float16), v.astype(mx.float16), m)
    with pytest.raises(ValueError):
        block_sparse_attention_metal(q, k, v, mx.ones((1, 2, 1, 2), dtype=mx.bool_))


def test_kernel_vs_numpy_fp32_reference():
    rng = np.random.default_rng(7)
    b, h, l_q, l_kv, d = 2, 2, 256, 384, 128
    q_np = rng.standard_normal((b, h, l_q, d)).astype(np.float32)
    k_np = rng.standard_normal((b, h, l_kv, d)).astype(np.float32)
    v_np = rng.standard_normal((b, h, l_kv, d)).astype(np.float32)
    mask_np = rng.random((b, h, l_q // 128, l_kv // 128)) > 0.45
    mask_np[0, 1, 0, :] = False
    mask_np[1, 0, 1, :] = False

    for dtype, tol in ((mx.bfloat16, 2e-2), (mx.float16, 5e-3)):
        q_mx = mx.array(q_np).astype(dtype)
        k_mx = mx.array(k_np).astype(dtype)
        v_mx = mx.array(v_np).astype(dtype)
        mask_mx = mx.array(mask_np)

        # Evaluate float32 reference on the actual quantized inputs as well as raw fp32 inputs.
        ref_np = _numpy_block_sparse_ref(
            _to_np_f32(q_mx), _to_np_f32(k_mx), _to_np_f32(v_mx), mask_np
        )
        out_metal = _to_np_f32(mf_attn.block_sparse_attention(q_mx, k_mx, v_mx, mask_mx, backend="metal"))
        out_dense = _to_np_f32(mf_attn.block_sparse_attention(q_mx, k_mx, v_mx, mask_mx, backend="dense"))

        assert np.all(np.isfinite(out_metal))
        assert np.max(np.abs(out_metal[0, 1, :128])) == 0.0
        assert np.max(np.abs(out_metal[1, 0, 128:256])) == 0.0

        max_abs, rel_max, cos = _metrics(out_metal, ref_np)
        _, rel_dense, cos_dense = _metrics(out_metal, out_dense)
        print(
            f"[Kernel vs NumPy fp32 ({dtype})] max_abs={max_abs:.3e} "
            f"rel={rel_max:.3e} cos={cos:.6f} (vs dense rel={rel_dense:.3e} cos={cos_dense:.6f})"
        )
        assert rel_max <= tol
        assert rel_dense <= tol


@pytest.mark.parametrize("dtype,tol", [(mx.bfloat16, 2e-2), (mx.float16, 5e-3)])
@pytest.mark.parametrize(
    "case_name,shape,density,force_empty",
    [
        ("single_block", (1, 2, 128, 128, 128), 1.0, False),
        ("all_empty", (1, 2, 256, 256, 128), 0.0, True),
        ("sparse_5pct", (1, 4, 384, 512, 128), 0.05, True),
        ("medium_50pct", (2, 3, 256, 512, 128), 0.50, True),
        ("dense_95pct", (1, 4, 256, 384, 128), 0.95, False),
        ("512x256_f6", (1, 12, 3072, 3072, 128), 0.50, False),
        ("512x256_f2", (1, 12, 1024, 4096, 128), 0.50, True),
        ("768x1408_f2_h2", (1, 2, 8448, 33792, 128), 0.50, True),
    ],
)
def test_kernel_shapes_and_densities(dtype, tol, case_name, shape, density, force_empty):
    b, h, l_q, l_kv, d = shape
    mx.random.seed(42)
    q = mx.random.normal((b, h, l_q, d)).astype(dtype)
    k = mx.random.normal((b, h, l_kv, d)).astype(dtype)
    v = mx.random.normal((b, h, l_kv, d)).astype(dtype)

    n_qb, n_kb = l_q // 128, l_kv // 128
    rng = np.random.default_rng(100 + l_q + l_kv)
    if density <= 0.0:
        mask_np = np.zeros((b, h, n_qb, n_kb), dtype=bool)
    elif density >= 1.0:
        mask_np = np.ones((b, h, n_qb, n_kb), dtype=bool)
    else:
        mask_np = rng.random((b, h, n_qb, n_kb)) < density
    if force_empty:
        mask_np[:, 0, 0, :] = False
    mask = mx.array(mask_np)
    mx.eval(q, k, v, mask)

    out_metal = mf_attn.block_sparse_attention(q, k, v, mask, backend="metal")
    out_dense = mf_attn.block_sparse_attention(q, k, v, mask, backend="dense")
    ref_fp32 = mf_attn.block_sparse_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), mask, backend="dense"
    )
    mx.eval(out_metal, out_dense, ref_fp32)

    out_m_np = _to_np_f32(out_metal)
    out_d_np = _to_np_f32(out_dense)
    ref_np = _to_np_f32(ref_fp32)

    assert np.all(np.isfinite(out_m_np))
    if force_empty:
        assert np.max(np.abs(out_m_np[:, 0, :128])) == 0.0

    max_abs, rel_max, cos = _metrics(out_m_np, ref_np)
    _, rel_dense, cos_dense = _metrics(out_m_np, out_d_np)
    print(
        f"[{case_name} {dtype}] max_abs={max_abs:.3e} rel={rel_max:.3e} "
        f"cos={cos:.6f} (vs dense rel={rel_dense:.3e} cos={cos_dense:.6f})"
    )
    assert rel_max <= tol
    assert rel_dense <= tol

    del q, k, v, mask, out_metal, out_dense, ref_fp32
    gc.collect()
    mx.clear_cache()


@pytest.mark.parametrize("dtype,tol", [(mx.bfloat16, 2e-2), (mx.float16, 5e-3)])
@pytest.mark.parametrize(
    "name,bh,bw,f_q,f_kv,nheads,topk_ratio",
    [
        ("512x256_f6_local", 2, 4, 6, 6, 12, 2.0 * 768 * 1280 / (256 * 512)),
        ("512x256_f2_local", 2, 4, 2, 8, 12, 2.0 * 768 * 1280 / (256 * 512)),
        ("896x512_f2_local", 4, 7, 2, 8, 12, 2.0 * 768 * 1280 / (512 * 896)),
        ("768x1408_f2_h2_local", 11, 6, 2, 8, 2, 2.0 * 768 * 1280 / (1408 * 768)),
    ],
)
def test_kernel_local_window_masks(dtype, tol, name, bh, bw, f_q, f_kv, nheads, topk_ratio):
    mx.random.seed(77)
    d = 128
    dim = nheads * d
    seqlen = f_q // 2
    n_qb = seqlen * bh * bw
    n_kb = (f_kv // 2) * bh * bw

    q_w = mx.random.normal((n_qb, 128, dim)).astype(dtype)
    k_w = mx.random.normal((n_kb, 128, dim)).astype(dtype)
    v_w = mx.random.normal((n_kb, 128, dim)).astype(dtype)

    local_mask = build_local_block_mask_shifted_vec_normal_slide(bh, bw, 11, 11, include_self=True)
    square_num = bh * bw
    topk = max(1, int(square_num * topk_ratio) - 1)
    mask = generate_draft_block_mask(
        1, nheads, seqlen, q_w, k_w, topk=topk, local_attn_mask=local_mask
    )

    q_h = q_w.reshape(1, n_qb * 128, nheads, d).transpose(0, 2, 1, 3)
    k_h = k_w.reshape(1, n_kb * 128, nheads, d).transpose(0, 2, 1, 3)
    v_h = v_w.reshape(1, n_kb * 128, nheads, d).transpose(0, 2, 1, 3)
    mx.eval(q_h, k_h, v_h, mask)

    out_metal = mf_attn.block_sparse_attention(q_h, k_h, v_h, mask, backend="metal")
    out_dense = mf_attn.block_sparse_attention(q_h, k_h, v_h, mask, backend="dense")
    ref_fp32 = mf_attn.block_sparse_attention(
        q_h.astype(mx.float32), k_h.astype(mx.float32), v_h.astype(mx.float32), mask, backend="dense"
    )
    mx.eval(out_metal, out_dense, ref_fp32)

    max_abs, rel_max, cos = _metrics(_to_np_f32(out_metal), _to_np_f32(ref_fp32))
    _, rel_dense, _ = _metrics(_to_np_f32(out_metal), _to_np_f32(out_dense))
    density = float(mx.mean(mask.astype(mx.float32)).item())
    print(
        f"[{name} {dtype} density={density:.2%}] max_abs={max_abs:.3e} "
        f"rel={rel_max:.3e} cos={cos:.6f} (vs dense rel={rel_dense:.3e})"
    )
    assert rel_max <= tol
    assert rel_dense <= tol

    del q_w, k_w, v_w, q_h, k_h, v_h, mask, out_metal, out_dense, ref_fp32
    gc.collect()
    mx.clear_cache()


def test_default_backend_is_metal():
    assert mf_attn._BLOCK_SPARSE_BACKEND == "metal"
