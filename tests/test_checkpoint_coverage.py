# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hermes Reisner. Part of the MLX port of FlashVSR (https://github.com/OpenImagingLab/FlashVSR).

import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

import mlx_flashvsr as mf

_CKPT_KEYS_PATH = Path(__file__).resolve().parent / "checkpoint_keys_v1.1.json"


def test_checkpoint_keys_v1_1_coverage():
    """Verify full 30-layer WanModel and Causal_LQ4x_Proj key/shape parity without materializing weights."""
    data = json.loads(_CKPT_KEYS_PATH.read_text())
    dit_spec = data["dit"]
    lq_spec = data["LQ_proj_in.ckpt"]

    assert len(dit_spec) == 825
    assert len(lq_spec) == 8

    # Construct full-size models lazily; MLX defers buffer allocation until mx.eval().
    dit = mf.WanModel(**mf.WAN_1_3B_CONFIG)
    lq_proj = mf.Causal_LQ4x_Proj(in_dim=3, out_dim=mf.WAN_1_3B_CONFIG["dim"], layer_num=1)

    dit_params = dict(tree_flatten(dit.parameters()))
    lq_params = dict(tree_flatten(lq_proj.parameters()))

    assert len(dit_params) == len(dit_spec)
    assert len(lq_params) == len(lq_spec)

    mapped_dit = {mf.map_pytorch_key(k): (k, tuple(shape)) for k, (shape, _) in dit_spec.items()}
    assert set(mapped_dit.keys()) == set(dit_params.keys())
    for mlx_key, (pt_key, pt_shape) in mapped_dit.items():
        expected_shape = mf.map_pytorch_shape(pt_key, pt_shape)
        assert tuple(dit_params[mlx_key].shape) == expected_shape, (
            f"Shape mismatch for {pt_key} -> {mlx_key}: "
            f"got {tuple(dit_params[mlx_key].shape)}, expected {expected_shape}"
        )

    mapped_lq = {mf.map_pytorch_key(k): (k, tuple(shape)) for k, (shape, _) in lq_spec.items()}
    assert set(mapped_lq.keys()) == set(lq_params.keys())
    for mlx_key, (pt_key, pt_shape) in mapped_lq.items():
        expected_shape = mf.map_pytorch_shape(pt_key, pt_shape)
        assert tuple(lq_params[mlx_key].shape) == expected_shape, (
            f"Shape mismatch for {pt_key} -> {mlx_key}: "
            f"got {tuple(lq_params[mlx_key].shape)}, expected {expected_shape}"
        )

    # Exercise convert_state_dict + load_torch_state_dict(strict=True) using 0-overhead broadcast placeholders.
    dit_placeholders = {
        k: mx.broadcast_to(mx.zeros((1,), dtype=mx.float32), tuple(shape))
        for k, (shape, _) in dit_spec.items()
    }
    lq_placeholders = {
        k: mx.broadcast_to(mx.zeros((1,), dtype=mx.float32), tuple(shape))
        for k, (shape, _) in lq_spec.items()
    }

    mf.load_torch_state_dict(dit, dit_placeholders, dtype="float32", strict=True)
    mf.load_torch_state_dict(lq_proj, lq_placeholders, dtype="float32", strict=True)

    # Confirm lazy construction kept active MLX device memory tiny (< 50 MB).
    assert mx.get_active_memory() < 50 * 1024 * 1024
