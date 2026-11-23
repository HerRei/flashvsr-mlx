# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hermes Reisner. Part of the MLX port of FlashVSR (https://github.com/OpenImagingLab/FlashVSR).

import sys
import types
from pathlib import Path

import mlx.core as mx
import pytest
from einops import rearrange

REPO_ROOT = Path(__file__).resolve().parents[1]
# The upstream PyTorch code is the reference: a pinned checkout (git submodule) of FlashVSR with PR #114,
# whose diffsynth.models.rocm_attention.reference_attention computes the sparse attention exactly.
UPSTREAM_ROOT = REPO_ROOT / "third_party" / "FlashVSR"
for _p in (REPO_ROOT, UPSTREAM_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _ensure_package_stub(name: str, path: Path):
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        pkg.__package__ = name
        sys.modules[name] = pkg


def _install_reference_shims():
    """Install test-only shims so upstream diffsynth and utils modules import and run on macOS CPU."""
    # Bypass diffsynth/__init__.py and diffsynth/models/__init__.py (which pull unused deps like imageio).
    _ensure_package_stub("diffsynth", UPSTREAM_ROOT / "diffsynth")
    _ensure_package_stub("diffsynth.models", UPSTREAM_ROOT / "diffsynth" / "models")
    _ensure_package_stub("diffsynth.pipelines", UPSTREAM_ROOT / "diffsynth" / "pipelines")
    _ensure_package_stub("diffsynth.schedulers", UPSTREAM_ROOT / "diffsynth" / "schedulers")

    examples_wan_dir = str(UPSTREAM_ROOT / "examples" / "WanVSR")
    if examples_wan_dir not in sys.path:
        sys.path.insert(0, examples_wan_dir)

    if "PIL" not in sys.modules:
        try:
            import PIL  # noqa: F401
        except ModuleNotFoundError:
            pil_pkg = types.ModuleType("PIL")
            pil_image = types.ModuleType("PIL.Image")
            pil_pkg.Image = pil_image
            sys.modules["PIL"] = pil_pkg
            sys.modules["PIL.Image"] = pil_image

    if "tqdm" not in sys.modules:
        try:
            import tqdm  # noqa: F401
        except ModuleNotFoundError:
            class _NoopTqdm:
                def __init__(self, iterable=None, *args, **kwargs):
                    self._iterable = iterable if iterable is not None else ()

                def __iter__(self):
                    return iter(self._iterable)

                def update(self, n=1):
                    pass

                def close(self):
                    pass

            tqdm_mod = types.ModuleType("tqdm")
            tqdm_mod.__path__ = []
            tqdm_mod.tqdm = _NoopTqdm
            tqdm_auto = types.ModuleType("tqdm.auto")
            tqdm_auto.tqdm = _NoopTqdm
            sys.modules["tqdm"] = tqdm_mod
            sys.modules["tqdm.auto"] = tqdm_auto

    if "torchvision" not in sys.modules:
        try:
            import torchvision  # noqa: F401
        except ModuleNotFoundError:
            tv_pkg = types.ModuleType("torchvision")
            tv_transforms = types.ModuleType("torchvision.transforms")
            tv_transforms.GaussianBlur = object
            tv_pkg.transforms = tv_transforms
            sys.modules["torchvision"] = tv_pkg
            sys.modules["torchvision.transforms"] = tv_transforms

    if "block_sparse_attn" not in sys.modules:
        stub = types.ModuleType("block_sparse_attn")

        def _unpatched(*args, **kwargs):
            raise RuntimeError("block_sparse_attn_func should not be called directly under test shim")

        stub.block_sparse_attn_func = _unpatched
        sys.modules["block_sparse_attn"] = stub

    diffsynth_models = sys.modules["diffsynth.models"]
    if not hasattr(diffsynth_models, "ModelManager"):
        diffsynth_models.ModelManager = object
    if "diffsynth.models.wan_video_vae" not in sys.modules:
        vae_stub = types.ModuleType("diffsynth.models.wan_video_vae")
        vae_stub.WanVideoVAE = object
        vae_stub.RMS_norm = object
        vae_stub.CausalConv3d = object
        vae_stub.Upsample = object
        sys.modules["diffsynth.models.wan_video_vae"] = vae_stub

    import diffsynth.models.rocm_attention as rocm_attention
    import diffsynth.models.wan_video_dit as wan_dit

    if not getattr(wan_dit, "_mlx_test_patched", False):
        orig_flash_attention = wan_dit.flash_attention

        def _flash_attention_ref(
            q,
            k,
            v,
            num_heads,
            compatibility_mode=False,
            attention_mask=None,
            return_KV=False,
        ):
            if attention_mask is not None:
                q_h = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
                k_h = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
                v_h = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
                out = rocm_attention.reference_attention(q_h, k_h, v_h, attention_mask)
                return rearrange(out, "b n s d -> b s (n d)", n=num_heads)
            return orig_flash_attention(
                q,
                k,
                v,
                num_heads=num_heads,
                compatibility_mode=compatibility_mode,
                attention_mask=None,
                return_KV=return_KV,
            )

        wan_dit.flash_attention = _flash_attention_ref
        wan_dit._mlx_test_patched = True

    import diffsynth.pipelines.flashvsr_tiny as flashvsr_tiny
    from utils import TCDecoder as pt_tcdecoder
    from utils.utils import Causal_LQ4x_Proj

    return wan_dit, rocm_attention, flashvsr_tiny, Causal_LQ4x_Proj, pt_tcdecoder


WAN_DIT, ROCM_ATTN, FLASHVSR_TINY, PT_CAUSAL_LQ4X_PROJ, PT_TCDECODER = _install_reference_shims()


@pytest.fixture(scope="session")
def wan_dit():
    return WAN_DIT


@pytest.fixture(scope="session")
def rocm_attn():
    return ROCM_ATTN


@pytest.fixture(scope="session")
def flashvsr_tiny():
    return FLASHVSR_TINY


@pytest.fixture(scope="session")
def pt_lq_proj_cls():
    return PT_CAUSAL_LQ4X_PROJ


@pytest.fixture(scope="session")
def pt_tcdecoder_mod():
    return PT_TCDECODER


@pytest.fixture(scope="session", autouse=True)
def verify_mlx_gpu():
    mx.set_default_device(mx.gpu)
    assert mx.default_device() == mx.Device(mx.gpu)
