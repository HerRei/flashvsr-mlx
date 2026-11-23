# FlashVSR for Apple Silicon

An MLX port of [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) v1.1, the one-step diffusion model for
video super-resolution by OpenImagingLab. It runs the Tiny pipeline (4x) on Macs with Apple Silicon and uses the
official v1.1 weights unchanged.

This is an unofficial port. It is not made or reviewed by the FlashVSR authors.

## What it does

FlashVSR upscales a video 4x in one diffusion step. It streams the video in chunks of frames, keeps a cache of
earlier frames, and limits attention to nearby regions (locality-constrained sparse attention). This port
reimplements the Tiny pipeline in MLX: the Wan 2.1 1.3B transformer with the sparse attention and the streaming
cache, the low-resolution projector, the TCDecoder and the colour correction. The sparse attention runs in a
Metal kernel that only computes the blocks the mask keeps.

Not included: training, the Full pipeline with the Wan VAE, and FlashVSR+.

## Requirements

- A Mac with Apple Silicon and macOS 14 or newer (the oldest system MLX 0.32 supports).
- Python 3.10 or newer.
- Enough memory for the output size: on a 16 GB Mac, 896x512 output peaked at 12.2 GB and 768x1408 at
  17.2 GB, which swaps (see the table below).

Video is read and written through imageio-ffmpeg, which brings its own ffmpeg.

## Install

```sh
pip install git+https://github.com/HerRei/flashvsr-mlx
```

Download the three Tiny weight files (about 6.4 GB) from the official Hugging Face repository
[JunhaoZhuang/FlashVSR-v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1):

```sh
hf download JunhaoZhuang/FlashVSR-v1.1 diffusion_pytorch_model_streaming_dmd.safetensors \
    LQ_proj_in.ckpt TCDecoder.ckpt --local-dir models/FlashVSR-v1.1
```

## Usage

```sh
flashvsr-mlx input.mp4 output.mp4 --model-dir models/FlashVSR-v1.1
```

The input is prepared like the official `infer_flashvsr_v1.1_tiny.py`: each frame is upscaled 4x and
centre-cropped to a multiple of 128 pixels, and the frame count is padded to the form 8n+1. The output has the
input's frame rate and no audio. Options: `--dtype bfloat16|float16`, `--max-frames N`, `--scale`, `--seed`,
`--sparse-ratio`, `--kv-ratio`, `--local-range` and `--no-color-fix`; `flashvsr-mlx --help` explains them.

From Python, `FlashVSRTinyPipeline.from_pretrained("models/FlashVSR-v1.1")` loads the model;
`mlx_flashvsr/cli.py` shows how a video is prepared and passed to it.

## Speed and memory

Measured on a MacBook Pro with M1 Pro and 16 GB, bfloat16, inference only (loading the weights takes about 11 s
more, and the first chunk includes compiling the Metal kernels):

| Output size | Frames | Frames per second | Peak memory footprint |
|---|---|---|---|
| 512x256 | 61 | 1.78 | 11.7 GB |
| 896x512 | 29 | 0.45 | 12.2 GB |
| 768x1408 | 29 | 0.17 | 17.2 GB |

For comparison, the authors report about 17 FPS at 768x1408 on an NVIDIA A100.

## How the port is checked

The tests compare every part with the original PyTorch code. The reference is the upstream repository with the
ROCm pull request [#114](https://github.com/OpenImagingLab/FlashVSR/pull/114), pinned as the git submodule
`third_party/FlashVSR`, whose float32 reference attention computes the sparse attention exactly. With random
weights, float32 results match within 1e-4 relative error and bfloat16 results reach a cosine similarity of at
least 0.999, for:

- the building blocks, the sparse-attention masks (exact match) and the transformer block over three streaming
  steps with its cache;
- the whole transformer, the low-resolution projector with its caches and the streaming loop;
- the decoder with its temporal state, the colour correction and the full pipeline to pixels;
- the Metal kernel against float32 attention for many mask densities, including rows with no blocks.

With the real weights, a 512x256 clip was compared with the upstream pipeline run on a GPU with the same input
and noise: the outputs agree at 44.0 dB PSNR, and against the ground truth the port scores 23.68 dB and the
original 23.67 dB.

Run the tests with:

```sh
git clone --recursive https://github.com/HerRei/flashvsr-mlx
cd flashvsr-mlx
pip install -e ".[test]"
pytest tests
```

The real-weight test is skipped unless `models/FlashVSR-v1.1` and the reference outputs in `results_mlx/` exist;
the reference comes from the upstream pipeline on a CUDA or ROCm GPU.

## Sparse attention on Metal

FlashVSR splits the tokens into windows of 128 and, for each window, keeps only the most relevant windows nearby.
The kernel in `mlx_flashvsr/block_sparse_attn.py` turns the block mask into a list of kept key blocks per query
block and visits only those: 32 queries per threadgroup, keys in steps of 16, 8x8 simdgroup matrix operations and
a float32 online softmax, in 12.75 KB of threadgroup memory. One attention call is 2.2 to 2.9 times faster than
dense attention with a mask; at small output sizes nearly all blocks are kept, so the gain grows with the size.

## Limits

- 4x only, Tiny pipeline only; width and height become multiples of 128.
- Large outputs need more memory than a 16 GB Mac has. Spatial tiling is not implemented yet.
- Only tested on an M1 Pro.

## Licence and credits

Apache License 2.0, like FlashVSR. See `LICENSE` and `NOTICE`.

- FlashVSR, its code, the v1.1 weights and the prompt embedding in `mlx_flashvsr/assets/`: Junhao Zhuang and
  co-authors, OpenImagingLab (Apache-2.0).
- Wan 2.1 1.3B, the base model: Wan team, Alibaba (Apache-2.0).
- DiffSynth-Studio, whose model code FlashVSR builds on: ModelScope (Apache-2.0).
- TCDecoder: based on [TAEHV](https://github.com/madebyollin/taehv) by Ollin Boer Bohan (MIT, text in
  `LICENSES/TAEHV-MIT.txt`).
- The Metal kernel adapts parts of MLX's attention kernels, Apple (MIT, text in `LICENSES/MLX-MIT.txt`).
- ROCm reference path used by the tests: [flowerinheart](https://github.com/flowerinheart), upstream PR #114.
