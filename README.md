# FlashVSR for Apple Silicon

An MLX port of [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) v1.1, the one-step diffusion model for
video super-resolution by OpenImagingLab. It runs the Tiny pipeline (4x) on Macs with Apple Silicon and uses the
official v1.1 weights unchanged.

This is an unofficial port. It is not made or reviewed by the FlashVSR authors.

## What it does

FlashVSR upscales a video 4x in one diffusion step. It streams the video in chunks of frames, keeps a cache of
earlier frames, and limits attention to nearby regions (locality-constrained sparse attention). This port
reimplements the Tiny pipeline in MLX: the Wan 2.1 1.3B transformer with the sparse attention and the streaming
cache, the low-resolution projector, the TCDecoder and the colour correction.

Not included: training, the Full pipeline with the Wan VAE, and FlashVSR+.

## Requirements

- A Mac with Apple Silicon. [fill: minimum macOS supported by the MLX version in pyproject.toml]
- Python 3.10 or newer.
- [fill: RAM needed per output size, from the measurements below]
- ffmpeg for reading and writing video. [fill: confirm once the CLI's video I/O is settled]

## Install

```sh
pip install git+https://github.com/[fill: owner]/flashvsr-mlx
```

The weights (about 6.4 GB) come from the official Hugging Face repository
[JunhaoZhuang/FlashVSR-v1.1](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1):

```sh
[fill: download command, e.g. huggingface-cli download JunhaoZhuang/FlashVSR-v1.1 ...]
```

## Usage

```sh
[fill: CLI example from task 3b]
```

[fill: short Python API example]

## Speed and memory

Measured on [fill: Mac model, RAM, macOS], [fill: clip length]:

| Output size | Frames per second | Peak memory |
|---|---|---|
| [fill] | [fill] | [fill] |

For comparison, the authors report about 17 FPS at 768x1408 on an NVIDIA A100.

## How the port is checked

Every part is compared with the original PyTorch code. The tests run the upstream modules on the CPU in float32 as
the reference (with the sparse attention computed exactly by gathering the allowed blocks) and the MLX modules on
the GPU, with the same random weights:

- building blocks, the sparse-attention masks (exact match) and one transformer block, also over three streaming
  steps with the cache;
- the whole model, the low-resolution projector with its caches, and the streaming loop up to the final latents;
- [fill: decoder, colour correction and the full pipeline to pixels (task 3a)];
- [fill: real weights end to end against PyTorch (task 3b)].

float32 results match within 1e-4 relative error; bfloat16 results reach a cosine similarity of at least 0.999.
Run the tests with:

```sh
[fill: test command and the upstream checkout they need]
```

## Sparse attention on Metal

[fill: how the block-sparse kernel works and what it saves (task 4)]

## Licence and credits

Apache License 2.0, like FlashVSR. See `LICENSE` and `NOTICE`.

- FlashVSR, its code and the v1.1 weights: Junhao Zhuang and co-authors, OpenImagingLab (Apache-2.0).
- Wan 2.1 1.3B, the base model: Wan team, Alibaba (Apache-2.0).
- DiffSynth-Studio, whose model code FlashVSR builds on: ModelScope (Apache-2.0).
- TCDecoder: based on [TAEHV](https://github.com/madebyollin/taehv) by Ollin Boer Bohan (MIT, text in
  `LICENSES/TAEHV-MIT.txt`).
