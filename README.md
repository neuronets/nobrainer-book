# Nobrainer Book

Tutorials and guides for [Nobrainer](https://github.com/neuronets/nobrainer) —
a deep learning framework for 3D brain image processing.

## Tutorials

Follow these tutorials in order for a complete learning journey:

| # | Tutorial | What you'll learn |
|---|----------|-------------------|
| 01 | Getting Started | Install, import, explore available models |
| 02 | Download & Inspect | Get sample brain MRI data, understand NIfTI format |
| 03 | Extract & Batch | Prepare training data with patches and datasets |
| 04 | Brain Segmentation | Train a UNet in 3 lines with the estimator API |
| 05 | Uncertainty | Bayesian inference with variance and entropy maps |
| 06 | Brain Generation | Synthesize brain volumes with Progressive GAN |
| 07 | Advanced Training | Custom PyTorch loops for full control |
| 08 | Model Management | Save/load with Croissant-ML metadata |
| 09 | Zarr Pipeline | Cloud-optimized multi-resolution storage |
| 10 | Multi-GPU | Scale training across GPUs with DDP |
| 11 | Contributing | Add models, write tests, submit PRs |

## Run in Google Colab

**Stable release** (master):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuronets/nobrainer-book/blob/master)

**Pre-release** (alpha):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuronets/nobrainer-book/blob/alpha)

## Branches

| Branch | Nobrainer version | Install |
|--------|-------------------|---------|
| master | Stable | `pip install nobrainer` |
| alpha | Pre-release | `pip install --pre nobrainer` |

## Run locally

```bash
uv venv --python 3.14
source .venv/bin/activate
uv pip install "nobrainer[bayesian,generative,zarr]" monai pyro-ppl nilearn matplotlib
for script in docs/nobrainer-guides/scripts/[01]*.py; do python "$script"; done
```
