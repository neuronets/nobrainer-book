# Nobrainer Book

Tutorials and guides for the [Nobrainer](https://github.com/neuronets/nobrainer)
deep learning framework for 3D brain image processing.

## Branches

| Branch | Nobrainer version | Install command |
|--------|-------------------|-----------------|
| **master** | Latest stable release | `uv pip install nobrainer` |
| **alpha** | Latest pre-release | `uv pip install --pre nobrainer` |

## Tutorials

| # | Tutorial | Description |
|---|----------|-------------|
| 01 | Getting Started | Installation, module overview, available models |
| 02 | Preparing Training Data | Loading brain MRI, extracting patches, building DataLoaders |
| 03 | Train Brain Extraction | UNet and MeshNet segmentation with real brain data |
| 04 | Train Brain Generation | Progressive GAN with PyTorch Lightning |
| 05 | Training with Augmentation | On-the-fly data augmentation (noise, flips) |
| 06 | Train with Checkpoints | Model checkpointing and training resumption |

## Run in Google Colab

**Stable release** (master branch):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuronets/nobrainer-book/blob/master)

**Pre-release** (alpha branch) — for testing upcoming features:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuronets/nobrainer-book/blob/alpha)

## Run locally

```bash
uv venv --python 3.14
source .venv/bin/activate
uv pip install nobrainer monai pyro-ppl nilearn

for script in docs/nobrainer-guides/scripts/0*.py; do
    python "$script"
done
```

## Release workflow

The **alpha** branch tutorials are validated automatically before each
nobrainer alpha release. If any tutorial fails, the release is blocked.
Once an alpha is promoted to a stable release on master, the nobrainer-book
master branch is updated to match.
