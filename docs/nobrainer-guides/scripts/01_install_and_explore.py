# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Getting Started with Nobrainer
#
# Nobrainer is a deep learning framework for neuroimaging built on PyTorch.
# This tutorial covers installation verification, checking hardware
# availability, listing built-in models, and importing key processing modules.

# %%
PRE_RELEASE = False
import subprocess, sys
try:
    import google.colab  # noqa: F401
    cmd = [sys.executable, "-m", "pip", "install", "-q",
           "nobrainer", "nilearn", "matplotlib"]
    if PRE_RELEASE:
        cmd.insert(4, "--pre")
    subprocess.check_call(cmd)
except ImportError:
    pass

# %% [markdown]
# ## 1. Import nobrainer and check the version

# %%
import nobrainer

print("nobrainer version:", nobrainer.__version__)

# %% [markdown]
# ## 2. Check PyTorch and device availability
#
# Nobrainer uses PyTorch as its backend. Training is faster on GPU, but all
# tutorials work on CPU as well.
#
# `nobrainer.training.get_device()` automatically selects the best available
# device in priority order: **CUDA > MPS (Apple Silicon) > CPU**.

# %%
import torch
from nobrainer.training import get_device

print("PyTorch version:", torch.__version__)
device = get_device()
print("Selected device:", device)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("Apple MPS (Metal) GPU available")
else:
    print("Running on CPU (this is fine for tutorials)")

# %% [markdown]
# ## 3. List available models
#
# Nobrainer ships with several model architectures for segmentation,
# generation, and self-supervised learning. Some models require optional
# dependencies (`pyro-ppl` for Bayesian models, `pytorch-lightning` for
# generative models).

# %%
from nobrainer.models import available_models, list_available_models

print("Available models:")
list_available_models()

# %% [markdown]
# You can retrieve a model factory by name:

# %%
from nobrainer.models import get as get_model

unet_factory = get_model("unet")
print("UNet factory:", unet_factory)

# %% [markdown]
# ## 4. Explore processing imports
#
# The `nobrainer.processing` module provides the high-level sklearn-style API.

# %%
from nobrainer.processing.segmentation import Segmentation
from nobrainer.processing.generation import Generation
from nobrainer.processing.dataset import Dataset, extract_patches

print("Segmentation:", Segmentation)
print("Generation:", Generation)
print("Dataset:", Dataset)
print("extract_patches:", extract_patches)

# %% [markdown]
# ## 5. Check optional dependencies
#
# Some features require optional packages. This cell reports what is available.

# %%
optional_deps = {
    "nibabel": "NIfTI I/O",
    "nilearn": "Neuroimaging visualization",
    "scipy": "Image processing utilities",
    "pyro": "Bayesian models (pyro-ppl)",
    "pytorch_lightning": "Generative model training",
    "zarr": "Zarr v3 data pipeline",
    "datalad": "Dataset versioning and OpenNeuro access",
}

for mod, description in optional_deps.items():
    try:
        __import__(mod)
        print(f"  [OK] {mod} -- {description}")
    except ImportError:
        print(f"  [--] {mod} -- {description} (not installed)")

# %% [markdown]
# ## Summary
#
# You now have a working nobrainer installation. In the next tutorial we will
# download sample brain data and begin exploring it.
