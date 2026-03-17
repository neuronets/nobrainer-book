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
# # Prepare Training Data
#
# Brain MRI volumes are large 3D arrays (typically 256x256x256). Training
# directly on full volumes is memory-intensive, so we extract smaller
# **patches** (sub-volumes). This tutorial covers three approaches:
#
# 1. Manual patch extraction with `extract_patches()`
# 2. Building a `Dataset` with the fluent API
# 3. Streaming mode for on-the-fly patch extraction

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
# ## Setup: download sample data

# %%
import csv
import nibabel as nib
import numpy as np
from nobrainer.utils import get_data

csv_path = get_data()
with open(csv_path) as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    filepaths = [(row[0], row[1]) for row in reader]

print(f"Loaded {len(filepaths)} subjects")
feature_path, label_path = filepaths[0]

# %% [markdown]
# ## 1. Manual patch extraction with `extract_patches()`
#
# This function extracts random patches from a 3D volume and optionally
# binarizes the labels.

# %%
from nobrainer.processing.dataset import extract_patches

vol = nib.load(feature_path).get_fdata()
lbl = nib.load(label_path).get_fdata()

print("Full volume shape:", vol.shape)
print("Full label shape:", lbl.shape)

# %% [markdown]
# ### 1a. Binary brain extraction (any tissue = 1)
#
# Setting `binarize=True` maps all non-zero labels to 1.

# %%
patches_binary = extract_patches(
    vol, lbl,
    block_shape=(16, 16, 16),
    n_patches=5,
    binarize=True,
)

img_patch, lbl_patch = patches_binary[0]
print("Image patch shape:", img_patch.shape)
print("Label patch shape:", lbl_patch.shape)
print("Unique binary labels:", np.unique(lbl_patch))

# %% [markdown]
# ### 1b. Select specific FreeSurfer regions
#
# Pass a set of label codes to `binarize` to isolate specific structures.
# Here we extract just the hippocampus (left=17, right=53).

# %%
patches_hippo = extract_patches(
    vol, lbl,
    block_shape=(16, 16, 16),
    n_patches=5,
    binarize={17, 53},
)

img_patch, lbl_patch = patches_hippo[0]
print("Hippocampus-only label unique values:", np.unique(lbl_patch))
print("Fraction of hippocampus voxels: {:.4f}".format(lbl_patch.mean()))

# %% [markdown]
# ## 2. Build a Dataset with the fluent API
#
# The `Dataset` class provides a chainable API to configure batching,
# binarization, augmentation, and more. It produces a PyTorch DataLoader.

# %%
from nobrainer.processing.dataset import Dataset

# Use a subset for speed
subset = filepaths[:3]

ds = (
    Dataset.from_files(subset, block_shape=(16, 16, 16), n_classes=2)
    .batch(2)
    .binarize()
)

print("Dataset info:")
print("  Number of files:", len(ds.data))
print("  Block shape:", ds.block_shape)
print("  Batch size:", ds.batch_size)
print("  Volume shape:", ds.volume_shape)

# %% [markdown]
# ### Iterate over a batch

# %%
loader = ds.dataloader
batch = next(iter(loader))

if isinstance(batch, dict):
    images, labels = batch["image"], batch["label"]
else:
    images, labels = batch[0], batch[1]

print("Batch images shape:", images.shape)
print("Batch labels shape:", labels.shape)
print("Image dtype:", images.dtype)
print("Label dtype:", labels.dtype)

# %% [markdown]
# ## 3. Streaming mode
#
# For large datasets or Zarr stores, streaming mode extracts patches
# directly from disk without loading full volumes into memory. This is
# especially efficient for cloud-hosted Zarr data.

# %%
ds_streaming = (
    Dataset.from_files(subset, block_shape=(16, 16, 16), n_classes=2)
    .batch(2)
    .binarize()
    .streaming(patches_per_volume=5)
)

print("Streaming dataset:")
print("  Patches per volume:", ds_streaming._patches_per_volume)
print("  Total patches per epoch:", len(ds_streaming.dataloader.dataset))

# Fetch one batch
batch_s = next(iter(ds_streaming.dataloader))
print("  Batch image shape:", batch_s["image"].shape)
print("  Batch label shape:", batch_s["label"].shape)

# %% [markdown]
# ## 4. Comparing approaches
#
# | Approach | Memory | Speed | Best for |
# |----------|--------|-------|----------|
# | `extract_patches()` | Low (manual) | Fast | Exploration, small datasets |
# | `Dataset.from_files()` | Medium | Fast | Standard training pipelines |
# | `Dataset.streaming()` | Very low | I/O bound | Large datasets, Zarr stores |
#
# For most training workflows, use `Dataset.from_files().batch().binarize()`.
# Switch to `.streaming()` when datasets are too large to fit in memory or
# when using Zarr v3 stores.

# %% [markdown]
# ## Summary
#
# We covered three ways to prepare brain data for training:
# - `extract_patches()` for manual control over patch extraction
# - `Dataset` fluent API for building standard training pipelines
# - Streaming mode for memory-efficient on-the-fly extraction
#
# In the next tutorial we will use these tools to train a brain segmentation
# model in just a few lines of code.
