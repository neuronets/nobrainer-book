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
# # Zarr v3 for Brain Data
#
# [Zarr](https://zarr.dev/) is a chunked, compressed array format ideal for
# large neuroimaging datasets. Nobrainer supports Zarr v3 with sharding,
# enabling efficient cloud-based and partial-read access. This tutorial
# covers:
#
# 1. Converting NIfTI to Zarr v3 (via `.nii.gz` intermediate)
# 2. Inspecting the Zarr store structure
# 3. Round-tripping back to NIfTI
# 4. Using Zarr stores with `PatchDataset`

# %%
PRE_RELEASE = False
import subprocess, sys
try:
    import google.colab  # noqa: F401
    cmd = [sys.executable, "-m", "pip", "install", "-q",
           "nobrainer", "nilearn", "matplotlib", "zarr"]
    if PRE_RELEASE:
        cmd.insert(4, "--pre")
    subprocess.check_call(cmd)
except ImportError:
    pass

# %% [markdown]
# ## 1. Prepare: convert .mgz to .nii.gz
#
# Nobrainer's sample data is in FreeSurfer MGZ format. The `nifti_to_zarr()`
# function expects NIfTI input, so we first convert using nibabel.

# %%
import csv
import os
import tempfile
import nibabel as nib
import numpy as np
from nobrainer.utils import get_data

csv_path = get_data()
with open(csv_path) as f:
    reader = csv.reader(f)
    next(reader)
    filepaths = [(row[0], row[1]) for row in reader]

work_dir = tempfile.mkdtemp(prefix="nobrainer_zarr_")

# Convert the first subject's T1 and labels from MGZ to NIfTI
feat_mgz, label_mgz = filepaths[0]

feat_nii_path = os.path.join(work_dir, "sub-01_t1.nii.gz")
label_nii_path = os.path.join(work_dir, "sub-01_label.nii.gz")

feat_img = nib.load(feat_mgz)
nib.save(nib.Nifti1Image(np.asarray(feat_img.dataobj), feat_img.affine), feat_nii_path)

label_img = nib.load(label_mgz)
nib.save(nib.Nifti1Image(np.asarray(label_img.dataobj), label_img.affine), label_nii_path)

print("Converted to NIfTI:")
print(f"  Feature: {feat_nii_path}")
print(f"  Label:   {label_nii_path}")

# %% [markdown]
# ## 2. Convert NIfTI to Zarr v3
#
# `nifti_to_zarr()` creates a sharded Zarr v3 store with optional
# multi-resolution levels. Setting `levels=2` creates the full resolution
# (level 0) and a 2x-downsampled version (level 1).

# %%
from nobrainer.io import nifti_to_zarr

feat_zarr_path = os.path.join(work_dir, "sub-01_t1.zarr")
label_zarr_path = os.path.join(work_dir, "sub-01_label.zarr")

nifti_to_zarr(feat_nii_path, feat_zarr_path, chunk_shape=(64, 64, 64), levels=2)
nifti_to_zarr(label_nii_path, label_zarr_path, chunk_shape=(64, 64, 64), levels=2)

print("Zarr stores created:")
print(f"  Feature: {feat_zarr_path}")
print(f"  Label:   {label_zarr_path}")

# %% [markdown]
# ## 3. Inspect the Zarr store structure

# %%
import zarr

store = zarr.open_group(feat_zarr_path, mode="r")

print("Zarr group contents:")
print(f"  Group attrs: {dict(store.attrs)}")
print()

# List arrays at each resolution level
for key in sorted(store.keys()):
    arr = store[key]
    print(f"  Level '{key}':")
    print(f"    Shape: {arr.shape}")
    print(f"    Dtype: {arr.dtype}")
    print(f"    Chunks: {arr.chunks}")

# %% [markdown]
# ### Provenance metadata
#
# `nifti_to_zarr()` stores provenance in the group attributes, including the
# source file name, creation timestamp, chunk shape, and nobrainer version.

# %%
import json

provenance = dict(store.attrs).get("nobrainer_provenance", {})
print("Provenance:")
print(json.dumps(provenance, indent=2))

# %% [markdown]
# ## 4. Round-trip: Zarr back to NIfTI
#
# `zarr_to_nifti()` reconstructs a NIfTI file from a Zarr store.
# You can select which resolution level to export.

# %%
from nobrainer.io import zarr_to_nifti

roundtrip_path = os.path.join(work_dir, "sub-01_t1_roundtrip.nii.gz")
zarr_to_nifti(feat_zarr_path, roundtrip_path, level=0)

# Verify the round-trip
original = nib.load(feat_nii_path).get_fdata()
roundtrip = nib.load(roundtrip_path).get_fdata()

print("Original shape:", original.shape)
print("Round-trip shape:", roundtrip.shape)
print("Max absolute difference:", np.abs(original - roundtrip).max())
print("Arrays match:", np.allclose(original, roundtrip, atol=1e-5))

# %% [markdown]
# ## 5. Use Zarr stores with PatchDataset
#
# The `PatchDataset` can read patches directly from Zarr stores, loading
# only the chunks that overlap the requested patch. This is memory-efficient
# and works well for cloud-hosted data.

# %%
from nobrainer.processing.dataset import PatchDataset
from torch.utils.data import DataLoader

zarr_data = [
    {"image": feat_zarr_path, "label": label_zarr_path},
]

patch_ds = PatchDataset(
    data=zarr_data,
    block_shape=(16, 16, 16),
    patches_per_volume=5,
    binarize=True,
)

loader = DataLoader(patch_ds, batch_size=2, shuffle=True)

print(f"PatchDataset: {len(patch_ds)} patches")
print()

# Fetch a batch
batch = next(iter(loader))
print("Batch image shape:", batch["image"].shape)
print("Batch label shape:", batch["label"].shape)
print("Image value range: [{:.2f}, {:.2f}]".format(
    batch["image"].min().item(), batch["image"].max().item()))

# %% [markdown]
# ## 6. Zarr advantages for neuroimaging
#
# | Feature | NIfTI | Zarr v3 |
# |---------|-------|---------|
# | Partial reads | No (loads full volume) | Yes (chunk-level) |
# | Cloud storage | Requires full download | Native S3/GCS support |
# | Multi-resolution | Separate files | Built-in levels |
# | Compression | gzip only | blosc, zlib, zstd, etc. |
# | Sharding | N/A | Groups chunks into shards |
# | Metadata | Fixed NIfTI header | Extensible JSON attributes |

# %% [markdown]
# ## Cleanup

# %%
import shutil
shutil.rmtree(work_dir)
print("Temporary files cleaned up")

# %% [markdown]
# ## Summary
#
# Zarr v3 provides efficient, cloud-ready storage for brain volumes.
# Nobrainer's `nifti_to_zarr()` / `zarr_to_nifti()` handle conversion,
# and `PatchDataset` reads patches directly from Zarr stores with minimal
# memory overhead. In the next tutorial we will discuss multi-GPU training.
