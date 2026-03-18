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
# # Quantify Prediction Uncertainty
#
# Standard segmentation models output a single "best guess" label per voxel.
# Bayesian models instead sample multiple predictions, letting us quantify
# **uncertainty** -- how confident the model is at each voxel. This is
# critical for clinical applications where knowing *what the model does not
# know* matters as much as knowing what it predicts.
#
# This tutorial uses a Bayesian VNet with Pyro-ppl.

# %%
PRE_RELEASE = False
import subprocess, sys
try:
    import google.colab  # noqa: F401
    cmd = [sys.executable, "-m", "pip", "install", "-q",
           "nobrainer", "nilearn", "matplotlib", "pyro-ppl"]
    if PRE_RELEASE:
        cmd.insert(4, "--pre")
    subprocess.check_call(cmd)
except ImportError:
    pass

# %% [markdown]
# ## 1. Prepare data

# %%
import csv
from nobrainer.utils import get_data
from nobrainer.processing.dataset import Dataset

csv_path = get_data()
with open(csv_path) as f:
    reader = csv.reader(f)
    next(reader)
    filepaths = [(row[0], row[1]) for row in reader]

BLOCK_SHAPE = (16, 16, 16)

train_files = filepaths[:3]
eval_feature_path = filepaths[3][0]

ds = (
    Dataset.from_files(train_files, block_shape=BLOCK_SHAPE, n_classes=2)
    .batch(2)
    .binarize()
)

print("Dataset ready:", len(ds.data), "subjects")

# %% [markdown]
# ## 2. Train a Bayesian VNet
#
# The `bayesian_vnet` uses Pyro's stochastic weight layers. We keep
# the model tiny for this tutorial: `base_filters=4`, `levels=2`.

# %%
from nobrainer.processing.segmentation import Segmentation

seg = Segmentation(
    "bayesian_vnet",
    model_args={
        "in_channels": 1,
        "n_classes": 2,
        "base_filters": 4,
        "levels": 2,
    },
)

seg.fit(ds, epochs=2)
print("Bayesian VNet training complete!")

# %% [markdown]
# ## 3. Predict with uncertainty
#
# When `n_samples > 0`, the model runs multiple forward passes with
# different weight samples and returns three volumes:
#
# - **label**: the most frequent (mode) prediction across samples
# - **variance**: per-voxel variance across samples
# - **entropy**: Shannon entropy of the predictive distribution
#
# Higher variance or entropy means the model is less certain.

# %%
result = seg.predict(
    eval_feature_path,
    block_shape=BLOCK_SHAPE,
    n_samples=3,
)

# Unpack the result tuple
label_img, variance_img, entropy_img = result

print("Label shape:", label_img.shape)
print("Variance shape:", variance_img.shape)
print("Entropy shape:", entropy_img.shape)

# %% [markdown]
# ## 4. Examine uncertainty statistics

# %%
import numpy as np

var_data = np.asarray(variance_img.dataobj)
ent_data = np.asarray(entropy_img.dataobj)

print("Variance statistics:")
print(f"  Min:  {var_data.min():.6f}")
print(f"  Max:  {var_data.max():.6f}")
print(f"  Mean: {var_data.mean():.6f}")
print(f"  Std:  {var_data.std():.6f}")
print()
print("Entropy statistics:")
print(f"  Min:  {ent_data.min():.6f}")
print(f"  Max:  {ent_data.max():.6f}")
print(f"  Mean: {ent_data.mean():.6f}")
print(f"  Std:  {ent_data.std():.6f}")

# %% [markdown]
# ## 5. Interpreting uncertainty
#
# In a well-trained model, you would expect:
#
# - **Low uncertainty** in clearly brain or clearly background regions
# - **High uncertainty** at tissue boundaries (gray/white matter interface)
# - **High uncertainty** in ambiguous or pathological regions
#
# With our tiny model and 2 epochs of training, the uncertainty values
# will not be meaningful -- but the workflow is the same for production
# models.

# %%
# Percentage of voxels with above-average variance
high_var_pct = 100 * (var_data > var_data.mean()).sum() / var_data.size
print(f"Voxels with above-average variance: {high_var_pct:.1f}%")

# Percentage of voxels with above-average entropy
high_ent_pct = 100 * (ent_data > ent_data.mean()).sum() / ent_data.size
print(f"Voxels with above-average entropy: {high_ent_pct:.1f}%")

# %% [markdown]
# ## 6. Visualize uncertainty maps

# %%
import nibabel as nib
import matplotlib.pyplot as plt

feature_vol = np.asarray(nib.load(eval_feature_path).dataobj)
label_data = np.asarray(label_img.dataobj)
mid_slice = feature_vol.shape[2] // 2

plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.imshow(feature_vol[:, :, mid_slice].T, cmap="gray", origin="lower")
plt.title("Input volume")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(label_data[:, :, mid_slice].T, cmap="gray", origin="lower")
plt.title("Predicted label")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(var_data[:, :, mid_slice].T, cmap="hot", origin="lower")
plt.title("Variance map")
plt.colorbar()
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(ent_data[:, :, mid_slice].T, cmap="hot", origin="lower")
plt.title("Entropy map")
plt.colorbar()
plt.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# Bayesian models provide per-voxel uncertainty estimates via multiple
# stochastic forward passes. This is valuable for:
#
# - Flagging uncertain regions for expert review
# - Active learning (selecting the most informative samples to label)
# - Quality control in automated pipelines
#
# In the next tutorial we will explore synthetic brain generation with GANs.
