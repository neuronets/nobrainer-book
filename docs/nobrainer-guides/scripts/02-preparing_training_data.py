# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
# ---

# %% [markdown]
# # Preparing training data
#
# In this tutorial, we will download sample brain MRI data and prepare it for
# training a PyTorch model. We load volumes with nibabel, extract random 3-D
# patches, binarize the labels, and build a standard PyTorch DataLoader. This
# DataLoader can then be used for training, evaluation, or prediction.
#
# This tutorial will use a small publicly available dataset. To use your own
# data, you will need to create a CSV file (or nested list) of feature and label
# volume paths (see below).


# %%
# Set to True on the alpha branch to install pre-release versions
PRE_RELEASE = False

import subprocess
import sys

try:
    import google.colab  # noqa: F401
    cmd = [sys.executable, "-m", "pip", "install", "-q",
           "nobrainer", "nilearn", "matplotlib"]
    if PRE_RELEASE:
        cmd.insert(4, "--pre")
    subprocess.check_call(cmd)
except ImportError:
    pass  # Not on Colab; install manually with: uv pip install nobrainer
# %%
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nobrainer.io import read_csv
from nobrainer.utils import get_data


# %% [markdown]
# ## Get sample data
#
# Here, we download 10 T1-weighted brain scans and their corresponding
# FreeSurfer segmentations. These volumes take up about 46 MB and are saved to
# a temporary directory. The returned string `csv_path` is the path to a CSV
# file, each row of which contains the paths to a pair of features and labels
# volumes.

# %%
csv_path = get_data()
filepaths = read_csv(csv_path)
print(f"Number of subjects: {len(filepaths)}")
print(f"First pair: {filepaths[0]}")


# %% [markdown]
# ### Visualize one training example, with the brainmask overlayed on the
# anatomical image.

# %%
import matplotlib.pyplot as plt
from nilearn import plotting

fig = plt.figure(figsize=(12, 6))
plotting.plot_roi(
    filepaths[0][1],
    bg_img=filepaths[0][0],
    alpha=0.4,
    vmin=0,
    vmax=1.5,
    figure=fig,
)


# %% [markdown]
# ## Load volumes and extract random patches
#
# The raw volumes are 256^3 .mgz files. For fast CPU training, we extract
# small random 32^3 patches from each volume and binarize the labels
# (0 = background, 1 = brain).

# %%
BLOCK = 32
N_PATCHES_PER_VOL = 4


def extract_random_patches(img_path, label_path, block=BLOCK, n_patches=N_PATCHES_PER_VOL):
    """Load a volume pair and extract random cubic patches."""
    vol = np.asarray(nib.load(img_path).dataobj, dtype=np.float32)
    lab = np.asarray(nib.load(label_path).dataobj, dtype=np.int64)
    # Binarize: any label > 0 is brain
    lab = (lab > 0).astype(np.int64)

    patches_x, patches_y = [], []
    for _ in range(n_patches):
        starts = [np.random.randint(0, max(s - block, 1)) for s in vol.shape[:3]]
        sl = tuple(slice(s, s + block) for s in starts)
        patches_x.append(vol[sl])
        patches_y.append(lab[sl])
    return patches_x, patches_y


# %%
np.random.seed(42)
train_pairs = filepaths[:9]
eval_pair = filepaths[9]

all_x, all_y = [], []
for img_path, label_path in train_pairs:
    px, py = extract_random_patches(img_path, label_path)
    all_x.extend(px)
    all_y.extend(py)

x_train = torch.from_numpy(np.stack(all_x)[:, None])  # (N, 1, D, H, W)
y_train = torch.from_numpy(np.stack(all_y))  # (N, D, H, W)
print(f"Training patches: x={x_train.shape}, y={y_train.shape}")


# %% [markdown]
# ## Create a PyTorch DataLoader
#
# We wrap the patches in a `TensorDataset` and `DataLoader` for batched,
# shuffled iteration during training.

# %%
BATCH_SIZE = 4
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Number of batches per epoch: {len(train_loader)}")
print(f"Batch size: {BATCH_SIZE}")

# Inspect one batch
for xb, yb in train_loader:
    print(f"Batch features shape: {xb.shape}")
    print(f"Batch labels shape:   {yb.shape}")
    break

# %% [markdown]
# You are now ready to train a model using this example dataset!
# Next, learn how to
# [train a brain extraction model](https://colab.research.google.com/github/neuronets/nobrainer-book/blob/master/docs/nobrainer-guides/notebooks/03-train_brain_extraction.ipynb)
