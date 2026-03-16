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
# # Train a neural network for binary volumetric brain extraction
#
# In this notebook, we will use the `nobrainer` Python API to train two models
# for brain extraction. Brain extraction is a common step in processing
# neuroimaging data. It is a voxel-wise, binary classification task, where each
# voxel is classified as brain or not brain.
#
# In the following cells, we will:
#
# 1. Get sample T1-weighted MR scans as features and FreeSurfer segmentations
#    as labels.
# 2. Load volumes and extract random 3-D patches for training.
# 3. Build a PyTorch DataLoader for batched iteration.
# 4. Instantiate a 3D convolutional neural network model (U-Net) for
#    segmentation.
# 5. Train on part of the data and evaluate on the rest.
# 6. Predict a brain mask using the trained model.
# 7. Save the model to disk for future prediction and/or training.
# 8. Load the model back from disk and show that brain extraction works as
#    before saving.
# 9. Demonstrate the same workflow using a different model (MeshNet).


# %% [markdown]
# ## Google Colaboratory
#
# If you are using Colab, please switch your runtime to GPU. To do this, select
# `Runtime > Change runtime type` in the top menu. Then select GPU under
# `Hardware accelerator`. A GPU greatly speeds up training.


# %% [markdown]
# # Install and setup `nobrainer`

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
import torch.nn as nn

from nobrainer.io import read_csv
from nobrainer.models.segmentation import unet
from nobrainer.models.meshnet import meshnet
from nobrainer.prediction import predict
from nobrainer.utils import get_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %% [markdown]
# # Get sample features and labels
#
# We use 9 pairs of volumes for training and 1 pair of volumes for evaluation.
# Many more volumes would be required to train a model for any useful purpose.

# %%
torch.manual_seed(42)
np.random.seed(42)

csv_path = get_data()
filepaths = read_csv(csv_path)
print(f"Downloaded {len(filepaths)} subject pairs")

train_pairs = filepaths[:9]
eval_pair = filepaths[9]


# %% [markdown]
# # Extract random patches for training
#
# The full volumes are ~256^3 which is too large for CPU training. We extract
# small random 32^3 patches and binarize labels (0 = background, 1 = brain).

# %%
BLOCK = 32
N_PATCHES_PER_VOL = 2


def extract_random_patches(img_path, label_path, block=BLOCK, n_patches=N_PATCHES_PER_VOL):
    """Load a volume pair and extract random cubic patches."""
    vol = np.asarray(nib.load(img_path).dataobj, dtype=np.float32)
    lab = np.asarray(nib.load(label_path).dataobj, dtype=np.int64)
    lab = (lab > 0).astype(np.int64)

    patches_x, patches_y = [], []
    for _ in range(n_patches):
        starts = [np.random.randint(0, max(s - block, 1)) for s in vol.shape[:3]]
        sl = tuple(slice(s, s + block) for s in starts)
        patches_x.append(vol[sl])
        patches_y.append(lab[sl])
    return patches_x, patches_y


all_x, all_y = [], []
for img_path, label_path in train_pairs:
    px, py = extract_random_patches(img_path, label_path)
    all_x.extend(px)
    all_y.extend(py)

x_train = torch.from_numpy(np.stack(all_x)[:, None])  # (N, 1, D, H, W)
y_train = torch.from_numpy(np.stack(all_y))  # (N, D, H, W)
print(f"Training patches: x={x_train.shape}, y={y_train.shape}")

# %%
from torch.utils.data import DataLoader, TensorDataset

BATCH_SIZE = 4
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# %% [markdown]
# # U-Net model for brain mask extraction

# %% [markdown]
# ## Construct the model
# Here we train `nobrainer`'s implementation of the U-Net model for biomedical
# image segmentation, based on https://arxiv.org/abs/1606.06650.
#
# `nobrainer` provides several other segmentation models that could be used
# instead of `unet`. Another example is provided at the bottom of this guide,
# and for a complete list, see
# [this list](https://github.com/neuronets/nobrainer#models).
#
# Note that a useful segmentation model would need to be trained on *many* more
# examples than the 10 we are using here for demonstration.

# %%
model = unet(n_classes=2, channels=(8, 16, 32), strides=(2, 2)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

n_params = sum(p.numel() for p in model.parameters())
print(f"UNet parameters: {n_params:,}")


# %% [markdown]
# ## Train the model on the example data
#
# Note that the loss function after training is still high and the accuracy is
# low -- this is expected since we are training briefly on very few small
# patches. During successful training of a more practical model, you would see
# the loss drop as training progressed.

# %%
N_EPOCHS = 2

model.train()
for epoch in range(N_EPOCHS):
    epoch_loss = 0.0
    n_batches = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    avg_loss = epoch_loss / max(n_batches, 1)
    print(f"Epoch {epoch + 1}/{N_EPOCHS}: loss={avg_loss:.4f}")


# %% [markdown]
# ## Use the trained model to predict a binary brain mask
# The segmentation is bad, but that is not surprising given the small dataset
# and short training.

# %%
import matplotlib.pyplot as plt
from nilearn import plotting

model.eval()
image_path = filepaths[0][0]

out = predict(
    inputs=image_path,
    model=model,
    block_shape=(32, 32, 32),
    batch_size=4,
    device=str(device),
    return_labels=True,
)
print(f"Prediction shape: {np.asarray(out.dataobj).shape}")

fig = plt.figure(figsize=(12, 6))
plotting.plot_roi(
    out,
    bg_img=image_path,
    cut_coords=(0, 10, -21),
    alpha=0.4,
    vmin=0,
    vmax=5,
    figure=fig,
)


# %% [markdown]
# ## Save the trained model

# %%
import os

os.makedirs("data", exist_ok=True)
torch.save(model.state_dict(), "data/unet-brainmask-toy.pth")
print("Model saved to data/unet-brainmask-toy.pth")


# %% [markdown]
# ## Load the model from disk

# %%
model_loaded = unet(n_classes=2, channels=(8, 16, 32), strides=(2, 2)).to(device)
model_loaded.load_state_dict(torch.load("data/unet-brainmask-toy.pth", weights_only=True))
model_loaded.eval()
print("Model loaded successfully")


# %% [markdown]
# ## Predict a brain mask from the loaded model
# The brain mask is identical to that predicted before saving.

# %%
out = predict(
    inputs=image_path,
    model=model_loaded,
    block_shape=(32, 32, 32),
    batch_size=4,
    device=str(device),
    return_labels=True,
)

fig = plt.figure(figsize=(12, 6))
plotting.plot_roi(
    out,
    bg_img=image_path,
    cut_coords=(0, 10, -21),
    alpha=0.4,
    vmin=0,
    vmax=5,
    figure=fig,
)


# %% [markdown]
# # Another model for brain mask extraction

# %% [markdown]
# ## Construct a MeshNet model
# Here we train `nobrainer`'s implementation of the MeshNet model for
# biomedical image segmentation, based on https://arxiv.org/abs/1612.00940.

# %%
mesh_model = meshnet(n_classes=2, filters=25).to(device)
mesh_optimizer = torch.optim.Adam(mesh_model.parameters(), lr=1e-3)
mesh_criterion = nn.CrossEntropyLoss()

n_params = sum(p.numel() for p in mesh_model.parameters())
print(f"MeshNet parameters: {n_params:,}")


# %% [markdown]
# ## Train the MeshNet model on the example data

# %%
mesh_model.train()
for epoch in range(N_EPOCHS):
    epoch_loss = 0.0
    n_batches = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        mesh_optimizer.zero_grad()
        pred = mesh_model(xb)
        loss = mesh_criterion(pred, yb)
        loss.backward()
        mesh_optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    avg_loss = epoch_loss / max(n_batches, 1)
    print(f"Epoch {epoch + 1}/{N_EPOCHS}: loss={avg_loss:.4f}")


# %% [markdown]
# ## Use the trained MeshNet model to predict a binary brain mask

# %%
mesh_model.eval()
out = predict(
    inputs=image_path,
    model=mesh_model,
    block_shape=(32, 32, 32),
    batch_size=4,
    device=str(device),
    return_labels=True,
)

fig = plt.figure(figsize=(12, 6))
plotting.plot_roi(
    out,
    bg_img=image_path,
    cut_coords=(0, 10, -21),
    alpha=0.4,
    vmin=0,
    vmax=5,
    figure=fig,
)


# %% [markdown]
# ## Save the MeshNet model

# %%
torch.save(mesh_model.state_dict(), "data/meshnet-brainmask-toy.pth")
print("MeshNet model saved to data/meshnet-brainmask-toy.pth")

# %% [markdown]
# Next, learn how to
# [train a brain volume generation model](https://colab.research.google.com/github/neuronets/nobrainer-book/blob/master/docs/nobrainer-guides/notebooks/04-train_brain_generation.ipynb)
