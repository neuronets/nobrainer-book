# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Training a segmentation model with data augmentation
#
# In this notebook, we will demonstrate how to train a brain mask extraction
# model using `nobrainer` with training data augmentation. Augmentation is
# useful for improving the robustness of neural network models when a limited
# amount of training examples are available. The basic idea is that, by applying
# certain transformations to the training data prior to fitting the model, one
# can expand the input space to cover situations unaccounted for in the base
# training set.
#
# Nobrainer provides several methods of augmenting volumetric data including
# spatial and intensity transforms.
#
# In the following cells, we will:
#
# 1. Get sample T1-weighted MR scans as features and FreeSurfer segmentations
#    as labels.
# 2. Load volumes and extract random 3-D patches.
# 3. Define augmentation transforms applied on-the-fly during training.
# 4. Instantiate a 3D convolutional neural network.
# 5. Train on part of the data.
# 6. Predict a brain mask on the held-out subject.

# %% [markdown]
# ## Google Colaboratory
#
# If you are using Colab, please switch your runtime to GPU. To do this, select
# `Runtime > Change runtime type` in the top menu. Then select GPU under
# `Hardware accelerator`. A GPU greatly speeds up training.

# %% [markdown]
# # Install and setup `nobrainer`

# %%
# !uv pip install --pre nobrainer nilearn

# %%
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from nobrainer.io import read_csv
from nobrainer.models.segmentation import unet
from nobrainer.prediction import predict
from nobrainer.utils import get_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# # Get sample features and labels

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


# %% [markdown]
# # Augmentation
#
# We define a custom PyTorch Dataset that applies augmentation transforms
# on-the-fly during training. Here we demonstrate two common augmentations:
#
# - **Gaussian noise**: adds random noise to the input volume
# - **Random flip**: randomly flips the volume along the left-right axis
#
# For training without augmentation, set `augment=False`.

# %%
class AugmentedDataset(Dataset):
    """Dataset with on-the-fly augmentation for 3-D patches."""

    def __init__(self, features, labels, augment=True, noise_mean=0.0, noise_std=0.1):
        self.features = features
        self.labels = labels
        self.augment = augment
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx].clone()
        y = self.labels[idx].clone()

        if self.augment:
            # Add Gaussian noise
            noise = torch.randn_like(x) * self.noise_std + self.noise_mean
            x = x + noise

            # Random left-right flip (along last axis)
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[-1])
                y = torch.flip(y, dims=[-1])

        return x, y


# %%
train_dataset = AugmentedDataset(
    x_train, y_train, augment=True, noise_mean=0.1, noise_std=0.5
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

print(f"Number of batches per epoch: {len(train_loader)}")


# %% [markdown]
# # Instantiate a neural network for brain mask extraction

# %%
model = unet(n_classes=2, channels=(8, 16, 32), strides=(2, 2)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# %% [markdown]
# # Train and evaluate the model

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
# Next, learn how to
# [train models using resumable checkpoints](https://colab.research.google.com/github/neuronets/nobrainer-book/blob/master/docs/nobrainer-guides/notebooks/06-train_with_checkpoints.ipynb)
