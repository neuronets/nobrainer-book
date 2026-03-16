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
# # Use checkpoints to resume training a model for binary volumetric brain
# extraction
#
# In the following cells, we will:
#
# 1. Get sample T1-weighted MR scans as features and FreeSurfer segmentations
#    as labels.
# 2. Load volumes and extract random 3-D patches.
# 3. Build a PyTorch DataLoader.
# 4. Instantiate a brain segmentation model with checkpointing to store
#    training results progressively.
# 5. Train a bit.
# 6. Load the partially trained model from a checkpoint and resume training.
# 7. Save the final model to disk.
# 8. Load the model back from disk and show that brain extraction works as
#    before saving.


# %% [markdown]
# ## Google Colaboratory
#
# If you are using Colab, please switch your runtime to GPU. To do this, select
# `Runtime > Change runtime type` in the top menu. Then select GPU under
# `Hardware accelerator`. A GPU greatly speeds up training.


# %% [markdown]
# # Install and setup `nobrainer`

# %%
import subprocess
import sys

try:
    import google.colab  # noqa: F401
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--pre", "-q",
         "nobrainer", "nilearn", "matplotlib"]
    )
except ImportError:
    pass  # Not on Colab; install manually with: uv pip install nobrainer

# %%
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from nobrainer.io import read_csv
from nobrainer.models.segmentation import unet
from nobrainer.prediction import predict
from nobrainer.training import fit
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
BATCH_SIZE = 4
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# %% [markdown]
# # U-Net model for brain mask extraction

# %% [markdown]
# ## Construct the model
# Set up the model to train in sessions, resuming from checkpoints each time.
# If no checkpoints exist in the specified location, training starts fresh.
#
# Here we train `nobrainer`'s implementation of the U-Net model for biomedical
# image segmentation, based on https://arxiv.org/abs/1606.06650.
#
# Note that a useful segmentation model would need to be trained on *many* more
# examples than the 10 we are using here for demonstration.

# %%
model_dir = "brain_mask_extraction_model"
checkpoint_dir = os.path.join(model_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

model = unet(n_classes=2, channels=(8, 16, 32), strides=(2, 2)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# %% [markdown]
# ## Train the model on the example data (first session)
#
# We use `nobrainer.training.fit()` which supports automatic checkpointing.
# The best model weights are saved to the checkpoint directory.

# %%
N_EPOCHS = 2

result = fit(
    model=model,
    loader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    max_epochs=N_EPOCHS,
    checkpoint_dir=checkpoint_dir,
)

print(f"Training complete: {result['epochs_completed']} epochs")
print(f"Final loss: {result['final_loss']:.4f}")
print(f"Best loss:  {result['best_loss']:.4f}")
print(f"Checkpoint: {result['checkpoint_path']}")


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
# ## Resume training from a checkpoint
# This paradigm is useful in situations where training takes a long time and
# compute resources may be preemptable or available in chunks.

# %%
# Create a fresh model and load the checkpoint weights
model_resumed = unet(n_classes=2, channels=(8, 16, 32), strides=(2, 2)).to(device)

ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
if os.path.exists(ckpt_path):
    model_resumed.load_state_dict(torch.load(ckpt_path, weights_only=True))
    print(f"Resumed from checkpoint: {ckpt_path}")

optimizer_resumed = torch.optim.Adam(model_resumed.parameters(), lr=1e-3)

result = fit(
    model=model_resumed,
    loader=train_loader,
    criterion=criterion,
    optimizer=optimizer_resumed,
    max_epochs=N_EPOCHS,
    checkpoint_dir=checkpoint_dir,
)

print(f"Resumed training complete: {result['epochs_completed']} epochs")
print(f"Final loss: {result['final_loss']:.4f}")


# %% [markdown]
# ## Save the trained model

# %%
final_path = os.path.join(model_dir, "final_model.pth")
torch.save(model_resumed.state_dict(), final_path)
print(f"Model saved to {final_path}")


# %% [markdown]
# ## Load the model from disk for prediction.

# %%
model_loaded = unet(n_classes=2, channels=(8, 16, 32), strides=(2, 2)).to(device)
model_loaded.load_state_dict(torch.load(final_path, weights_only=True))
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
