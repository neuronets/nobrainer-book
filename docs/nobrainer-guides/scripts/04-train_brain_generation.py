# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
#
# # Train a neural network to generate realistic brain volumes
#
# In this notebook, we will use `nobrainer` to train a model for generation of
# realistic, synthetic brain MRI volumes. We will use a Progressive GAN with
# PyTorch Lightning for training.
#
# In the following cells, we will:
#
# 1. Get sample T1-weighted MR scans as features.
# 2. Normalize and downsample the volumes to a small resolution for fast
#    training.
# 3. Build a PyTorch DataLoader.
# 4. Instantiate a ProgressiveGAN model.
# 5. Train with PyTorch Lightning.
# 6. Generate some synthetic brain images.
#
# ## Google Colaboratory
#
# If you are using Colab, please switch your runtime to GPU. To do this, select
# `Runtime > Change runtime type` in the top menu. Then select GPU under
# `Hardware accelerator`. A GPU greatly speeds up training.
#
# # Install and setup `nobrainer`

# %%
# !pip install --no-cache-dir nilearn nobrainer

# %%
import nibabel as nib
import numpy as np
import pytorch_lightning as pl
from scipy.ndimage import zoom
import torch
from torch.utils.data import DataLoader, TensorDataset

from nobrainer.io import read_csv
from nobrainer.models.generative import ProgressiveGAN
from nobrainer.utils import get_data

# %% [markdown]
# # Get sample features
#
# We use all 10 T1-weighted volumes for training. Many more volumes would be
# required to train a model for any useful purpose.

# %%
csv_path = get_data()
filepaths = read_csv(csv_path)
print(f"Downloaded {len(filepaths)} subjects")

# %% [markdown]
# # Downsample and normalize volumes
#
# The generative model expects inputs in a specific range. We normalize each
# volume to [0, 1] and downsample to 4^3 for fast CPU training. For higher
# quality generation, use larger resolutions and a multi-resolution schedule.

# %%
TARGET_SIZE = 4

volumes = []
for img_path, _ in filepaths:
    vol = np.asarray(nib.load(img_path).dataobj, dtype=np.float32)
    # Normalize to [0, 1]
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    # Downsample to TARGET_SIZE^3
    factors = [TARGET_SIZE / s for s in vol.shape[:3]]
    vol_small = zoom(vol, factors, order=1)
    volumes.append(vol_small)

imgs = torch.from_numpy(np.stack(volumes)[:, None])  # (N, 1, 4, 4, 4)
print(f"Training set: {imgs.shape[0]} volumes of shape {TARGET_SIZE}^3")
print(f"Value range: [{imgs.min():.3f}, {imgs.max():.3f}]")

# %% [markdown]
# # Create a DataLoader

# %%
torch.manual_seed(42)

BATCH_SIZE = 4
loader = DataLoader(TensorDataset(imgs), batch_size=BATCH_SIZE, shuffle=True)

# %% [markdown]
# # Build the ProgressiveGAN
#
# We use a single resolution level (4) to keep this demo fast on CPU. For
# multi-resolution training, pass e.g.
# `resolution_schedule=[4, 8, 16, 32, 64]`.

# %%
model = ProgressiveGAN(
    latent_size=64,
    fmap_base=64,
    fmap_max=64,
    resolution_schedule=[TARGET_SIZE],
    steps_per_phase=500,
)

n_params_g = sum(p.numel() for p in model.generator.parameters())
n_params_d = sum(p.numel() for p in model.discriminator.parameters())
print(f"Generator params:     {n_params_g:,}")
print(f"Discriminator params: {n_params_d:,}")

# %% [markdown]
# # Train with PyTorch Lightning
#
# The fit progresses through the resolutions defined in the schedule. For this
# demo we run only a small number of steps.

# %%
trainer = pl.Trainer(
    max_steps=100,
    accelerator="auto",
    devices=1,
    enable_checkpointing=False,
    logger=False,
    enable_progress_bar=True,
)

trainer.fit(model, loader)
print("Training complete")

# %% [markdown]
# # Generate synthetic brain images from the trained PGAN model
#
# The generated volumes are at 4^3 resolution. For useful images, train at
# higher resolutions with more data and training steps.

# %%
from nilearn import plotting
import matplotlib.pyplot as plt

model.eval()
model.generator.current_level = 0
model.generator.alpha = 1.0

with torch.no_grad():
    z = torch.randn(4, 64, device=model.device)
    generated = model.generator(z)

gen_np = generated.cpu().numpy()
print(f"Generated shape: {gen_np.shape}")
print(f"Value range: [{gen_np.min():.3f}, {gen_np.max():.3f}]")

# %% [markdown]
# # Save model

# %%
trainer.save_checkpoint("data/brain_generator.ckpt")
print("Model saved to data/brain_generator.ckpt")

# %% [markdown]
# # Load model
#
# ```python
# model = ProgressiveGAN.load_from_checkpoint("data/brain_generator.ckpt")
# ```

# %% [markdown]
# Next, learn how to
# [use augmentation to train models with less data](https://colab.research.google.com/github/neuronets/nobrainer-book/blob/master/docs/nobrainer-guides/notebooks/05-training_with_augmentation.ipynb)
