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
# # Generate Synthetic Brain Volumes
#
# Generative adversarial networks (GANs) can synthesize realistic brain MRI
# volumes. This is useful for data augmentation, privacy-preserving data
# sharing, and studying model biases. This tutorial trains a tiny
# Progressive GAN on downsampled brain data.

# %%
PRE_RELEASE = False
import subprocess, sys
try:
    import google.colab  # noqa: F401
    cmd = [sys.executable, "-m", "pip", "install", "-q",
           "nobrainer", "nilearn", "matplotlib", "pytorch-lightning"]
    if PRE_RELEASE:
        cmd.insert(4, "--pre")
    subprocess.check_call(cmd)
except ImportError:
    pass

# %% [markdown]
# ## 1. Prepare downsampled training data
#
# Progressive GANs start training at low resolution and grow. For this
# tutorial we downsample brain volumes to 4x4x4 so training completes
# in seconds.

# %%
import csv
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import torch
from torch.utils.data import DataLoader, TensorDataset
from nobrainer.utils import get_data

csv_path = get_data()
with open(csv_path) as f:
    reader = csv.reader(f)
    next(reader)
    filepaths = [(row[0], row[1]) for row in reader]

# Downsample volumes to 4^3
target_shape = (4, 4, 4)
volumes = []
for feat_path, _ in filepaths[:5]:
    vol = nib.load(feat_path).get_fdata().astype(np.float32)
    # Compute zoom factors
    factors = tuple(t / s for t, s in zip(target_shape, vol.shape[:3]))
    small = zoom(vol, factors, order=1)
    # Normalize to [0, 1]
    small = (small - small.min()) / (small.max() - small.min() + 1e-8)
    volumes.append(small)

# Stack into a tensor: (N, 1, 4, 4, 4)
data_tensor = torch.tensor(np.stack(volumes)[:, None], dtype=torch.float32)
print("Training tensor shape:", data_tensor.shape)

# Build a DataLoader
train_dataset = TensorDataset(data_tensor)
loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# %% [markdown]
# ## 2. Train the Progressive GAN
#
# We configure a very small model:
# - `latent_size=32`: small latent vector
# - `fmap_base=32`, `fmap_max=32`: minimal feature maps
# - `resolution_schedule=[4]`: single resolution level (4^3)
# - `steps_per_phase=100`: few steps per training phase

# %%
from nobrainer.processing.generation import Generation

gen = Generation(
    "progressivegan",
    model_args={
        "latent_size": 32,
        "fmap_base": 32,
        "fmap_max": 32,
        "resolution_schedule": [4],
        "steps_per_phase": 100,
    },
)

gen.fit(loader, epochs=50)
print("GAN training complete!")

# %% [markdown]
# ## 3. Generate synthetic volumes

# %%
synthetic_images = gen.generate(2)

print(f"Generated {len(synthetic_images)} synthetic volumes")
for i, img in enumerate(synthetic_images):
    arr = np.asarray(img.dataobj)
    print(f"  Volume {i}: shape={arr.shape}, "
          f"range=[{arr.min():.3f}, {arr.max():.3f}], "
          f"mean={arr.mean():.3f}")

# %% [markdown]
# ## 4. Visualize generated vs. real
#
# At this tiny resolution the images will be blurry blobs, but the workflow
# is identical for full-resolution training.

# %%
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    # Real volume (middle slice)
    real = volumes[0]
    axes[0].imshow(real[:, :, 2], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Real (4x4x4)")
    axes[0].axis("off")

    # Generated volumes
    for i, img in enumerate(synthetic_images[:2]):
        arr = np.asarray(img.dataobj)
        axes[i + 1].imshow(arr[:, :, 2], cmap="gray")
        axes[i + 1].set_title(f"Generated {i + 1}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()
except ImportError:
    print("Install matplotlib for visualization")

# %% [markdown]
# ## Notes for production use
#
# For realistic brain generation:
#
# - Use full-resolution data (e.g., 64^3 or 128^3)
# - Set `resolution_schedule=[4, 8, 16, 32, 64]` for progressive growing
# - Increase `fmap_base` and `fmap_max` (e.g., 512)
# - Train for thousands of steps per phase
# - Use GPU acceleration

# %% [markdown]
# ## Summary
#
# We trained a Progressive GAN to generate synthetic 3D brain volumes.
# The `Generation` estimator follows the same sklearn-style pattern:
# `.fit()` to train, `.generate()` to produce new samples. In the next
# tutorial we will look under the hood at custom training loops.
