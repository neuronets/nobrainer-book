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
# # Train Brain Segmentation in 3 Lines
#
# Nobrainer's sklearn-style API makes it possible to train a brain
# segmentation model with minimal code. This tutorial demonstrates the
# full train-predict-evaluate workflow using a tiny UNet.

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
# ## 1. Prepare the dataset

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

# Use a small subset for this tutorial
train_files = filepaths[:3]
eval_file = filepaths[3]

# Build the training dataset: binary brain mask, small patches, batch of 2
ds = (
    Dataset.from_files(train_files, block_shape=BLOCK_SHAPE, n_classes=2)
    .batch(2)
    .binarize()
)

print("Training dataset ready")
print("  Files:", len(ds.data))
print("  Block shape:", ds.block_shape)
print("  Batch size:", ds.batch_size)

# %% [markdown]
# ## 2. CPU vs CUDA
#
# Nobrainer detects available hardware automatically:
# - **CPU**: works everywhere (Colab, laptops, CI)
# - **CUDA GPU**: used automatically when available — no code changes needed
#
# The estimator API handles device placement transparently:
# - `.fit()` moves the model and data to GPU if CUDA is available
# - `.predict()` runs inference on the best available device
#
# To **force CPU** (e.g., for debugging), set `multi_gpu=False` and the
# model stays on CPU. For **explicit device control**, use the advanced
# API (see tutorial 07):
# ```python
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# ```

# %%
import torch  # noqa: E402

print(f"PyTorch device: {'cuda (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'cpu'}")

# %% [markdown]
# ## 3. Train with the Segmentation estimator
#
# The `Segmentation` class wraps model creation, optimizer setup, and
# training into a single `.fit()` call. We use tiny model parameters
# for speed: `channels=(4, 8)` and `strides=(2,)`.

# %%
from nobrainer.processing.segmentation import Segmentation  # noqa: E402

seg = Segmentation(
    "unet",
    model_args={"in_channels": 1, "channels": (4, 8), "strides": (2,)},
)

seg.fit(ds, epochs=2)
print("Training complete!")

# %% [markdown]
# ## 3. Predict on an evaluation volume
#
# `.predict()` accepts a file path, a nibabel image, or a numpy array.
# It handles patch extraction, inference, and reassembly automatically.

# %%
eval_feature_path, eval_label_path = eval_file

prediction = seg.predict(eval_feature_path, block_shape=BLOCK_SHAPE)
print("Prediction type:", type(prediction))
print("Prediction shape:", prediction.shape)

# %% [markdown]
# ## 4. Compute the Dice coefficient
#
# The Dice coefficient measures overlap between the predicted and true
# segmentation. A score of 1.0 is perfect agreement.

# %%
import nibabel as nib
import numpy as np

pred_data = np.asarray(prediction.dataobj)
true_data = np.asarray(nib.load(eval_label_path).dataobj)

# Binarize the ground truth to match our training labels
true_binary = (true_data > 0).astype(np.float32)
pred_binary = (pred_data > 0).astype(np.float32)

intersection = (pred_binary * true_binary).sum()
dice = 2.0 * intersection / (pred_binary.sum() + true_binary.sum() + 1e-8)

print(f"Dice coefficient: {dice:.4f}")
print("(Low score expected with tiny model and 2 epochs)")

# %% [markdown]
# ## 5. Visualize predictions

# %%
import matplotlib.pyplot as plt

feature_vol = np.asarray(nib.load(eval_feature_path).dataobj)
mid_slice = feature_vol.shape[2] // 2

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(feature_vol[:, :, mid_slice].T, cmap="gray", origin="lower")
plt.title("Input volume")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(pred_data[:, :, mid_slice].T, cmap="gray", origin="lower")
plt.title("Prediction")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(true_binary[:, :, mid_slice].T, cmap="gray", origin="lower")
plt.title("Ground truth")
plt.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Putting it all together
#
# Here is the entire workflow in three key lines:
#
# ```python
# ds = Dataset.from_files(filepaths, block_shape=(128,128,128), n_classes=2).batch(2).binarize()
# seg = Segmentation("unet").fit(ds, epochs=5)
# result = seg.predict("brain.nii.gz")
# ```
#
# For production training, increase the block shape (e.g., 128^3),
# use more channels, train for more epochs, and add augmentation
# with `.augment()`.

# %% [markdown]
# ## Summary
#
# We trained a UNet brain segmentation model, generated predictions, and
# computed the Dice score -- all with nobrainer's high-level API. In the
# next tutorial we will explore Bayesian models for uncertainty
# quantification.
