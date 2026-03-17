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
# # Advanced: Custom Training Loops
#
# The `Segmentation` estimator is convenient, but sometimes you need full
# control over the training loop -- for custom losses, learning rate
# schedules, gradient accumulation, or mixed precision. This tutorial shows
# how to use nobrainer's lower-level components directly.

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
# ## 1. Prepare patches and a raw DataLoader

# %%
import csv
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from nobrainer.utils import get_data
from nobrainer.processing.dataset import extract_patches

csv_path = get_data()
with open(csv_path) as f:
    reader = csv.reader(f)
    next(reader)
    filepaths = [(row[0], row[1]) for row in reader]

# Extract patches from the first 3 subjects
block_shape = (16, 16, 16)
all_img_patches = []
all_lbl_patches = []

for feat_path, label_path in filepaths[:3]:
    vol = nib.load(feat_path).get_fdata()
    lbl = nib.load(label_path).get_fdata()
    patches = extract_patches(
        vol, lbl,
        block_shape=block_shape,
        n_patches=10,
        binarize=True,
    )
    for img_p, lbl_p in patches:
        all_img_patches.append(img_p)
        all_lbl_patches.append(lbl_p)

# Convert to tensors: (N, 1, D, H, W)
X = torch.tensor(np.stack(all_img_patches)[:, None], dtype=torch.float32)
y = torch.tensor(np.stack(all_lbl_patches), dtype=torch.long)

print(f"Training patches: {X.shape[0]}")
print(f"Image tensor shape: {X.shape}")
print(f"Label tensor shape: {y.shape}")

# Build a standard PyTorch DataLoader
train_ds = TensorDataset(X, y)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

# %% [markdown]
# ## 2. Create the model directly
#
# Instead of using `Segmentation("unet")`, we call the model factory
# directly for full control over the architecture.

# %%
from nobrainer.models import get as get_model

# This is what Segmentation does internally
unet_factory = get_model("unet")
model = unet_factory(
    in_channels=1,
    n_classes=2,
    channels=(4, 8),
    strides=(2,),
)

print("Model created:", type(model).__name__)
print("Parameters:", sum(p.numel() for p in model.parameters()))

# %% [markdown]
# ## 3. Option A: Manual training loop
#
# Full control over every step.

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

epochs = 2
model.train()

for epoch in range(epochs):
    epoch_loss = 0.0
    n_batches = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    print(f"Epoch {epoch + 1}/{epochs} -- loss: {avg_loss:.4f}")

print("Manual training complete!")

# %% [markdown]
# ## 4. Option B: Use `nobrainer.training.fit()`
#
# If you want multi-GPU support and checkpointing but still need a custom
# model or loss, use the `fit()` function directly. This is what the
# `Segmentation` estimator calls internally.

# %%
from nobrainer.training import fit as training_fit

# Re-initialize model for a fresh start
model_b = unet_factory(
    in_channels=1,
    n_classes=2,
    channels=(4, 8),
    strides=(2,),
)
optimizer_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
criterion_b = torch.nn.CrossEntropyLoss()

# This mirrors what Segmentation.fit() does:
#   seg = Segmentation("unet")
#   seg.fit(ds, epochs=2)
# But here you can pass any custom model, optimizer, or loss.
result = training_fit(
    model=model_b,
    loader=train_loader,
    criterion=criterion_b,
    optimizer=optimizer_b,
    max_epochs=2,
    gpus=0,  # CPU for tutorial
)

print("training.fit() result:", result)

# %% [markdown]
# ## 5. Prediction with a manually-trained model
#
# You can still use the high-level `predict()` function.

# %%
from nobrainer.prediction import predict

eval_path = filepaths[3][0]
prediction = predict(
    inputs=eval_path,
    model=model,
    block_shape=block_shape,
    batch_size=4,
)

print("Prediction shape:", prediction.shape)

# %% [markdown]
# ## Mapping to the estimator API
#
# Here is how the manual steps correspond to the estimator API:
#
# | Manual step | Estimator equivalent |
# |-------------|---------------------|
# | `get_model("unet")(**args)` | `Segmentation("unet", model_args={...})` |
# | `Adam(model.parameters())` | Handled by `.fit(optimizer=Adam)` |
# | `training_fit(model, ...)` | `seg.fit(ds, epochs=N)` |
# | `predict(inputs, model, ...)` | `seg.predict(volume)` |
# | `torch.save(state_dict)` | `seg.save("dir")` |
#
# Use the estimator API for convenience, and drop to the manual level
# when you need fine-grained control.

# %% [markdown]
# ## Summary
#
# This tutorial showed two ways to train manually: a raw PyTorch loop and
# `nobrainer.training.fit()`. Both give you full control while still
# leveraging nobrainer's model zoo and prediction pipeline. In the next
# tutorial we will cover saving, loading, and reproducibility.
