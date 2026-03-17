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
# # Multi-GPU Training
#
# Brain MRI segmentation involves large 3D volumes and can be slow on a
# single GPU. Nobrainer supports multi-GPU training via PyTorch's
# Distributed Data Parallel (DDP). This tutorial explains the concepts
# and shows the API -- actual multi-GPU execution requires a machine with
# multiple GPUs.

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
# ## 1. Detect available GPUs

# %%
import torch

n_gpus = torch.cuda.device_count()
print(f"Available GPUs: {n_gpus}")

if n_gpus > 0:
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1e9
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")
else:
    print("  No GPUs detected. Multi-GPU examples are conceptual only.")

# %% [markdown]
# ## 2. Multi-GPU training with the Segmentation estimator
#
# The `Segmentation` class automatically uses all available GPUs when
# `multi_gpu=True` (the default). Internally, it calls
# `nobrainer.training.fit(gpus=N)` where N is `torch.cuda.device_count()`.
#
# ```python
# from nobrainer.processing.segmentation import Segmentation
# from nobrainer.processing.dataset import Dataset
#
# # Build dataset (same as single-GPU)
# ds = (
#     Dataset.from_files(filepaths, block_shape=(128, 128, 128), n_classes=2)
#     .batch(4)        # batch size per GPU
#     .binarize()
#     .augment()
# )
#
# # Train on all GPUs (automatic)
# seg = Segmentation("unet", multi_gpu=True)
# seg.fit(ds, epochs=50)
# ```
#
# With `multi_gpu=True`:
# - The model is replicated across all GPUs
# - Each GPU processes a different batch (data parallelism)
# - Gradients are synchronized across GPUs after each step
# - Effective batch size = `batch_size * n_gpus`

# %% [markdown]
# ## 3. The `nobrainer.training.fit()` function
#
# Under the hood, the estimator calls `nobrainer.training.fit()` with
# a `gpus` parameter:
#
# ```python
# from nobrainer.training import fit
#
# result = fit(
#     model=model,
#     loader=train_loader,
#     criterion=loss_fn,
#     optimizer=optimizer,
#     max_epochs=50,
#     gpus=4,                    # Number of GPUs to use
#     checkpoint_dir="ckpts/",   # Optional: save checkpoints
# )
# ```
#
# When `gpus > 1`, the function:
# 1. Spawns one process per GPU using `torch.multiprocessing`
# 2. Wraps the model in `DistributedDataParallel`
# 3. Uses a `DistributedSampler` on the DataLoader
# 4. Synchronizes gradients via NCCL backend
# 5. Only saves checkpoints from rank 0

# %% [markdown]
# ## 4. Batch size considerations
#
# With DDP, each GPU receives `batch_size` samples. The effective global
# batch size is:
#
# ```
# effective_batch = batch_size_per_gpu * n_gpus
# ```
#
# When scaling to more GPUs, you may want to:
# - Keep the per-GPU batch size the same (linear scaling)
# - Adjust the learning rate proportionally (`lr * n_gpus`)
# - Or keep the global batch size the same by reducing per-GPU batch size

# %%
# Example calculation
batch_per_gpu = 4
for n in [1, 2, 4, 8]:
    effective = batch_per_gpu * n
    suggested_lr = 1e-3 * n  # linear scaling rule
    print(f"  {n} GPUs: effective_batch={effective}, "
          f"suggested_lr={suggested_lr:.4f}")

# %% [markdown]
# ## 5. Multi-GPU prediction
#
# The `predict()` method also supports multi-GPU distribution. When a
# volume is split into patches for inference, the patches can be distributed
# across GPUs for faster prediction.
#
# ```python
# # Prediction distributes patches across available GPUs
# result = seg.predict("brain.nii.gz", batch_size=8)
# ```
#
# With 4 GPUs and `batch_size=8`, each GPU processes 2 patches at a time,
# giving approximately 4x speedup on the inference step.

# %% [markdown]
# ## 6. Tips for multi-GPU training
#
# ### Memory management
# - Monitor GPU memory with `nvidia-smi` or `torch.cuda.memory_summary()`
# - Reduce `batch_size` if you hit OOM errors
# - Use `block_shape=(64,64,64)` instead of `(128,128,128)` for tight memory
#
# ### Performance
# - Use `num_workers > 0` in DataLoaders for CPU-GPU overlap
# - Pin memory with `pin_memory=True` (done automatically by Dataset)
# - Use mixed precision (AMP) for ~2x speedup on modern GPUs
#
# ### Debugging
# - Start with `multi_gpu=False` and 1 GPU to verify correctness
# - Use `CUDA_VISIBLE_DEVICES=0,1` to control which GPUs are used
# - Check that all GPUs show similar utilization in `nvidia-smi`

# %%
# Quick single-GPU demo (works on any machine)
print("Running a minimal single-GPU/CPU training for verification...")

import csv
from nobrainer.utils import get_data
from nobrainer.processing.dataset import Dataset
from nobrainer.processing.segmentation import Segmentation

csv_path = get_data()
with open(csv_path) as f:
    reader = csv.reader(f)
    next(reader)
    filepaths = [(row[0], row[1]) for row in reader]

ds = (
    Dataset.from_files(filepaths[:2], block_shape=(16, 16, 16), n_classes=2)
    .batch(2)
    .binarize()
)

seg = Segmentation(
    "unet",
    model_args={"in_channels": 1, "channels": (4, 8), "strides": (2,)},
    multi_gpu=False,  # Force single device for this demo
)
seg.fit(ds, epochs=2)
print("Single-device training complete!")

# %% [markdown]
# ## Summary
#
# Nobrainer supports multi-GPU training transparently via the `multi_gpu`
# parameter. The same code works on 1 or N GPUs -- DDP handles distribution,
# gradient synchronization, and checkpointing. In the next tutorial we will
# look at how to contribute to the nobrainer project.
