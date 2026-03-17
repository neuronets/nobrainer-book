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
# # Save, Load, and Reproduce Models
#
# Reproducibility is essential in neuroimaging research. Nobrainer saves
# models with **Croissant-ML** metadata -- a JSON-LD standard for
# describing ML datasets and models. This tutorial covers:
#
# 1. Saving a trained model with `seg.save()`
# 2. Inspecting the `croissant.json` metadata
# 3. Loading a model with `Segmentation.load()`
# 4. Exporting dataset metadata with `Dataset.to_croissant()`
# 5. FAIR principles for neuroimaging models

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
# ## 1. Train a small model

# %%
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
    Dataset.from_files(filepaths[:3], block_shape=(16, 16, 16), n_classes=2)
    .batch(2)
    .binarize()
)

seg = Segmentation(
    "unet",
    model_args={"in_channels": 1, "channels": (4, 8), "strides": (2,)},
)
seg.fit(ds, epochs=2)
print("Model trained!")

# %% [markdown]
# ## 2. Save the model
#
# `seg.save()` creates a directory containing:
# - `model.pth` -- PyTorch state dict (weights)
# - `croissant.json` -- Croissant-ML metadata (architecture, training config,
#   dataset provenance)

# %%
import tempfile
import os

save_dir = os.path.join(tempfile.mkdtemp(), "my_model")
seg.save(save_dir)

print("Saved to:", save_dir)
print("Contents:")
for f in sorted(os.listdir(save_dir)):
    size = os.path.getsize(os.path.join(save_dir, f))
    print(f"  {f} ({size:,} bytes)")

# %% [markdown]
# ## 3. Inspect the Croissant-ML metadata
#
# The `croissant.json` file contains structured metadata about the model,
# training configuration, and data provenance in JSON-LD format.

# %%
import json

with open(os.path.join(save_dir, "croissant.json")) as f:
    metadata = json.load(f)

print(json.dumps(metadata, indent=2))

# %% [markdown]
# ### Key metadata fields
#
# The provenance section (`nobrainer:provenance`) captures:
#
# - **model_architecture**: which model was used (e.g., "unet")
# - **model_args**: constructor arguments for exact reconstruction
# - **n_classes**: number of output classes
# - **block_shape**: patch size used during training
# - **optimizer**: optimizer class and learning rate
# - **loss_function**: loss function used
# - **nobrainer_version**: version for reproducibility

# %% [markdown]
# ## 4. Load the model
#
# `Segmentation.load()` reads `croissant.json` to reconstruct the model
# architecture, then loads the weights from `model.pth`.

# %%
loaded_seg = Segmentation.load(save_dir)

print("Loaded model type:", type(loaded_seg).__name__)
print("Base model:", loaded_seg.base_model)
print("Model args:", loaded_seg.model_args)
print("N classes:", loaded_seg.n_classes_)

# %% [markdown]
# ### Verify the loaded model produces the same predictions

# %%
import numpy as np

eval_path = filepaths[3][0]

pred_original = seg.predict(eval_path, block_shape=(16, 16, 16))
pred_loaded = loaded_seg.predict(eval_path, block_shape=(16, 16, 16))

orig_data = np.asarray(pred_original.dataobj)
load_data = np.asarray(pred_loaded.dataobj)

print("Predictions match:", np.array_equal(orig_data, load_data))

# %% [markdown]
# ## 5. Export dataset metadata
#
# `Dataset.to_croissant()` exports a Croissant-ML description of the
# training data, including file paths, volume shapes, and processing steps.

# %%
ds_croissant_path = os.path.join(tempfile.mkdtemp(), "dataset_croissant.json")
ds.to_croissant(ds_croissant_path)

with open(ds_croissant_path) as f:
    ds_metadata = json.load(f)

print("Dataset Croissant metadata:")
print(json.dumps(ds_metadata, indent=2))

# %% [markdown]
# ## 6. FAIR principles for neuroimaging models
#
# Nobrainer's Croissant-ML metadata supports the FAIR principles:
#
# ### Findable
# - Each saved model has structured metadata in a standard format
# - JSON-LD enables indexing by search engines and data catalogs
#
# ### Accessible
# - Models are self-contained directories (weights + metadata)
# - No external dependencies needed to understand the model
#
# ### Interoperable
# - Croissant-ML is a community standard supported by Google, Hugging Face,
#   and others
# - JSON-LD links to schema.org and ML Commons vocabularies
#
# ### Reusable
# - Full provenance: architecture, hyperparameters, training data, software
#   versions
# - `Segmentation.load()` reconstructs the exact model from metadata

# %% [markdown]
# ## Summary
#
# Nobrainer saves models with Croissant-ML metadata for full
# reproducibility. The `save()`/`load()` cycle preserves architecture,
# weights, and training provenance. Dataset metadata can also be exported
# for data-level documentation. In the next tutorial we will explore the
# Zarr v3 data pipeline.
