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
# # Download and Inspect Brain Data
#
# Nobrainer provides a small sample dataset of T1-weighted brain MRI volumes
# with corresponding FreeSurfer parcellation labels. This tutorial shows how
# to download the data, read the CSV manifest, load volumes with nibabel,
# and understand the label encoding.

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
# ## 1. Download sample data
#
# `get_data()` downloads 10 T1-weighted volumes and their FreeSurfer
# aparc+aseg labels (~46 MB total) and returns a path to a CSV file.

# %%
from nobrainer.utils import get_data

csv_path = get_data()
print("CSV file:", csv_path)

# %% [markdown]
# ## 2. Read the CSV manifest

# %%
import csv

with open(csv_path) as f:
    reader = csv.reader(f)
    header = next(reader)
    filepaths = [(row[0], row[1]) for row in reader]

print("Header:", header)
print(f"Number of subjects: {len(filepaths)}")
print("First entry:")
print("  Feature:", filepaths[0][0])
print("  Label:  ", filepaths[0][1])

# %% [markdown]
# ## 3. Load a volume with nibabel
#
# The sample data are in FreeSurfer MGZ format, which nibabel reads natively.

# %%
import nibabel as nib
import numpy as np

feature_path, label_path = filepaths[0]

# Load the T1-weighted image
img = nib.load(feature_path)
print("Image shape:", img.shape)
print("Voxel size:", img.header.get_zooms())
print("Data type:", img.get_data_dtype())
print()
print("Affine matrix (voxel-to-world):")
print(img.affine)

# %% [markdown]
# ## 4. Inspect the image data

# %%
data = np.asarray(img.dataobj)
print("Array shape:", data.shape)
print("Value range: [{:.1f}, {:.1f}]".format(data.min(), data.max()))
print("Mean: {:.1f}, Std: {:.1f}".format(data.mean(), data.std()))
print("Non-zero voxels: {:,} / {:,} ({:.1f}%)".format(
    (data > 0).sum(),
    data.size,
    100 * (data > 0).sum() / data.size,
))

# %% [markdown]
# ## 5. Inspect the label volume

# %%
label_img = nib.load(label_path)
labels = np.asarray(label_img.dataobj)

print("Label shape:", labels.shape)
print("Data type:", labels.dtype)

unique_labels = np.unique(labels)
print(f"\nNumber of unique label values: {len(unique_labels)}")
print("Unique labels:", unique_labels)

# %% [markdown]
# ## 6. Understanding FreeSurfer label codes
#
# The labels come from FreeSurfer's `aparc+aseg` parcellation. Here are some
# commonly used region codes:
#
# | Code | Region |
# |------|--------|
# | 0 | Background |
# | 2 | Left Cerebral White Matter |
# | 3 | Left Cerebral Cortex |
# | 4 | Left Lateral Ventricle |
# | 7 | Left Cerebellar White Matter |
# | 8 | Left Cerebellar Cortex |
# | 10 | Left Thalamus |
# | 11 | Left Caudate |
# | 12 | Left Putamen |
# | 13 | Left Pallidum |
# | 17 | Left Hippocampus |
# | 18 | Left Amygdala |
# | 26 | Left Accumbens |
# | 41 | Right Cerebral White Matter |
# | 42 | Right Cerebral Cortex |
# | 43 | Right Lateral Ventricle |
# | 53 | Right Hippocampus |
# | 54 | Right Amygdala |
#
# Codes 1000-2999 are cortical parcellation labels from the Desikan-Killiany
# atlas (1000s = left hemisphere, 2000s = right hemisphere).

# %%
# Count voxels for a few key regions
regions = {
    0: "Background",
    2: "L Cerebral WM",
    3: "L Cerebral Cortex",
    17: "L Hippocampus",
    53: "R Hippocampus",
}

print("Voxel counts for selected regions:")
for code, name in regions.items():
    count = (labels == code).sum()
    print(f"  {code:4d} ({name}): {count:>8,} voxels")

# %% [markdown]
# ## 7. Quick visualization (optional)
#
# If nilearn is installed, we can display a slice.

# %%
try:
    from nilearn import plotting
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    mid_slice = data.shape[2] // 2
    axes[0].imshow(data[:, :, mid_slice].T, cmap="gray", origin="lower")
    axes[0].set_title("T1-weighted image")
    axes[0].axis("off")

    axes[1].imshow(labels[:, :, mid_slice].T, cmap="nipy_spectral", origin="lower")
    axes[1].set_title("FreeSurfer labels")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    # Binarized brain mask overlay
    brain_mask = (labels > 0).astype(np.float32)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(data[:, :, mid_slice].T, cmap="gray", origin="lower")
    plt.title("T1-weighted volume")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(labels[:, :, mid_slice].T, cmap="nipy_spectral", origin="lower")
    plt.title("Label volume")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(data[:, :, mid_slice].T, cmap="gray", origin="lower")
    plt.imshow(brain_mask[:, :, mid_slice].T, cmap="Reds", alpha=0.3, origin="lower")
    plt.title("Binarized label overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
except ImportError:
    print("Install nilearn and matplotlib for visualization: "
          "pip install nilearn matplotlib")

# %% [markdown]
# ## Summary
#
# We downloaded 10 brain volumes with FreeSurfer labels, inspected their
# shape and affine, and explored the label encoding. In the next tutorial
# we will prepare this data for training by extracting patches and building
# a Dataset.
