# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="jlHhhplL7opm"
# # Train a neural network to generate realistic brain volumes
#
# In this notebook, we will use `nobrainer` to train a model for generation of realistic, synthetic brain MRI volumes. We will use a Generative Adversarial Network to model the generation and use a progressive growing training method for high quality generation at higher resolutions.
#
# In the following cells, we will:
#
# 1. Get sample T1-weighted MR scans as features.
# 2. Convert the data to TFRecords format.
# 3. Instantiate a progressive convolutional neural network for generator and discriminator.
# 4. Create a Dataset of the features.
# 5. Instantiate a trainer and choose a loss function to use.
# 6. Train on part of the data in two phases (transition and resolution).
# 7. Repeat steps 4-6 for each growing resolution.
# 8. Generate some images using trained model

# %% [markdown]
# ## Google Colaboratory
#
# If you are using Colab, please switch your runtime to GPU. To do this, select `Runtime > Change runtime type` in the top menu. Then select GPU under `Hardware accelerator`. A GPU greatly speeds up training.


# %% [markdown]
# # Install and setup `nobrainer`

# %% id="sUwl5vYH7rrD"
# !pip install --no-cache-dir nilearn nobrainer

# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# %% id="Qpm8u9O47opq"
import nobrainer


# %% [markdown] id="QAl3sk8e7opr"
# # Get sample features and labels
#
# We use 9 pairs of volumes for training and 1 pair of volumes for evaluation. Many more volumes would be required to train a model for any useful purpose.

# %% id="yV9F64HE7opr"
csv_of_filepaths = nobrainer.utils.get_data()
filepaths = nobrainer.io.read_csv(csv_of_filepaths)

train_paths = filepaths[:9]

# %% [markdown] id="jqzlAmyI7ops"
# # Convert medical images to TFRecords
#
# Remember how many full volumes are in the TFRecords files. This will be necessary to know how many steps are in on training epoch. The default training method needs to know this number, because Datasets don't always know how many items they contain.

# %%
from nobrainer.dataset import write_multi_resolution

# %%
datasets = write_multi_resolution(train_paths,
                                  tfrecdir="data/generate",
                                  n_processes=None)

# %%
print(datasets)

# %% [markdown]
# The datasets will look like the following. One can adjust the `batch size` depending on compute power and available GPUs, but also epochs and normalizers.

# %%
datasets = {8: {'file_pattern': 'data/generate/*res-008.tfrec',
                'batch_size': 1,
                'normalizer': None},
            16: {'file_pattern': 'data/generate/*res-016.tfrec',
                 'batch_size': 1,
                 'normalizer': None},
            32: {'file_pattern': 'data/generate/*res-032.tfrec',
                 'batch_size': 1,
                 'normalizer': None},
            64: {'file_pattern': 'data/generate/*res-064.tfrec',
                 'batch_size': 1,
                 'normalizer': None},
            128: {'file_pattern': 'data/generate/*res-128.tfrec',
                  'batch_size': 1,
                  'normalizer': None},
            256: {'file_pattern': 'data/generate/*res-256.tfrec',
                  'batch_size': 1,
                  'normalizer': None}}

# %%
from nobrainer.volume import normalize, adjust_dynamic_range

def scale(x):
    """Scale data to -1 to 1"""
    return adjust_dynamic_range(normalize(x), [0, 1], [-1, 1])

# Adjust number of epochs
datasets[8]["epochs"] = 1000
datasets[16]["epochs"] = 1000
datasets[32]["epochs"] = 400
datasets[64]["epochs"] = 200

# Adjust batch size from the default of 1
datasets[8]["batch_size"] = 8
datasets[16]["batch_size"] = 8
datasets[32]["batch_size"] = 8
datasets[64]["batch_size"] = 4

# %%
from nobrainer.processing.generation import ProgressiveGeneration
gen = ProgressiveGeneration()

# %%
# epochs and normalizer can be overwritten by resolution specific settings in datasets
gen.fit(datasets,
        epochs=10,
        normalizer=scale)

# %% [markdown]
# #### Scale output to uint16
#
# One can return the native datatype by not passing a `data_type` argument.

# %%
from nilearn import plotting
import numpy as np
import matplotlib.pyplot as plt

images = gen.generate(data_type=np.uint8, n_images=10)

fig, ax = plt.subplots(len(images), 1, figsize=(18, 30))
index = 0
for img in images:
    plotting.plot_anat(anat_img=img, figure=fig, axes=ax[index],
                       draw_cross=False)
    index += 1

# %% [markdown]
# #### Warm restart at the last resolution
#
# We can warm start the training, but for the moment it will only retrain using the final resolution of the data or higher. This can be used to:
# - split different resolution phases
# - use different datasets for different resolutions
# - fine tune on a specific dataset.

# %%
gen.fit(datasets, warm_start=True)

# %% [markdown]
# ### Save model

# %%
gen.save("data/saved_gen_model")

# %% [markdown]
# ### TODO: Load and reuse model