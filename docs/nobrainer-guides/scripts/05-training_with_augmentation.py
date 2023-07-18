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
# In this notebook, we will demonstrate how to train a brain mask extraction model using `nobrainer` with training data augmentation. Augmentation is useful for improving the robustness of neural network models when a limited amount of training examples are available. The basic idea is that, by applying certain transformations to the training data prior to fitting the model, one can expand the input space to cover situations unaccounted for in the base training set.
#
# Nobrainer provides several methods of augmenting volumetric data including spatial and intensity transforms.
#
#
# In the following cells, we will:
#
# 1. Get sample T1-weighted MR scans as features and FreeSurfer segmentations as labels.
# 2. Convert the data to TFRecords format.
# 3. Create two Datasets of the features and labels.
# 4. Define a sequence of augmentations to be applied to the Dataset prior to training.
# 5. Instantiate a 3D convolutional neural network.
# 6. Train on part of the data.
# 7. Evaluate on the rest of the data.

# %% [markdown]
# ## Google Colaboratory
#
# If you are using Colab, please switch your runtime to GPU. To do this, select `Runtime > Change runtime type` in the top menu. Then select GPU under `Hardware accelerator`. A GPU greatly speeds up training.

# %% [markdown]
# # Install and setup `nobrainer`

# %% id="WhBnt2WdDlx9"
# !pip install nobrainer nilearn

# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import nobrainer

# %% [markdown]
# # Get sample features and labels

# %%
csv_of_filepaths = nobrainer.utils.get_data()
filepaths = nobrainer.io.read_csv(csv_of_filepaths)

# %% id="YpqTxNu4Dkt4"
csv_path = nobrainer.utils.get_data()
filepaths = nobrainer.io.read_csv(csv_path)


# %% [markdown] id="ScgF78rmDkt4"
# # Convert medical images to TFRecords

# %% id="n7lCL-55Ta4R"
from nobrainer.dataset import Dataset


# %% id="Q3zPyRlbTa4R"
n_epochs = 2
DT = Dataset(
    n_classes=1,
    batch_size=2,
    block_shape=(128, 128, 128),
    n_epochs=n_epochs,
)

# %% [markdown]
# # Augmentation
# Take a look at different augmentation options in Nobrainer spatial and intensity transforms. To set training with multiple augmentations, the parameter `augment` will be set as a list where their order will determine the sequence of execution. For example augment option below will first add Gaussian noise and will then perform the random flip.
# Parameters of any given augmentation techniques can be set as shown below ( eg. 'noise_mean':0.1') otherwise default parameter settings will be applied.
#
# For training without augmentation, set 'augment = None'.

# %%
from nobrainer.intensity_transforms import addGaussianNoise
from nobrainer.spatial_transforms import randomflip_leftright
augment = [
    (addGaussianNoise, {'noise_mean': 0.1, 'noise_std': 0.5}),
    (randomflip_leftright, {}),
]

# %%
dataset_train, dataset_eval = DT.from_files(
    paths=filepaths,
    eval_size=0.1,
    tfrecdir="data/binseg",
    shard_size=3,
    augment=augment,
    shuffle_buffer_size=10,
    num_parallel_calls=None,
)

# %% [markdown]
# # Instantiate a neural network fro brain mask extraction

# %%
from nobrainer.processing.segmentation import Segmentation
from nobrainer.models import unet
model = Segmentation(unet, model_args=dict(batchnorm=True))


# %% [markdown]
# # Train and evaluate the model
#

# %%
history = model.fit(
    dataset_train=dataset_train,
    dataset_validate=dataset_eval,
    epochs=n_epochs,
)


# %% [markdown]
# ## Use the trained model to predict a binary brain mask

# %%
import matplotlib.pyplot as plt
from nilearn import plotting
from nobrainer.volume import standardize

image_path = filepaths[0][0]
out = model.predict(image_path, normalizer=standardize)
out.shape

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
