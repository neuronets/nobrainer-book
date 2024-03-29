# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
# ---

# %% [markdown] id="ijHnNTIjDkt0"
# # Use checkpoints to resume training a model for binary volumetric brain extraction
#
# In the following cells, we will:
#
# 1. Get sample T1-weighted MR scans as features and FreeSurfer segmentations as labels.
# 2. Convert the data to TFRecords format for use with neural networks.
# 3. Create two `Datasets` of features and labels, one for training, one for evaluation.
# 4. Instantiate a brain segmentation model with checkpointing to store training results progressively.
# 5. Train a bit.
# 6. Load the partially trained model from disk and resume training.
# 7. Save the model to disk for future prediction and/or training.
# 8. Load the model back from disk and show that brain extraction works as before saving.


# %% [markdown]
# ## Google Colaboratory
#
# If you are using Colab, please switch your runtime to GPU. To do this, select `Runtime > Change runtime type` in the top menu. Then select GPU under `Hardware accelerator`. A GPU greatly speeds up training.


# %% [markdown]
# # Install and setup `nobrainer`

# %% id="WhBnt2WdDlx9"
# !pip install nobrainer nilearn

# %% id="Ht_CGSk1Dkt3"
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import nobrainer


# %% [markdown] id="hVCchp9uDkt3"
# # Get sample features and labels
#
# We use 9 pairs of volumes for training and 1 pair of volumes for evaluation. Many more volumes would be required to train a model for any useful purpose.

# %% id="YpqTxNu4Dkt4"
csv_path = nobrainer.utils.get_data()
filepaths = nobrainer.io.read_csv(csv_path)


# %% [markdown] id="ScgF78rmDkt4"
# # Convert medical images to TFRecords

# %% id="n7lCL-55Ta4R"
from nobrainer.dataset import Dataset


# %% id="Q3zPyRlbTa4R"
n_epochs = 2
dataset_train, dataset_eval = Dataset.from_files(
    filepaths,
    out_tfrec_dir="data/binseg",
    shard_size=3,
    num_parallel_calls=None,
    n_classes=1,
    block_shape=(128, 128, 128),
)

dataset_train.\
    shuffle(10).\
    repeat(n_epochs)
    


# %% [markdown]
# # U-Net model for brain mask extraction

# %% [markdown]
# ## Construct the model
# Set up the model to train in sessions, resuming from checkpoints each time. If no checkpoints exist in the specified location, training starts fresh.
#
# Here we'll train `nobrainer`'s implementation of the U-Net model for biomedical image segmentation, based on https://arxiv.org/abs/1606.06650.
#
# `nobrainer` provides several other segmentation models that could be used instead of `unet`. Another example is provided at the bottom of this guide, and for a complete list, see [this list](https://github.com/neuronets/nobrainer#models).
#
# Note that a useful segmentation model would need to be trained on *many* more examples than the 10 we are using here for demonstration.

# %% id="X8u_owicTa4T"
from nobrainer.processing.segmentation import Segmentation
from nobrainer.models import unet

model_dir = "brain_mask_extraction_model"
checkpoint_filepath = os.path.join(model_dir, "checkpoints", "epoch_{epoch:03d}")
bem = Segmentation.init_with_checkpoints(
    unet,
    model_args=dict(batchnorm=True),
    checkpoint_filepath=checkpoint_filepath,
)


# %% [markdown]
# ## Train the model on the example data
# A summary of the model layers is printed before training starts.
#
# Note that the loss function after training is very high, and the dice coefficient (a measure of the accuracy of the model) is very low, indicating that the model is not doing a good job of binary segmentation. This is expected, as this is a toy problem to demonstrate the API. During successful training of a more practical model, you would see the loss drop and the dice rise as training progressed.

# %%
history = bem.fit(
    dataset_train=dataset_train,
    dataset_validate=dataset_eval,
    epochs=n_epochs,
)


# %% [markdown]
# ## Use the trained model to predict a binary brain mask
# The segmentation is bad, but that isn't surprising given the small dataset and short training.

# %% id="OWqLu2xFTa4U"
import matplotlib.pyplot as plt
from nilearn import plotting
from nobrainer.volume import standardize

image_path = filepaths[0][0]
out = bem.predict(image_path, normalizer=standardize)
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


# %% [markdown]
# ## Train the model a bit more, picking up where the last training session left off.
# This paradigm is useful in situations where training takes a long time and compute resources may be preemptable or available in chunks.

# %%
bem = Segmentation.init_with_checkpoints(
    unet,
    model_args=dict(batchnorm=True),
    checkpoint_filepath=checkpoint_filepath,
)
history = bem.fit(
    dataset_train=dataset_train,
    dataset_validate=dataset_eval,
    epochs=n_epochs,
)


# %% [markdown]
# ## Save the trained model

# %%
bem.save(model_dir)


# %% [markdown]
# ## Load the model from disk for prediction.

# %%
bem = Segmentation.load(model_dir)


# %% [markdown]
# ## Predict a brain mask from the loaded model
# The brain mask is identical to that predicted before saving.

# %%
out = bem.predict(image_path, normalizer=standardize)
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
