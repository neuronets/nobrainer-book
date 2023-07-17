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
# In this notebook, we will demonstrate how to train a brain mask extraction model using `nobrainer` with training data augmentation. Augmentation is useful for improving the robustness of neural network models when a limited amount of training examples are available. The basic idea is that, by applying certain transformations to the training data prior to feeding it through the model, one can expand the input space to cover situations unaccounted for in the base training set.
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

train_paths = filepaths[:9]
evaluate_paths = filepaths[9:]

# %% [markdown]
# # Convert medical images to TFRecords
# invalid = nobrainer.io.verify_features_labels(train_paths)
# assert not invalid
#
# invalid = nobrainer.io.verify_features_labels(evaluate_paths)
# assert not invalid


# %%
# !mkdir -p data

# %%
# Convert training and evaluation data to TFRecords.
nobrainer.tfrecord.write(
    features_labels=train_paths,
    filename_template='data/data-train_shard-{shard:03d}.tfrec',
    examples_per_shard=3)

nobrainer.tfrecord.write(
    features_labels=evaluate_paths,
    filename_template='data/data-evaluate_shard-{shard:03d}.tfrec',
    examples_per_shard=1)

# %%
# !ls data

# %% [markdown]
# # Create Datasets

# %%
n_classes = 1
batch_size = 2
volume_shape = (256, 256, 256)
block_shape = (32, 32, 32)
n_epochs = None
shuffle_buffer_size = 10
num_parallel_calls = 2

# %% [markdown]
# # Augmentation
# Take a look at different augmentation options in Nobrainer spatial and intensity transforms. To set training with multiple augmentations, the parameter `augment` will be set as a list where their order will determine the sequence of execution. For example augment option below will first add Gaussian noise and will then perform the random flip.
# Parameters of any given augmentation techniques can be set as shown below ( eg. 'noise_mean':0.1') otherwise default parameter settings will be applied.
#
# For training without augmentation, set 'augment = None'.

# %%
from nobrainer.intensity_transforms import addGaussianNoise
from nobrainer.spatial_transforms import randomflip_leftright

augment = [(addGaussianNoise, {'noise_mean':0.1,'noise_std':0.5}), (randomflip_leftright)]

# %%
dataset_train = nobrainer.dataset.get_dataset(
    file_pattern='data/data-train_shard-*.tfrec',
    n_classes=n_classes,
    batch_size=batch_size,
    volume_shape=volume_shape,
    block_shape=block_shape,
    n_epochs=n_epochs,
    augment=augment,
    shuffle_buffer_size=shuffle_buffer_size,
    num_parallel_calls=num_parallel_calls,
)

dataset_evaluate = nobrainer.dataset.get_dataset(
    file_pattern='data/data-evaluate_shard-*.tfrec',
    n_classes=n_classes,
    batch_size=batch_size,
    volume_shape=volume_shape,
    block_shape=block_shape,
    n_epochs=1,
    augment=False,
    shuffle_buffer_size=None,
    num_parallel_calls=1,
)

# %%
dataset_train

# %%
dataset_evaluate

# %% [markdown]
# # Instantiate a neural network

# %%
model = nobrainer.models.unet(
    n_classes=n_classes,
    input_shape=(*block_shape, 1),
    batchnorm=True,
)

# %%
model.summary()

# %% [markdown]
# # Choose a loss function and metrics
#
# We have many choices of loss functions for binary segmentation. One can choose from binary crossentropy, Dice, Jaccard, Tversky, and many other loss functions.

# %%
import tensorflow as tf

# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-04)

model.compile(
    optimizer=optimizer,
    loss=nobrainer.losses.dice,
    metrics=[nobrainer.metrics.dice, nobrainer.metrics.jaccard],
)

# %% [markdown]
# # Train and evaluate model
#
# $$
# steps = \frac{nBlocks}{volume} * \frac{nVolumes}{batchSize}
# $$

# %%
steps_per_epoch = nobrainer.dataset.get_steps_per_epoch(
    n_volumes=len(train_paths),
    volume_shape=volume_shape,
    block_shape=block_shape,
    batch_size=batch_size)

steps_per_epoch

# %%
validation_steps = nobrainer.dataset.get_steps_per_epoch(
    n_volumes=len(evaluate_paths),
    volume_shape=volume_shape,
    block_shape=block_shape,
    batch_size=batch_size)

validation_steps

# %%
model.fit(
    dataset_train,
    epochs=5,
    steps_per_epoch=steps_per_epoch,
    validation_data=dataset_evaluate,
    validation_steps=validation_steps)
