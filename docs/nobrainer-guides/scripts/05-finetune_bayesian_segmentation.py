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

# %% [markdown] id="kNguCLDMCDxD"
# # Train a neural network with very little data
#
# In this notebook, we will use Nobrainer to train a Bayesian neural network with limited data. We will start off with a pre-trained model. You can find available pre-trained Nobrainer models at https://github.com/neuronets/nobrainer-models.
#
# The pre-trained models can be used to train models for the same task as they were trained for or to transfer learn a new task. For instance, a pre-trained brain labelling model can be re-trained for tumor labeling. In this notebook, we will train a brain labeling model, but keep in mind that you can retrain these models for many 3D semantic segmentation tasks.
#
# In the following cells, we will:
#
# 1. Get sample T1-weighted MR scans as features and FreeSurfer segmentations as labels.
# 2. Convert the data to TFRecords format.
# 3. Create two Datasets of the features and labels.
#     - One dataset will be for training and the other will be for evaluation.
# 4. Load a pre-trained 3D semantic segmentation Bayesian model.
# 5. Choose a loss function and metrics to use.
# 6. Train on part of the data.
# 7. Evaluate on the rest of the data.
#
# ## Google Colaboratory
#
# If you are using Colab, please switch your runtime to GPU. To do this, select `Runtime > Change runtime type` in the top menu. Then select GPU under `Hardware accelerator`. A GPU greatly speeds up training.

# %% id="JQ-P_PzHCJb6"
# !pip install nobrainer

# %% id="O8zmcBiACDxI"
import nobrainer

# %% [markdown] id="kfHn42x_CDxJ"
# # Get sample features and labels
#
# We use 9 pairs of volumes for training and 1 pair of volumes for evaulation. Many more volumes would be required to train a model for any useful purpose.

# %% id="WnBCrxRUCDxK"
csv_of_filepaths = nobrainer.utils.get_data()
filepaths = nobrainer.io.read_csv(csv_of_filepaths)

train_paths = filepaths[:9]
evaluate_paths = filepaths[9:]

# %% [markdown] id="6dsTOMVhCDxL"
# # Convert medical images to TFRecords
#
# Remember how many full volumes are in the TFRecords files. This will be necessary to know how many steps are in on training epoch. The default training method needs to know this number, because Datasets don't always know how many items they contain.

# %% id="lHefZL7XCDxL"
# Verify that all volumes have the same shape and that labels are integer-ish.

invalid = nobrainer.io.verify_features_labels(train_paths, num_parallel_calls=2)
assert not invalid

invalid = nobrainer.io.verify_features_labels(evaluate_paths)
assert not invalid

# %% id="FMH1QG1ACDxL"
# !mkdir -p data

# %% id="x1AdoUfLCDxM"
# Convert training and evaluation data to TFRecords.

nobrainer.tfrecord.write(
    features_labels=train_paths,
    filename_template='data/data-train_shard-{shard:03d}.tfrec',
    examples_per_shard=3)

nobrainer.tfrecord.write(
    features_labels=evaluate_paths,
    filename_template='data/data-evaluate_shard-{shard:03d}.tfrec',
    examples_per_shard=1)

# %% id="8Q5FvIGUCDxN"
# !ls data

# %% [markdown] id="-4hBG14lCDxO"
# # Create Datasets

# %% id="pkOvJBT8CDxO"
import tensorflow as tf

n_classes = 50
batch_size = 2
volume_shape = (256, 256, 256)
block_shape = (32, 32, 32)
n_epochs = 2

def _to_blocks(x, y):
    """Separate `x` into blocks and repeat `y` by number of blocks."""
    x = nobrainer.volume.to_blocks(x, block_shape)
    y = nobrainer.volume.to_blocks(y, block_shape)
    return (x, y)

def process_dataset(dset):
    # Standard score the features.
    dset = dset.map(lambda x, y: (nobrainer.volume.standardize(x), y))
    # Separate features into blocks.
    dset = dset.map(_to_blocks)
    # This step is necessary because separating into blocks adds a dimension.
    dset = dset.unbatch()
    # Add a grayscale channel to the features.
    dset = dset.map(lambda x, y: (tf.expand_dims(x, -1), y))
    # Batch features and labels.
    dset = dset.batch(batch_size, drop_remainder=True)
    dset = dset.repeat(n_epochs)
    return dset

# Create a `tf.data.Dataset` instance.
dataset_train = nobrainer.dataset.tfrecord_dataset(
    file_pattern="data/data-train_shard-*.tfrec",
    volume_shape=volume_shape,
    shuffle=True,
    scalar_label=False,
    num_parallel_calls=None,
)
dataset_train = process_dataset(dataset_train)

# Create a `tf.data.Dataset` instance.
dataset_evaluate = nobrainer.dataset.tfrecord_dataset(
    file_pattern="data/data-evaluate_shard-*.tfrec",
    volume_shape=volume_shape,
    shuffle=False,
    scalar_label=False,
    num_parallel_calls=None,
)
dataset_evaluate = process_dataset(dataset_evaluate)

# %% id="Y4pabFvcCDxP"
dataset_train

# %% id="zBKDgf93CDxP"
dataset_evaluate

# %% [markdown] id="XScYNuNoCDxP"
# # Load pre-trained model

# %% id="STBCfp_NCDxQ"
import tensorflow as tf

from nobrainer.models.bayesian import variational_meshnet

# %% id="Dyz8SXmCCDxQ"
model = variational_meshnet(
    n_classes=50,
    input_shape=(32, 32, 32, 1),
    filters=96,
    dropout="concrete",
    receptive_field=37,
    is_monte_carlo=True,
)

# %% id="69XPBEAzCDxR"
weights_path = tf.keras.utils.get_file(
    fname="nobrainer_spikeslab_32iso_weights.h5",
    origin="https://dl.dropbox.com/s/rojjoio9jyyfejy/nobrainer_spikeslab_32iso_weights.h5")

model.load_weights(weights_path)

# %% id="_Zn_WWLgCDxR"
model.summary()

# %% [markdown] id="Sr_SsvDkCDxR"
# # Considerations for transfer learning
#
# Training a neural network changes the model's weights. A pre-trained network has learned weights for a task, and we do not want to forget these weights during training. In other words, we do not want to ruin the pre-trained weights when using our new data. To avoid dramatic changes in the learnable parameters, we can use a relatively small learning rate.
#
# We also want to optimize the evidence lower bound ([ELBO](https://en.wikipedia.org/wiki/Evidence_lower_bound)). Specifically, we will minimize $-ELBO$.

# %% id="QjIgEvuPCDxS"
import numpy as np

loss_fn = nobrainer.losses.ELBO(model=model, num_examples=np.prod(block_shape))

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-06)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    # See https://github.com/tensorflow/probability/issues/519
    experimental_run_tf_function=False
)

# %% [markdown] id="rmGh44FTCDxS"
# # Train and evaluate model
#
# $$
# steps = \frac{nBlocks}{volume} * \frac{nVolumes}{batchSize}
# $$

# %% id="0ZeMaDNaCDxS"
steps_per_epoch = nobrainer.dataset.get_steps_per_epoch(
    n_volumes=len(train_paths),
    volume_shape=volume_shape,
    block_shape=block_shape,
    batch_size=batch_size)

steps_per_epoch

# %% id="GxFb0nKaCDxS"
validation_steps = nobrainer.dataset.get_steps_per_epoch(
    n_volumes=len(evaluate_paths),
    volume_shape=volume_shape,
    block_shape=block_shape,
    batch_size=batch_size)

validation_steps

# %% id="5IRNkG-bCDxT"
model.fit(
    dataset_train,
    epochs=n_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=dataset_evaluate,
    validation_steps=validation_steps)

# %% id="Z23HRhuLDRjS"
