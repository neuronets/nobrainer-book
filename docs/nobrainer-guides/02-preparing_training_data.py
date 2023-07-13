# ---
# kernelspec:
#   display_name: Python 3
#   name: python3
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
# ---

# %% [markdown] id="8g2WKuIv5tPg"
# # Preparing training data
#
# In this tutorial, we will convert medical imaging data to the TFRecords format. Having data in the TFRecords format speeds up training and allows us to use standard, highly-optimized TensorFlow I/O methods. We will then create a `tf.data.Dataset` object using the TFRecords data. This `tf.data.Dataset` object can be used for training, evaluation, or prediction.
#
# This tutorial will use a small publicly available dataset. To convert your own data, you will need to create a nested list of features and labels volumes (see below).

# %% bash
pip install --no-cache-dir nobrainer

# %% id="egda7m595tPi"
import nobrainer

# %% [markdown] id="HeWmDZXq5tPj"
# ## Get sample data
#
# Here, we download 10 T1-weighted brain scans and their corresponding FreeSurfer segmentations. These volumes take up about 46 MB and are saved to a temporary directory. The returned string `csv_path` is the path to a CSV file, each row of which contains the paths to a pair of features and labels volumes.

# %% id="U1DD5tCh5tPk"
csv_path = nobrainer.utils.get_data()
filepaths = nobrainer.io.read_csv(csv_of_filepaths)
# !cat {csv}


# %% [markdown]
# ### Visualize one training example, with the brainmask overlayed on the anatomical image.

# %%
import matplotlib.pyplot as plt
from nilearn import plotting
fig = plt.figure(figsize=(12, 6))
plotting.plot_roi(filepaths[0][1], bg_img=filepaths[0][0], alpha=0.4, vmin=0, vmax=1.5, figure=fig)

# %% [markdown] id="rm8aVxsc5tPk"
# ## Convert to volume files to TFRecords
#
# To achieve the best performance, training data should be in TFRecords format. This is the preferred file format for TensorFlow, Training can be done on medical imaging volume files but will be slower.
#

# %%
from nobrainer.dataset import Dataset

n_epochs = 2
DT = Dataset(n_classes=1,
             batch_size=2,
             block_shape=(128, 128, 128),
             n_epochs=n_epochs)

dataset_train, dataset_eval = DT.from_files(paths=filepaths,
                                            eval_size=0.1,
                                            tfrecdir="data/binseg",
                                            shard_size=3,
                                            augment=None,
                                            shuffle_buffer_size=10,
                                            num_parallel_calls=None)


# %% [markdown] id="9oG79IlJ5tPl"
# # Create input data pipeline
#
# We will now create an data pipeline to feed our models with training data. The steps below will create a `tensorflow.data.Dataset` object that is built according to [TensorFlow's guidelines](https://www.tensorflow.org/guide/performance/datasets). The basic pipeline is summarized below.
#
# - Read data
# - Separate volumes into non-overlapping sub-volumes
#     - This is done to get around memory limitations with larger models.
#     - For example, a volume with shape (256, 256, 256) can be separated into eight non-overlapping blocks of shape (128, 128, 128).
# - Apply random rigid augmentations if requested.
# - Standard score volumes of features.
# - Binarize labels if binary segmentation.
# - Replace values according to some mapping if multi-class segmentation.
# - Batch the results so every iteration yields `batch_size` elements.

# %% id="Au6YBWec5tPm"
# A glob pattern to match the files we want to train on.
file_pattern = 'data/data_shard-*.tfrec'

# The number of classes the model predicts. A value of 1 means the model performs
# binary classification (i.e., target vs background).
n_classes = 1

# Batch size is the number of features and labels we train on with each step.
batch_size = 2

# The shape of the original volumes.
volume_shape = (256, 256, 256)

# The shape of the non-overlapping sub-volumes. Most models cannot be trained on
# full volumes because of hardware and memory constraints, so we train and evaluate
# on sub-volumes.
block_shape = (128, 128, 128)

# Whether or not to apply random rigid transformations to the data on the fly.
# This can improve model generalizability but increases processing time.
augment = False

# The tfrecords filepaths will be shuffled before reading, but we can also shuffle
# the data. This will shuffle 10 volumes at a time. Larger buffer sizes will require
# more memory, so choose a value based on how much memory you have available.
shuffle_buffer_size = 10

# Number of parallel processes to use.
num_parallel_calls = 6

# %% id="gOLBC_8R5tPn"
# !ls $file_pattern

# %% id="3qaZ8eoe5tPn"
dataset = nobrainer.dataset.get_dataset(
    file_pattern=file_pattern,
    n_classes=n_classes,
    batch_size=batch_size,
    volume_shape=volume_shape,
    block_shape=block_shape,
    augment=augment,
    n_epochs=1,
    shuffle_buffer_size=shuffle_buffer_size,
    num_parallel_calls=num_parallel_calls)

dataset

# %% [markdown] id="EM8fATVF5tPn"
# # Train a model
#
# We will briefly demonstrate how to train a model given the `tf.data.Dataset` we created.

# %% [markdown] id="qYrNoHSx5tPo"
# ## Instantiate a pre-defined `nobrainer` model
#
# Users can find pre-defined models under the namespace `nobrainer.models`. All models are implemented using the `tf.keras` API, which makes definitions highly readable and hackable, despite being a high-level interface.

# %% id="iPDE6n8e5tPo"
model = nobrainer.models.unet(n_classes=n_classes, input_shape=(*block_shape, 1))

# %% [markdown] id="iEo-bJAo5tPo"
# ## Compile the model
#
# All Keras models must be compiled before they can be trained. This is where you choose the optimizer, loss function, and any metrics that should be reported during training. Nobrainer implements several loss functions useful for semantic segmentation, including Dice, Generalized Dice, Focal, Jaccard, and Tversky losses.

# %% id="YvhD-oAf5tPo"
import tensorflow as tf

# %% id="FfB-X1RP5tPp"
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-04),
    loss=nobrainer.losses.jaccard,
    metrics=[nobrainer.metrics.dice])

# %% [markdown] id="bhXfMXI_5tPp"
# ## Train on a single GPU
#
# To learn how to train on multiple GPUs or on a TPU, please refer to the other notebooks in the nobrainer guide.

# %% id="SGhxkll75tPp"
steps_per_epoch = nobrainer.dataset.get_steps_per_epoch(
    n_volumes=10,
    volume_shape=volume_shape,
    block_shape=block_shape,
    batch_size=batch_size)

steps_per_epoch

# %% [markdown] id="S_rIE6pL7b2O"
# The following code may take some time to initialize and on a GPU (if enabled) will take about a minute and a half to run.

# %% id="IIYcof_M5tPp"
model.fit(dataset, steps_per_epoch=steps_per_epoch)
