# ---
# jupyter:
#   jupytext:
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


# %%
# !pip install nobrainer nilearn

# %% id="egda7m595tPi"
import nobrainer


# %% [markdown] id="HeWmDZXq5tPj"
# ## Get sample data
#
# Here, we download 10 T1-weighted brain scans and their corresponding FreeSurfer segmentations. These volumes take up about 46 MB and are saved to a temporary directory. The returned string `csv_path` is the path to a CSV file, each row of which contains the paths to a pair of features and labels volumes.

# %% id="U1DD5tCh5tPk"
csv_path = nobrainer.utils.get_data()
filepaths = nobrainer.io.read_csv(csv_path)
# !cat {csv_path}


# %% [markdown]
# ### Visualize one training example, with the brainmask overlayed on the anatomical image.

# %%
import matplotlib.pyplot as plt
from nilearn import plotting
fig = plt.figure(figsize=(12, 6))
plotting.plot_roi(filepaths[0][1], bg_img=filepaths[0][0], alpha=0.4, vmin=0, vmax=1.5, figure=fig)


# %% [markdown] id="rm8aVxsc5tPk"
# ## Convert the raw volumes to TFRecords

# %%
from nobrainer.dataset import Dataset

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
# You are now ready to train a model using this example dataset!
# Next, learn how to [train a brain extraction model](https://colab.research.google.com/github/neuronets/nobrainer-book/blob/master/docs/nobrainer-guides/notebooks/03-train_brain_extraction.ipynb)
