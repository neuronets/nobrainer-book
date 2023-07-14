# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# %% [markdown] id="ijHnNTIjDkt0" pycharm={"name": "#%% md\n"}
# # Train a neural network for binary volumetric brain mask segmentation
#
# In this notebook, we will use Nobrainer to train a model for brain extraction. Brain extraction is a common step in processing neuroimaging data. It is a voxel-wise, binary classification task, where each voxel is classified as brain or not brain.
#
# In the following cells, we will:
#
# 1. Get sample T1-weighted MR scans as features and FreeSurfer segmentations as labels.
#     - We will binarize the FreeSurfer to get a precise brainmask.
# 2. Convert the data to TFRecords format.
# 3. Create two Datasets of the features and labels.
#     - One dataset will be for training and the other will be for evaluation.
# 4. Instantiate a 3D convolutional neural network.
# 5. Choose a loss function and metrics to use.
# 6. Train on part of the data.
# 7. Evaluate on the rest of the data.
#
# ## Google Colaboratory
#
# If you are using Colab, please switch your runtime to GPU. To do this, select `Runtime > Change runtime type` in the top menu. Then select GPU under `Hardware accelerator`. A GPU greatly speeds up training.


# %% id="WhBnt2WdDlx9"
# !pip install nobrainer nilearn
# !export TF_CPP_MIN_LOG_LEVEL=2

# %% id="Ht_CGSk1Dkt3"
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


# %% [markdown]
# ## Construct a U-Net model
# Here we'll train nobrainer's implementation of the U-Net model for biomendical image segmentation, based on https://arxiv.org/abs/1606.06650. Note that a useful segmentation model would need to be trained on *many* more examples than the 10 we are using here for demonstration.

# %% id="X8u_owicTa4T"
from nobrainer.processing.segmentation import Segmentation
from nobrainer.models import unet

bem = Segmentation(unet, model_args=dict(batchnorm=True))
bem.fit(dataset_train=dataset_train,
        dataset_validate=dataset_eval,
        epochs=n_epochs,
        multi_gpu=True,
       )


# %% [markdown]
# ## Use the trained model to predict the binary brian mask
#

# %% id="OWqLu2xFTa4U"
from nobrainer.volume import standardize

image_path = filepaths[0][0]
out = bem.predict(image_path, normalizer=standardize)
out.shape

# %% id="4xJxR7Ddbd-0"
fig = plt.figure(figsize=(12, 6))
plotting.plot_roi(out, bg_img=image_path, alpha=0.4, vmin=0, vmax=5, figure=fig)

# %% [markdown]
# 1. Save model
# 2. Load model back as a class instance
# 3. Perform prediction
# 4. Continue training

# %%
bem.save("data/testsave")

# %%
from nobrainer.processing.segmentation import Segmentation
bem = Segmentation.load("data/testsave")

# %%
image_path = filepaths[0][0]
out = bem.predict(image_path, normalizer=standardize)
out.shape

# %%
bem.fit(dataset_train=dataset_train,
        dataset_validate=dataset_eval,
        epochs=1,
        multi_gpu=True,
        warm_start=True,
       )