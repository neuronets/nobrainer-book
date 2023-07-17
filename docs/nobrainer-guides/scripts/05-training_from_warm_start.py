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
# # Train a neural network for binary volumetric brain extraction
#
# In this notebook, we will use the `nobrainer` python API to train a model for brain extraction. Brain extraction is a common step in processing neuroimaging data. It is a voxel-wise, binary classification task, where each voxel is classified as brain or not brain.
#
# In the following cells, we will:
#
# 1. Get sample T1-weighted MR scans as features and FreeSurfer segmentations as labels.
# 2. Convert the data to TFRecords format for use with neural networks.
# 3. Create two `Datasets` of features and labels, one for training, one for evaluation.
# 4. Instantiate a 3D convolutional neural network model for image segmentation (U-Net).
# 5. Train on part of the data and evaluate on the rest of the data.
# 6. Predict a brain mask using the trained model.
# 7. Save the model to disk for future prediction and/or training.
# 8. Load the model back from disk and show that brain extraction works as before saving.
#
#
# ## Google Colaboratory
#
# If you are using Colab, please switch your runtime to GPU. To do this, select `Runtime > Change runtime type` in the top menu. Then select GPU under `Hardware accelerator`. A GPU greatly speeds up training.


# %% id="WhBnt2WdDlx9"
# !pip install nobrainer nilearn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import nobrainer

csv_path = nobrainer.utils.get_data()
filepaths = nobrainer.io.read_csv(csv_path)

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

from nobrainer.processing.segmentation import Segmentation
from nobrainer.models import unet

bem = Segmentation(unet, model_args=dict(batchnorm=True))

history = bem.fit(dataset_train=dataset_train,
                  dataset_validate=dataset_eval,
                  epochs=n_epochs,
                  multi_gpu=True,
                  )

bem.save("data/unet-brainmask-toy")


# %% [markdown]
# ## Load the model from disk

# %%
bem = Segmentation.load("data/unet-brainmask-toy")


# %% [markdown]
# ## Restart training where it left off

# %%
history = bem.fit(dataset_train=dataset_train,
                  dataset_validate=dataset_eval,
                  epochs=n_epochs,
                  multi_gpu=True,
                  warm_start=True
                  )
