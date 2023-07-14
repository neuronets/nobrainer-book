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
#     name: python3
# ---

# %% [markdown] id="JZjOrKRnq5aa"
# # Getting started with Nobrainer
#
# Nobrainer is a deep learning framework for 3D image processing. It implements several 3D convolutional models from recent literature, methods for loading and augmenting volumetric data than can be used with any TensorFlow or Keras model, losses and metrics for 3D data, and utilities for model training, evaluation, prediction, and transfer learning.
#
# The code for the Nobrainer framework is available on GitHub at https://github.com/neuronets/nobrainer. The Nobrainer project is supported by NIH RF1MH121885 and previously by R01EB020470. It is distributed under the Apache 2.0 license.
#
# ## Questions or issues
#
# If you have questions about Nobrainer or encounter any issues using the framework, please [submit a GitHub issue](https://github.com/neuronets/helpdesk/issues/new/choose). If you have a feature request, we encourage you to submit a pull request.

# %% [markdown] id="bwaWX82pq5ad"
# # Using the guide notebooks
#
# Please install `nobrainer` before using these notebooks. You can learn how to do this below. Most of the notebooks also require some data on which to train or evaluate. You can use your own data, but `nobrainer` also provides a utility to download a small public dataset. Please refer to the notebook `02-preparing_training_data.ipynb` to download and prepare the example data for use or to prepare your own data for use.
#
# After you have gone through `02-preparing_training_data.ipynb`, please take a look at the other notebooks in this guide.
#
# ## Google Colaboratory
#
# These notebooks can be [run for free](https://colab.research.google.com/github/neuronets/nobrainer-book/blob/ohinds-guide-api) on Google Colaboratory (you must be signed into a Google account). If you are using Colab, please note that multiple open tabs of Colab notebooks will use the same resources (RAM, GPU). Downloading data in multiple Colab notebooks at the same time or training multiple models can quickly exhaust the available resources. For this reason, please run one notebook at a time, and keep an eye on the resources used.
#
# Users can choose to run Colab notebooks on CPU, GPU, or TPU. By default, the notebooks will use the CPU runtime. To use a different runtime, please select `Runtime > Change runtime type` in the menu bar. Then, choose either `GPU` or `TPU` under `Hardware accelerator`. No code changes are required when running on CPU or GPU runtime. When using the TPU runtime, however, special care must be taken for things to work properly. Please refer to the TPU guide notebook in this directory for more information.
#
# ## Jupyter Notebook
#
# These notebooks can use whatever hardware you have available, whether it is CPU, GPU, or TPU. Please note that training models on CPU can take a very long time. GPUs will greatly increase speed of training and inference. Some of the notebooks download example data, but you can feel free to use your own data.

# %% [markdown] id="1fO5RSFbq5ad"
# # Install Nobrainer
#
# Nobrainer can be installed using `pip`.

# %% id="_xELN1Hcq5ae"
# !pip install --no-cache-dir nobrainer

# %% [markdown] id="UVMXci7Mq5ae"
# # Accessing Nobrainer
#
# ## Command-line
#
# Nobrainer provides the command-line program `nobrainer`, which provides various methods for preparing data, training and evaluating models, generating predictions, etc. While the `nobrainer` command provides easy-to-access functionality to perform simple tasks, this guide is focused on using the `nobrainer` python API to demonstrate advanced, low-level training capabilities.

# %% id="HU6edUxDq5af"
# !nobrainer --help

# %% [markdown] id="AP6dWeNXq5af"
# ## Python
#
# The `nobrainer` Python package can be imported as below. This gives you access to all of nobrainer's modules.

# %% id="G8mYOLaWq5af"
import nobrainer

# %% [markdown] id="eKpEQi0Xq5ag"
# ### Layout
#
# - `nobrainer.dataset`: `tf.data.Dataset` creation utilities
# - `nobrainer.io`: input/output methods
# - `nobrainer.layers`: custom Keras layers
# - `nobrainer.losses`: loss functions for volumetric segmentation
# - `nobrainer.metrics`: metrics for volumetric segmentation
# - `nobrainer.models`: pre-defined Keras models
# - `nobrainer.processing`: API for neuroimaging model training, inference, and generation
# - `nobrainer.transform`: rigid transformations for data augmentation
# - `nobrainer.volume`: utilities for manipulating and augmenting volumetric data

# %% [markdown] id="VG0lFj2gq5ag"
# # Next steps
#
# The subsequent Jupyter notebooks in this guide demonstrate how to use `nobrainer` to prepare training data, train models, and more. These tutorial notebooks will be updated and enhanced regularly. If you think something is missing or could be improved, please [submit a GitHub issue](https://github.com/neuronets/helpdesk/issues/new/choose).
#
# ## Guides:
#
# - [Preparing training data](https://colab.research.google.com/github/neuronets/nobrainer-book/blob/ohinds-guide-api/docs/nobrainer-guides/notebooks/02-preparing_training_data.ipynb)
# - [Train brain extraction](https://colab.research.google.com/github/neuronets/nobrainer-book/blob/ohinds-guide-api/docs/nobrainer-guides/notebooks/03-train_brain_extraction.ipynb)
#
