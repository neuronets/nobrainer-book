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

# %% [markdown]
# # Getting started with Nobrainer
#
# Nobrainer is a deep learning framework for 3D image processing built on
# PyTorch and MONAI. It implements several 3D convolutional models from recent
# literature, methods for loading and augmenting volumetric data, losses and
# metrics for 3D data, and utilities for model training, evaluation, prediction,
# and transfer learning.
#
# The code for the Nobrainer framework is available on GitHub at
# https://github.com/neuronets/nobrainer. The Nobrainer project is supported by
# NIH RF1MH121885 and previously by R01EB020470. It is distributed under the
# Apache 2.0 license.
#
# ## Questions or issues
#
# If you have questions about Nobrainer or encounter any issues using the
# framework, please
# [submit a GitHub issue](https://github.com/neuronets/helpdesk/issues/new/choose).
# If you have a feature request, we encourage you to submit a pull request.

# %% [markdown]
# # Using the guide notebooks
#
# Please install `nobrainer` before using these notebooks. You can learn how to
# do this below. Most of the notebooks also require some data on which to train
# or evaluate. You can use your own data, but `nobrainer` also provides a
# utility to download a small public dataset. Please refer to the notebook
# `02-preparing_training_data.ipynb` to download and prepare the example data
# for use or to prepare your own data for use.
#
# After you have gone through `02-preparing_training_data.ipynb`, please take a
# look at the other notebooks in this guide.
#
# ## Google Colaboratory
#
# These notebooks can be
# [run for free](https://colab.research.google.com/github/neuronets/nobrainer-book/blob/master)
# on Google Colaboratory (you must be signed into a Google account). If you are
# using Colab, please note that multiple open tabs of Colab notebooks will use
# the same resources (RAM, GPU). Downloading data in multiple Colab notebooks at
# the same time or training multiple models can quickly exhaust the available
# resources. For this reason, please run one notebook at a time, and keep an eye
# on the resources used.
#
# Users can choose to run Colab notebooks on CPU or GPU. By default, the
# notebooks will use the CPU runtime. To use a GPU runtime, please select
# `Runtime > Change runtime type` in the menu bar. Then, choose `GPU` under
# `Hardware accelerator`. No code changes are required -- PyTorch will
# automatically detect and use the available hardware.
#
# ## Jupyter Notebook
#
# These notebooks can use whatever hardware you have available, whether it is
# CPU or GPU. Please note that training models on CPU can take a very long time.
# GPUs will greatly increase speed of training and inference. Some of the
# notebooks download example data, but you can feel free to use your own data.

# %% [markdown]
# # Install Nobrainer
#
# Nobrainer can be installed using `pip`.

# %%
# !uv pip install --pre nobrainer

# %% [markdown]
# # Accessing Nobrainer
#
# ## Command-line
#
# Nobrainer provides the command-line program `nobrainer`, which provides
# various methods for preparing data, training and evaluating models, generating
# predictions, etc. While the `nobrainer` command provides easy-to-access
# functionality to perform simple tasks, this guide is focused on using the
# `nobrainer` python API to demonstrate advanced, low-level training
# capabilities.

# %%
# !nobrainer --help

# %% [markdown]
# ## Python
#
# The `nobrainer` Python package can be imported as below. This gives you
# access to all of nobrainer's modules.

# %%
import torch

import nobrainer

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")

# %% [markdown]
# ### Layout
#
# - `nobrainer.io`: input/output methods
# - `nobrainer.models`: pre-defined PyTorch models (segmentation, generative,
#   Bayesian)
# - `nobrainer.prediction`: block-based inference and uncertainty estimation
# - `nobrainer.training`: training utilities with optional multi-GPU DDP support
# - `nobrainer.transform`: rigid transformations for data augmentation
# - `nobrainer.volume`: utilities for manipulating and augmenting volumetric
#   data
# - `nobrainer.utils`: data download and miscellaneous helpers

# %% [markdown]
# # Next steps
#
# The subsequent notebooks in this guide demonstrate how to use `nobrainer` to
# prepare training data, train models, and more. These tutorial notebooks will
# be updated and enhanced regularly. If you think something is missing or could
# be improved, please
# [submit a GitHub issue](https://github.com/neuronets/helpdesk/issues/new/choose).
#
# Now, learn how to
# [prepare data for training](https://colab.research.google.com/github/neuronets/nobrainer-book/blob/master/docs/nobrainer-guides/notebooks/02-preparing_training_data.ipynb)
