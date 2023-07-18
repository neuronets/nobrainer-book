# `nobrainer` Python API Guides

This set of guides demonstrates basic usage of the `nobrainer` python API to train and use neural network models for neuroimaging. The `nobrainer` project was developed under the support of NIH RF1MH121885 and R01EB020470. It is distributed under the Apache 2.0 license.

## Accessing the guides

These guides are available in both jupyter notebook versions and as standalone python scripts.

### Jupyter notebooks [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuronets/nobrainer)


See the [notebooks](notebooks) directory for the guides in jupyter notebook format. These notebooks can be run for free on Google Colab, accessed [here](https://colab.research.google.com/github/neuronets/nobrainer-book/blob/master).

### Standalone python scripts

See the [scripts](scripts) directory for the guides in python format. These notebooks can be run anywhere with a an installation of python >= 3.8.

## List of guides

1. Getting started ([notebook](notebooks/01-getting_started.ipynb), [python](scripts/01-getting_started.py))
   Basic  `nobrainer` info and installation
2. Preparing training data ([notebook](notebooks/02-preparing_training_data.ipynb), [python](scripts/02-preparing_training_data.py))
   How to obtain some example neuroimaging data and prepare it for neural network training.
3. Train brain extraction ([notebook](notebooks/03-train_brain_extraction.ipynb), [python](scripts/03-train_brain_extraction.py))
   Train a standard brain extraction model on the example dataset.
4. Train brain volume generation ([notebook](notebooks/04-train_brain_generation.ipynb), [python](scripts/03-train_brain_generation.py))
   Train a model to generate synthetic brain volumes on the example dataset.
5. Augmentation of training data ([notebook](notebooks/05-training_with_augmentation.ipynb), [python](scripts/05-training_with_augmentation.py))
   Use augmented training data to train a standard brain extraction model on the example dataset.

## Adding to the guide

These guides are maintained in parallel using `jupytext`. Edits should be made to the python files in the [scripts](scripts) directory, then notebooks can be generated via
```
jupytext --sync scripts/<python-file>
```
