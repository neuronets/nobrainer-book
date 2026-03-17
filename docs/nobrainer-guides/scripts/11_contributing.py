# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Contributing to Nobrainer
#
# This guide covers the nobrainer project structure, how to add new models,
# the testing framework, and the contribution workflow.

# %%
PRE_RELEASE = False
import subprocess, sys
try:
    import google.colab  # noqa: F401
    cmd = [sys.executable, "-m", "pip", "install", "-q",
           "nobrainer", "nilearn", "matplotlib"]
    if PRE_RELEASE:
        cmd.insert(4, "--pre")
    subprocess.check_call(cmd)
except ImportError:
    pass

# %% [markdown]
# ## 1. Project structure
#
# ```
# nobrainer/
# +-- nobrainer/
# |   +-- __init__.py           # Package init, version
# |   +-- dataset.py            # Low-level dataset utilities (get_dataset, ZarrDataset)
# |   +-- io.py                 # I/O: nifti_to_zarr, zarr_to_nifti, read/write
# |   +-- losses.py             # Loss functions (Dice, Jaccard, etc.)
# |   +-- metrics.py            # Evaluation metrics
# |   +-- prediction.py         # predict(), predict_with_uncertainty()
# |   +-- training.py           # fit() -- core training loop with DDP
# |   +-- utils.py              # get_data(), StreamingStats, utilities
# |   +-- layers/               # Custom PyTorch layers
# |   +-- models/               # Model zoo
# |   |   +-- __init__.py       # Registry: get(), available_models()
# |   |   +-- segmentation.py   # unet, vnet, attention_unet, unetr
# |   |   +-- bayesian.py       # bayesian_vnet, bayesian_meshnet (requires pyro)
# |   |   +-- generative.py     # progressivegan, dcgan (requires lightning)
# |   |   +-- meshnet.py        # MeshNet
# |   |   +-- highresnet.py     # HighResNet
# |   |   +-- autoencoder.py    # Autoencoder
# |   |   +-- simsiam.py        # SimSiam (self-supervised)
# |   +-- processing/           # High-level sklearn-style API
# |   |   +-- base.py           # BaseEstimator (save/load with Croissant-ML)
# |   |   +-- segmentation.py   # Segmentation estimator
# |   |   +-- generation.py     # Generation estimator
# |   |   +-- dataset.py        # Dataset builder, extract_patches, PatchDataset
# |   |   +-- croissant.py      # Croissant-ML metadata writers
# |   +-- tests/
# |   |   +-- unit/             # Fast unit tests (no GPU, no network)
# |   |   +-- integration/      # Integration tests (may need GPU)
# |   +-- sr-tests/             # Self-referential tests (model outputs validated)
# +-- setup.cfg                 # Package metadata, extras: [bayesian], [zarr], etc.
# +-- pyproject.toml            # Build system configuration
# ```

# %% [markdown]
# ## 2. How to add a new model
#
# ### Step 1: Create the model module
#
# Add a new file in `nobrainer/models/` (e.g., `my_model.py`):
#
# ```python
# """My custom segmentation model."""
# import torch.nn as nn
#
#
# class MyModel(nn.Module):
#     def __init__(self, in_channels=1, n_classes=2, **kwargs):
#         super().__init__()
#         # Build your architecture here
#         self.conv = nn.Conv3d(in_channels, n_classes, 3, padding=1)
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# def my_model(**kwargs):
#     """Factory function for MyModel."""
#     return MyModel(**kwargs)
# ```
#
# ### Step 2: Register in the model zoo
#
# Edit `nobrainer/models/__init__.py`:
#
# ```python
# from .my_model import my_model
#
# _models = {
#     ...
#     "my_model": my_model,
# }
# ```
#
# ### Step 3: Add tests
#
# Create `nobrainer/tests/unit/test_my_model.py`:
#
# ```python
# import torch
# from nobrainer.models import get as get_model
#
#
# def test_my_model_forward():
#     factory = get_model("my_model")
#     model = factory(in_channels=1, n_classes=2)
#     x = torch.randn(1, 1, 16, 16, 16)
#     out = model(x)
#     assert out.shape == (1, 2, 16, 16, 16)
# ```
#
# ### Step 4: Verify with the estimator API
#
# Your model automatically works with `Segmentation`:
#
# ```python
# seg = Segmentation("my_model", model_args={"in_channels": 1})
# seg.fit(dataset, epochs=5)
# ```

# %% [markdown]
# ## 3. Testing framework
#
# ### Unit tests
#
# Fast tests that run without GPU or network access:
#
# ```bash
# # Run all unit tests
# python -m pytest nobrainer/tests/unit/ -v
#
# # Run a specific test file
# python -m pytest nobrainer/tests/unit/test_models.py -v
#
# # Run tests matching a pattern
# python -m pytest nobrainer/tests/unit/ -k "test_unet" -v
# ```
#
# ### GPU tests
#
# Tests marked with `@pytest.mark.gpu` are automatically skipped when CUDA
# is unavailable:
#
# ```python
# import pytest
# import torch
#
# @pytest.mark.gpu
# def test_training_on_gpu():
#     # This test only runs when CUDA is available
#     ...
# ```
#
# The `conftest.py` auto-skip mechanism handles this -- no manual skipping
# needed.
#
# ### Self-referential (sr) tests
#
# Located in `nobrainer/sr-tests/`, these validate that model outputs
# match expected results. They are more expensive and typically run in CI
# on GPU instances:
#
# ```bash
# python -m pytest nobrainer/sr-tests/ -v
# ```

# %% [markdown]
# ## 4. Pre-commit hooks
#
# Nobrainer uses pre-commit for code quality. The hooks run automatically
# before each commit:
#
# ```bash
# # Install pre-commit hooks
# pre-commit install
#
# # Run hooks manually on all files
# pre-commit run --all-files
#
# # Run on specific files
# pre-commit run --files nobrainer/models/my_model.py
# ```
#
# The hooks include:
# - **ruff**: Fast Python linter (replaces flake8, isort, pyflakes)
# - **ruff-format**: Code formatting (replaces black)
# - **trailing-whitespace**: Remove trailing whitespace
# - **end-of-file-fixer**: Ensure files end with a newline
# - **check-yaml**: Validate YAML syntax

# %% [markdown]
# ## 5. Pull request workflow
#
# Nobrainer uses a branching model with three long-lived branches:
#
# ```
# master (stable releases)
#   ^
#   |  merge for release
# alpha (integration branch)
#   ^
#   |  PR from feature branch
# feature/my-feature
# ```
#
# ### Step-by-step:
#
# 1. **Fork and clone** the repository
#
# 2. **Create a feature branch** from `alpha`:
#    ```bash
#    git checkout alpha
#    git pull origin alpha
#    git checkout -b feature/my-feature
#    ```
#
# 3. **Make your changes** and commit with clear messages:
#    ```bash
#    git add nobrainer/models/my_model.py nobrainer/tests/unit/test_my_model.py
#    git commit -m "feat(models): add MyModel segmentation architecture"
#    ```
#
# 4. **Run tests** locally:
#    ```bash
#    python -m pytest nobrainer/tests/unit/ -v
#    pre-commit run --all-files
#    ```
#
# 5. **Push and create a PR** targeting `alpha`:
#    ```bash
#    git push origin feature/my-feature
#    # Create PR on GitHub: feature/my-feature -> alpha
#    ```
#
# 6. **CI checks** run automatically (unit tests, linting, GPU tests)
#
# 7. After review and merge to `alpha`, changes are **released** by merging
#    `alpha` into `master`

# %% [markdown]
# ## 6. Release workflow
#
# Releases follow this process:
#
# 1. All features merged to `alpha` are tested together
# 2. When ready, `alpha` is merged to `master`
# 3. A version tag is created on `master`
# 4. CI builds and publishes the package to PyPI
# 5. Docker images are built and pushed to Docker Hub
#
# Version numbering follows [Semantic Versioning](https://semver.org/):
# - **MAJOR**: breaking API changes
# - **MINOR**: new features (backward compatible)
# - **PATCH**: bug fixes

# %% [markdown]
# ## 7. Quick reference
#
# | Task | Command |
# |------|---------|
# | Install dev deps | `uv pip install -e ".[dev,test]"` |
# | Run unit tests | `python -m pytest nobrainer/tests/unit/ -v` |
# | Run linter | `ruff check nobrainer/` |
# | Format code | `ruff format nobrainer/` |
# | Pre-commit | `pre-commit run --all-files` |
# | Build docs | `cd docs && make html` |
# | List models | `python -c "from nobrainer.models import list_available_models; list_available_models()"` |

# %% [markdown]
# ## Summary
#
# Contributing to nobrainer involves:
# 1. Understanding the project structure (models, processing, tests)
# 2. Following the factory-function pattern for new models
# 3. Writing unit tests (and GPU tests where needed)
# 4. Using pre-commit hooks for code quality
# 5. Creating PRs targeting the `alpha` branch
#
# Thank you for considering contributing to nobrainer!
