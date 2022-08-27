# Introduction

_Nobrainer_ is a deep learning framework for 3D image processing. It implements
several 3D convolutional models from recent literature, methods for loading and
augmenting volumetric data that can be used with any TensorFlow or Keras model,
losses and metrics for 3D data, and simple utilities for model training, evaluation,
prediction, and transfer learning.

_Nobrainer_ also provides pre-trained models for brain extraction, brain segmentation,
brain generation and other tasks. Please see the [_Trained_ models](https://github.com/neuronets/trained-models)
repository for more information.

The _Nobrainer_ project is supported by NIH RF1MH121885 and is distributed under
the Apache 2.0 license. It was started under the support of NIH R01 EB020470.

## Package layout

- `nobrainer.io`: input/output methods
- `nobrainer.layers`: custom layers, which conform to the Keras API
- `nobrainer.losses`: loss functions for volumetric segmentation
- `nobrainer.metrics`: metrics for volumetric segmentation
- `nobrainer.models`: pre-defined Keras models
- `nobrainer.training`: training utilities (supports training on single and multiple GPUs)
- `nobrainer.transform`: random rigid transformations for data augmentation
- `nobrainer.volume`: `tf.data.Dataset` creation and data augmentation utilities

## Citation
If you use this package, please [cite](https://github.com/neuronets/nobrainer/blob/master/CITATION) it.

## Questions or issues

If you have questions about _Nobrainer_ or encounter any issues using the framework,
please [submit a GitHub issue](https://github.com/neuronets/helpdesk/issues/new/choose).
