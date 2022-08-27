# Installation

## Container

We recommend using the official _Nobrainer_ Docker container, which includes all
the dependencies necessary to use the framework. Please see the available images
on [DockerHub](https://hub.docker.com/r/neuronets/nobrainer)

#### GPU support

The _Nobrainer_ containers with GPU support use the Tensorflow jupyter GPU containers. Please
check the containers for the version of CUDA installed. Nvidia drivers are not included in
the container.

```
$ docker pull neuronets/nobrainer:latest-gpu
$ singularity pull docker://neuronets/nobrainer:latest-gpu
```

#### CPU only

This container can be used on all systems that have Docker or Singularity and
does not require special hardware. This container, however, should not be used
for model training (it will be very slow).

```
$ docker pull neuronets/nobrainer:latest-cpu
$ singularity pull docker://neuronets/nobrainer:latest-cpu
```

### pip

_Nobrainer_ can also be installed with pip.

```
$ pip install nobrainer
```

## Package layout

- `nobrainer.io`: input/output methods
- `nobrainer.layers`: custom layers, which conform to the Keras API
- `nobrainer.losses`: loss functions for volumetric segmentation
- `nobrainer.metrics`: metrics for volumetric segmentation
- `nobrainer.models`: pre-defined Keras models
- `nobrainer.training`: training utilities (supports training on single and multiple GPUs)
- `nobrainer.transform`: random rigid transformations for data augmentation
- `nobrainer.volume`: `tf.data.Dataset` creation and data augmentation utilities
-
## Citation
If you use this package, please [cite](https://github.com/neuronets/nobrainer/blob/master/CITATION) it.

## Questions or issues

If you have questions about _Nobrainer_ or encounter any issues using the framework,
please [submit a GitHub issue](https://github.com/neuronets/helpdesk/issues/new/choose).
