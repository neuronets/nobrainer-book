# Implementations

## Models
| Model | Type | Application |
|:-----------|:-------------:|:-------------:|
|[**Highresnet**](https://github.com/neuronets/nobrainer/blob/master/nobrainer/models/highresnet.py) [(source)](https://arxiv.org/abs/1707.01992)| supervised  | segmentation/classification |
|[**Unet**](https://github.com/neuronets/nobrainer/blob/master/nobrainer/models/unet.py) [(source)](https://arxiv.org/abs/1606.06650)| supervised | segmentation/classification |
|[**Vnet**](https://github.com/neuronets/nobrainer/blob/master/nobrainer/models/vnet.py) [(source)](https://arxiv.org/pdf/1606.04797.pdf)| supervised  | segmentation/classification |
|[**Meshnet**](https://github.com/neuronets/nobrainer/blob/master/nobrainer/models/meshnet.py) [(source)](https://arxiv.org/abs/1612.00940)| supervised  | segmentation/clssification |
|[**Bayesian Meshnet**](https://github.com/neuronets/nobrainer/blob/master/nobrainer/models/bayesian-meshnet.py) [(source)](https://www.frontiersin.org/articles/10.3389/fninf.2019.00067/full)| bayesian supervised | segmentation/classification |
|[**Bayesian Vnet**](https://github.com/neuronets/nobrainer/blob/master/nobrainer/models/bayesian_vnet.py) | bayesian supervised | segmentation/classification |
|[**Semi_Bayesian Vnet**](https://github.com/neuronets/nobrainer/blob/master/nobrainer/models/bayesian_vnet.py) | semi-bayesian supervised | segmentation/classification |
|[**DCGAN**](https://github.com/neuronets/nobrainer/blob/master/nobrainer/models/dcgan.py) | self supervised | generative model |
|[**Progressive GAN**](https://github.com/neuronets/nobrainer/blob/master/nobrainer/models/progressivegan.py) | self supervised | generative model |
|[**3D Autoencoder**](https://github.com/neuronets/nobrainer/blob/master/nobrainer/models/autoencoder.py) | self supervised | knowledge representation/dimensionality reduction |
|[**3D Progressive Autoencoder**](https://github.com/neuronets/nobrainer/blob/master/nobrainer/models/progressiveae.py) | self supervised | knowledge representation/dimensionality reduction |
|[**3D SimSiam**](https://github.com/neuronets/nobrainer/blob/master/nobrainer/models/brainsiam.py) [(source)](https://arxiv.org/abs/2011.10566)| self supervised | Siamese Representation Learning |

### Dropout and regularization layers
[Bernouli dropout layer](https://github.com/neuronets/nobrainer/blob/80d8a47a7f2bf4fe335bdf194c0be19044223629/nobrainer/layers/dropout.py#L15), [Concrete dropout layer](https://github.com/neuronets/nobrainer/blob/80d8a47a7f2bf4fe335bdf194c0be19044223629/nobrainer/layers/dropout.py#L71), [Gaussian dropout](https://github.com/neuronets/nobrainer/blob/80d8a47a7f2bf4fe335bdf194c0be19044223629/nobrainer/layers/dropout.py#L204), [Group normalization layer](https://github.com/neuronets/nobrainer/blob/master/nobrainer/layers/groupnorm.py), [Custom padding layer](https://github.com/neuronets/nobrainer/blob/master/nobrainer/layers/padding.py)

### Losses
[Dice](https://github.com/neuronets/nobrainer/blob/e3e71131373602484caf696fd78dd16e572adf9b/nobrainer/losses.py#L14), [Jaccard](https://github.com/neuronets/nobrainer/blob/e3e71131373602484caf696fd78dd16e572adf9b/nobrainer/losses.py#L90), [Tversky](https://github.com/neuronets/nobrainer/blob/e3e71131373602484caf696fd78dd16e572adf9b/nobrainer/losses.py#L129), [ELBO](https://github.com/neuronets/nobrainer/blob/e3e71131373602484caf696fd78dd16e572adf9b/nobrainer/losses.py#L167), [Wasserstien](https://github.com/neuronets/nobrainer/blob/e3e71131373602484caf696fd78dd16e572adf9b/nobrainer/losses.py#L196), [Gradient Penalty](https://github.com/neuronets/nobrainer/blob/e3e71131373602484caf696fd78dd16e572adf9b/nobrainer/losses.py#L235)

### Metrics
[Dice](https://github.com/neuronets/nobrainer/blob/e3e71131373602484caf696fd78dd16e572adf9b/nobrainer/metrics.py#L8), [Generalized Dice](https://github.com/neuronets/nobrainer/blob/e3e71131373602484caf696fd78dd16e572adf9b/nobrainer/metrics.py#L39), [Jaccard](https://github.com/neuronets/nobrainer/blob/e3e71131373602484caf696fd78dd16e572adf9b/nobrainer/metrics.py#L76), [Hamming](https://github.com/neuronets/nobrainer/blob/e3e71131373602484caf696fd78dd16e572adf9b/nobrainer/metrics.py#L66), [Tversky](https://github.com/neuronets/nobrainer/blob/e3e71131373602484caf696fd78dd16e572adf9b/nobrainer/metrics.py#L107)

### Augmentation methods
#### Spatial Transforms
[Center crop](https://github.com/neuronets/nobrainer/blob/e3e71131373602484caf696fd78dd16e572adf9b/nobrainer/spatial_transforms.py#L4), [Spacial Constant Padding](https://github.com/neuronets/nobrainer/blob/e3e71131373602484caf696fd78dd16e572adf9b/nobrainer/spatial_transforms.py#L54), [Random Crop](https://github.com/neuronets/nobrainer/blob/e3e71131373602484caf696fd78dd16e572adf9b/nobrainer/spatial_transforms.py#L106), [Resize](https://github.com/neuronets/nobrainer/blob/e3e71131373602484caf696fd78dd16e572adf9b/nobrainer/spatial_transforms.py#L149), [Random flip (left and right)](https://github.com/neuronets/nobrainer/blob/e3e71131373602484caf696fd78dd16e572adf9b/nobrainer/spatial_transforms.py#L197)

#### Intensity Transforms
[Add gaussian noise](https://github.com/neuronets/nobrainer/blob/e3e71131373602484caf696fd78dd16e572adf9b/nobrainer/intensity_transforms.py#L6), [Min-Max intensity scaling](https://github.com/neuronets/nobrainer/blob/e3e71131373602484caf696fd78dd16e572adf9b/nobrainer/intensity_transforms.py#L50), [Custom intensity scaling](https://github.com/neuronets/nobrainer/blob/e3e71131373602484caf696fd78dd16e572adf9b/nobrainer/intensity_transforms.py#L96), [Intensity masking](https://github.com/neuronets/nobrainer/blob/e3e71131373602484caf696fd78dd16e572adf9b/nobrainer/intensity_transforms.py#L159), [Contrast adjustment](https://github.com/neuronets/nobrainer/blob/e3e71131373602484caf696fd78dd16e572adf9b/nobrainer/intensity_transforms.py#L213)

#### Affine Transform
Affine transformation including [rotation, translation, reflection](https://github.com/neuronets/nobrainer/blob/master/nobrainer/transform.py).

