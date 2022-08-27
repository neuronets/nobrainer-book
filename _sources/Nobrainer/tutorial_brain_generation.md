# Generating a synthetic T1-weighted brain scan

![Model's generation of brain (sagittal)](https://github.com/neuronets/trained-models/blob/master/images/brain-generation/progressivegan_generation_sagittal.png?raw=true)
![Model's generation of brain (axial)](https://github.com/neuronets/trained-models/blob/master/images/brain-generation/progressivegan_generation_axial.png?raw=true)
![Model's generation of brain (coronal)](https://github.com/neuronets/trained-models/blob/master/images/brain-generation/progressivegan_generation_coronal.png?raw=true)
<sub>__Figure__: Progressive generation of T1-weighted brain MR scan starting
from a resolution of 32 to 256 (Left to Right: 32<sup>3</sup>, 64<sup>3</sup>,
128<sup>3</sup>, 256<sup>3</sup>). The brain scans are generated using the same
latents in all resolutions. It took about 6 milliseconds for the model to generate
the 256<sup>3</sup> brainscan using an NVIDIA TESLA V-100.</sub>

In the following examples, we will use a Progressive Generative Adversarial Network
trained for brain image generation and documented in
[_Trained_ models](https://github.com/neuronets/trained-models#brain-extraction).

In the base case, we generate a T1w scan through the model for a given resolution.
We need to pass the directory containing the models `(tf.SavedModel)` created
while training the networks.

```bash
docker run --rm -v $PWD:/data neuronets/nobrainer \
  generate \
    --model=/models/neuronets/braingen/0.1.0 \
    --output-shape=128 128 128 \
    /data/generated.nii.gz
```

We can also generate multiple resolutions of the brain image using the same
latents to visualize the progression

```bash
# Get sample T1w scan.
docker run --rm -v $PWD:/data neuronets/nobrainer \
  generate \
    --model=/models/neuronets/braingen/0.1.0 \
    --multi-resolution \
    /data/generated.nii.gz
```

In the above example, the multi resolution images will be saved as
`generated_res_{resolution}.nii.gz`
