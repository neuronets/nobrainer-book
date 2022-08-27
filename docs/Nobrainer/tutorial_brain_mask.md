# Predicting a brain mask for a T1-weighted brain scan

![Model's prediction of brain mask](https://github.com/neuronets/trained-models/blob/master/images/brain-extraction/unet-best-prediction.png?raw=true)
![Model's prediction of brain mask](https://github.com/neuronets/trained-models/blob/master/images/brain-extraction/unet-worst-prediction.png?raw=true)
<sub>__Figure__: In the first column are T1-weighted brain scans, in the middle
are a trained model's predictions, and on the right are binarized FreeSurfer
segmentations. Despite being trained on binarized FreeSurfer segmentations,
the model outperforms FreeSurfer in the bottom scan, which exhibits motion
distortion. It took about three seconds for the model to predict each brainmask
using an NVIDIA GTX 1080Ti. It takes about 70 seconds on a recent CPU.</sub>

In the following examples, we will use a 3D U-Net trained for brain extraction and
documented in [_Trained_ models](https://github.com/neuronets/trained-models#brain-extraction).

In the base case, we run the T1w scan through the model for prediction.

```bash
# Get sample T1w scan.
wget -nc https://dl.dropbox.com/s/g1vn5p3grifro4d/T1w.nii.gz
docker run --rm -v $PWD:/data neuronets/nobrainer \
  predict \
    --model=/models/neuronets/brainy/0.1.0/brain-extraction-unet-128iso-model.h5 \
    --verbose \
    /data/T1w.nii.gz \
    /data/brainmask.nii.gz
```

For binary segmentation where we expect one predicted region, as is the case with
brain extraction, we can reduce false positives by removing all predictions not
connected to the largest contiguous label.

```bash
# Get sample T1w scan.
wget -nc https://dl.dropbox.com/s/g1vn5p3grifro4d/T1w.nii.gz
docker run --rm -v $PWD:/data neuronets/nobrainer \
  predict \
    --model=/models/neuronets/brainy/0.1.0/brain-extraction-unet-128iso-model.h5 \
    --largest-label \
    --verbose \
    /data/T1w.nii.gz \
    /data/brainmask-largestlabel.nii.gz
```

Because the network was trained on randomly rotated data, it should be agnostic
to orientation. Therefore, we can rotate the volume, predict on it, undo the
rotation in the prediction, and average the prediction with that from the original
volume. This can lead to a better overall prediction but will at least double the
processing time. To enable this, use the flag `--rotate-and-predict` in
`nobrainer predict`.

```bash
# Get sample T1w scan.
wget -nc https://dl.dropbox.com/s/g1vn5p3grifro4d/T1w.nii.gz
docker run --rm -v $PWD:/data neuronets/nobrainer \
  predict \
    --model=/models/neuronets/brainy/0.1.0/brain-extraction-unet-128iso-model.h5 \
    --rotate-and-predict \
    --verbose \
    /data/T1w.nii.gz \
    /data/brainmask-withrotation.nii.gz
```

Combining the above, we can usually achieve the best brain extraction by using
`--rotate-and-predict` in conjunction with `--largest-label`.

```bash
# Get sample T1w scan.
wget -nc https://dl.dropbox.com/s/g1vn5p3grifro4d/T1w.nii.gz
docker run --rm -v $PWD:/data neuronets/nobrainer \
  predict \
    --model=/models/neuronets/brainy/0.1.0/brain-extraction-unet-128iso-model.h5 \
    --largest-label \
    --rotate-and-predict \
    --verbose \
    /data/T1w.nii.gz \
    /data/brainmask-maybebest.nii.gz
```
