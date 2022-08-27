# Transfer learning

The pre-trained models can be used for transfer learning. To avoid forgetting
important information in the pre-trained model, you can apply regularization to
the kernel weights and also use a low learning rate. For more information, please
see the _Nobrainer_ guide notebook on transfer learning.

As an example of transfer learning, [@kaczmarj](https://github.com/kaczmarj)
re-trained a brain extraction model to label meningiomas in 3D T1-weighted,
contrast-enhanced MR scans. The original model is publicly available and was
trained on 10,000 T1-weighted MR brain scans from healthy participants. These
were all research scans (i.e., non-clinical) and did not include any contrast
agents. The meningioma dataset, on the other hand, was composed of relatively
few scans, all of which were clinical and used gadolinium as a contrast agent.
You can observe the differences in contrast below.

![Brain extraction model prediction](https://github.com/neuronets/trained-models/blob/master/images/brain-extraction/unet-best-prediction.png?raw=true)
![Meningioma extraction model prediction](https://user-images.githubusercontent.com/17690870/55470578-e6cb7800-55d5-11e9-991f-fe13c03ab0bd.png)

Despite the differences between the two datasets, transfer learning led to a much
better model than training from randomly-initialized weights. As evidence, please
see below violin plots of Dice coefficients on a validation set. In the left plot
are Dice coefficients of predictions obtained with the model trained from
randomly-initialized weights, and on the right are Dice coefficients of predictions
obtained with the transfer-learned model. In general, Dice coefficients are higher
on the right, and the variance of Dice scores is lower. Overall, the model on the
right is more accurate and more robust than the one on the left.

<div style="text-align: center">
<img src="https://user-images.githubusercontent.com/17690870/56313232-1e7f0780-6120-11e9-8f1a-62b8c3d48e15.png" alt="" width="49%" />
<img src="https://user-images.githubusercontent.com/17690870/56313239-23dc5200-6120-11e9-88eb-0e9ebca6ba83.png" alt="" width="49%" />
</div>

