# Implicit Depth Learning in Convolutional Neural Networks On 3D Datasets
### Group Members: Yegor Kuznetsov, Zander Brumbaugh, Colton Jobes

## Project Summary
Our project aims to test our hypothesis that depth information can be reliably predicted from the latent representation from a CNN trained on a three-dimensional dataset without an explicit depth task. We additionally compare the accuracy between several models in order to provide further evidence in support of our hypothesis. Furthermore, we propose a two-parameter, generalized framework to determine whether a feature is implicitly learned given a feature and computer vision task.

## Datasets
We used the Diode dataset (Vasiljevic et al., 2019) and NYU Depth V2 dataset (Silberman et al., 2012). The NYU data was provided in a .mat format which we then preprocessed into a useable format with Python and later a Dataset object for use with PyTorch. We used this set for development as it was considerably smaller than the Diode dataset but consequently had lower image and depth map resolution. We then trained our final model on the images and depth maps from the Diode dataset.

## Methodology
We began by implementing our autoencoder based on the publicly available VQ-VAE (Vector Quantized - Variational Autoencoder) used by DALL-E from OpenAI. We removed the vector quantization, downsized the images to 256 x 256 pixels, and trained the model until convergence. For all training across the project, we used a Nvidia 3080Ti GPU on one of our own devices. The autoencoder took six hours to train and produced good results. With the autoencoder trained, we were ready to create the network for the linear probe. The probe is a 1 x k matrix, where we used k = 10, that is trained to predict depth information in a given image given its latent representation from the autoencoder. For the training of the linear probe, we used L2 loss and trained until convergence. The probe took only ten minutes to train and we qualitatively found the predicted depth maps to be largely represntative of the ground truth maps. We created our test set from 25,000 random downsampled images not included in the training data.

We then used a simpler model as a baseline where in order to prove that the model does actually extract depth information, we compare it to using no model. That is, simply downsampling the image as the encoding step such that the data that's mapped by the linear probe is averaged raw pixel data. This means that if depth can be predicted by surface level features then this implies our more complex approach would not work significantly better, however this is not our expectation.

Following this, we used a pretrained CNN, ConvNeXT. We used the feature map of the CNN as the "encoder", froze it, and added a linear probe like before. This is even more true to our broad hypothesis.

## Results
We found our hypothesis was to some extent true. The accuracy that the linear probe achieved shows that depth information is learnable from the latent representation from the autoencoder, supporting our hypothesis that depth information is implicitly learned by the autoencoder during training.

Both the autoencoder approach and using the pretrained frozen ConvNeXT+probe significantly outperformed the null model baseline, achieving around a third of the loss. This matches our expectation, so we qualitatively assess that the predicted depth maps are similar. Exactly how this information is extracted appears to be mostly from color for the autoencoder, with lighter colors correlating with further distances. We believe this could be related to images with the sky or other similar examples even when augmentation was applied. The ConvNeXT approach appears to rely more on shapes and structure, likely due to better data and training.

## Discussion
We believe that we would have been able to produce even stronger evidence that depth information is implicitly learned by CNNs on 3D datasets if we explored the possible model/task space further.
Because our hypothesis was true, we propose a generalized framework for proving that a specified feature is learned in the training of a given computer vision task.  This framework requires only a dataset with the ground truth data for the feature you wish to know is implicitly learned and the encoded input from the model for the task.
