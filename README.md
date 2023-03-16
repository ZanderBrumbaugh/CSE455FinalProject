# Implicit Depth Learning in Convolutional Neural Networks On 3D Datasets
### Group Members: Yegor Kuznetsov, Zander Brumbaugh, Colton Jobes

## Project Summary
Our project aims to test our hypothesis that depth information can be reliably predicted from the latent representation from an autoencoder, thus proving depth information is learned by the autoencoder when training. We additionally compare the accuracy of our prediction with a convolutional neural network (CNN) trained explicitly to predict depth information from an image to determine the extent of which depth information is used in the autoencoder. Furthermore, we propose a two-parameter, generalized framework to determine whether a feature is implicitly learned given a feature and computer vision task.

## Datasets
We used the Diode dataset (Vasiljevic et al., 2019) and NYU Depth V2 dataset (Silberman et al., 2012). The NYU data was provided in a .mat format which we then preprocessed into a useable format with Python and later a Dataset object for use with PyTorch. We used this set for development as it was considerably smaller than the Diode dataset but consequently had lower in image and depth map resolution. We then trained our final model on the images and depth maps from the Diode dataset.

## Methodology
We began by implementing our autoencoder based on the publicly available VQ-VAE (Vector Quantized - Variational Autoencoder) used by DALL-E from OpenAI. We removed the vector quantization, downsized the images to 256 x 256 pixels, and trained the model until convergence. For all training across the project, we used a Nvidia 3080Ti GPU on one of our own devices. The autoencoder took six hours to train and produced good results. With the autoencoder trained, we were ready to create the network for the linear probe. The probe is a 1 x k matrix, where we used k = 10, that is trained to predict depth information in a given image given its latent representation from the autoencoder. For the training of the linear probe, we used L2 loss and trained until convergence. The probe took only ten minutes to train and achieved a mean accuracy of XX.XX%. We define accuracy as the percent difference between the ground and predicted depth values for each pixel. That is:

accuracy = ![](https://latex.codecogs.com/svg.latex?\Large&space;\frac{1}{n} \sum^n\frac{abs(y_{pred} - y_{ground})}{y_{ground}}")

Having shown that depth information is learned by the autoencoder, we wanted to compare the linear probe's accuracy with the accuracy of a convolutional neural network training explicitly to predict depth information to determine the importance of depth information in the representation from the autoencoder. That is, if a network trained to predict depth information achieves only marginally better accuracy, that further supports our hypothesis and implies depth information is a critical feature in the task of the autoencoder.
We created a CNN with four groups, with each group having two residual blocks, and each block having four convolutional layers. We additionally include a 1x1 convolution layer for prediction. This is the same architecture as the encoder portion of the autoencoder with the additional convolutional layer for prediction. We use this similar architecture, the same dataset used to train the probe, and the same downsampling techniques to create fair conditions for the comparison of the accuracies between both models. We used L2 loss and trained the CNN until convergence. The CNN achieved an accuracy of XX.XX% over the test we created of XX random downsampled images not included in the training data.

## Results
We found our hypothesis was XX. The accuracy of XX.XX% that the linear probe achieved shows that depth information is learnable from the latent representation from the autoencoder, supporting our hypothesis that depth information is implicitly learned by the autoencoder during training.
The CNN trained to predict depth information achieved an accuracy of XX.XX%, *. This shows that *.

## Discussion
Because our hypothesis was XX, we propose a generalized framework for proving that a specified feature is learned in the training of a given computer vision task.  We call this framework xXCoolNameHereXx and it requires only a dataset with the ground truth data for the feature you wish to know is implicitly learned.
