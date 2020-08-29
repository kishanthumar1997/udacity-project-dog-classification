# udacity-project-dog-classification
This is a repository for the Dog Breed Classifier Project in Udacity Nanodegree. It is implemented by using PyTorch library, by developing the CNN to predict the breed of the Dog.

## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project in the AI  Nanodegree! In this project, you will learn how to build a pipeline that  can be used within a web or mobile app to process real-world,  user-supplied images.  Given an image of a dog, your algorithm will  identify an estimate of the canine’s breed.  If supplied an image of a  human, the code will identify the resembling dog breed.

[![Sample Output](https://github.com/udacity/deep-learning-v2-pytorch/raw/master/project-dog-classification/images/sample_dog_output.png)](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/project-dog-classification/images/sample_dog_output.png)

Along with exploring state-of-the-art CNN models for classification  and localization, you will make important design decisions about the  user experience for your app.  Our goal is that by completing this lab,  you understand the challenges involved in piecing together a series of  models designed to perform various tasks in a data processing pipeline.   Each model has its strengths and weaknesses, and engineering a  real-world application often involves solving many problems without a  perfect answer.  Your imperfect solution will nonetheless create a fun  user experience!

## Datasets Used

* Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
* Download the [human_dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)

## Results on Custom CNN Architecture.

* The input to the network is of the shape `3x224x224`. We have 133 classes in our dataset and so the number of output is the same as the number of classes. In the network architecture, I employed 3 convolution layer (`3x3` kernel size with the stride of 1 and padding of 1) that converts the depth of the image to `16`, `32`, `64` layers respectively, followed by the relu activation function. The flow of the image tensor is as such, `32x32x3` (first convolutional layer) -> `16x16x16` (second convolutional layer) -> `8x8x32` (third convolutional layer). The pooling layer (2,2) is used to reduce the size of the image to half.
* Test Accuracy of `11%` by training the custom CNN model for 30 epochs, which can be increased by training on larger dataset, increasing on the image augmentation and by using bayesian optimization for hyper-parameter tuning and use k-fold Cross-Validation.

## Results on Transfer learning.

* I chose VGG16 for transfer learning. I studied architecture and to save time I performed freeze training for all the feature layers. The final layer was modified to generate the output for 133 classes to satisfy our use case and the classification output. Afterward, I trained the classifier to generated good results for the use case.
* Test Accuracy of 87% was achieved.


