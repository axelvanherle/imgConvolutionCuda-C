# imgConvolutionCuda-C

A program written by Axel Vanherle, Sem Kirkels and Nathan Bruggeman in C using CUDA that manipulates a image using convolution.

**What is it?**
This application reads 10 images, and processes them using convolution.

The application calculates the 2D Convolution on these images as in following figure:


The results get saved into another image.

The application also calculates the maximum and average pooling.
Example:


We wrote the application in C without and with CUDA, so we can compare the speed gained.