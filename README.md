# imgConvolutionCuda-C

An application written by Sem Kirkels, Nathan Bruggeman and Axel Vanherle. in C using CUDA that manipulates a image using convolution.

### What is it?

This application reads 10 images, and processes them using convolution.

The application calculates the 2D Convolution on these images as in following figure:

![image](https://user-images.githubusercontent.com/94362354/197715244-afcae750-128c-4dba-95d8-8c450b977727.png)

Another example:

![image](https://user-images.githubusercontent.com/94362354/197715440-bc4313c3-287a-4676-9046-a6f026218e16.png)


The results get saved into another image.

The application also calculates the maximum and average pooling.

Example:

![image](https://user-images.githubusercontent.com/94362354/197715748-c407534a-eb89-494a-a06e-f54e60475493.png)

We wrote the application in C with and without CUDA, so we can compare the speed gained.
