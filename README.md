# imgConvolutionCuda-C

This was an assigment for the course Hardware Accelerated Computing.

An application written by Sem Kirkels, Nathan Bruggeman and Axel Vanherle in C and using CUDA that manipulates a image using convolution, and applies maximum and minimum pooling.

### What is it?

This application reads 10 images, and processes them using convolution, applies maximum and minimum pooling.

The application calculates the 2D Convolution on these images as in following figure:

![image](https://user-images.githubusercontent.com/94362354/197715244-afcae750-128c-4dba-95d8-8c450b977727.png)

Another example:

![image](https://user-images.githubusercontent.com/94362354/197715440-bc4313c3-287a-4676-9046-a6f026218e16.png)


The results get saved into another image.

The application also calculates the maximum and minimum pooling.

Example:

![image](https://user-images.githubusercontent.com/94362354/197715748-c407534a-eb89-494a-a06e-f54e60475493.png)

We wrote the application in C with and without CUDA, so we can compare the speed gained.

# Benchmarks

The threads in C is only faster because the files get writen away in these threads aswell. In CUDA it gets done sequential, the theards was a proof of concept that it can be accelerated even more.

| Program      |  Time       | 
|--------------|-------------|
| Sequential C | 188.987s    |
| Threads C    | 85.763s     |
| CUDA Total   | 168.849s    |
| CUDA Kernels | 0.571s      |

# Sources

We used the image loader: stb_image.h and the image writer: stb_image_write.h from https://github.com/nothings/stb.

Images are provid by the proffesor of this course, https://github.com/cteqeu/.
