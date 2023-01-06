#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define NUMBER_OF_IMAGES 2

typedef struct Pixel
{
    unsigned char r, g, b, a;
} Pixel;

__global__ void ConvertImageToGrayCpu(unsigned char *originalImage, unsigned char *imageDataGrayscale, int width, int height);
__global__ void convolveImage(unsigned char *imageDataGrayscale, unsigned char *imageDataConvolution, int width, int height);
__global__ void minPooling(unsigned char *originalImage, unsigned char *minPoolingImage, int width, int height);
__global__ void maxPooling(unsigned char *originalImage, unsigned char *maxPoolingImage, int width, int height);

int main(int argc, char **argv)
{
    clock_t timer_start, timer_end;
    timer_start = clock();

    size_t threadsPerBlock = 128;
    size_t numberOfBlocks = 32;

    printf("Building filepaths\r\n");

    const char *inputFileName = "Images/img_7.png";

    const char *fileNameOutMinPooling = "Output_Images/Pooling/OutputMinPoolingTEST0.png";

    int width, height, componentCount, size;

    unsigned char *originalImageHost = stbi_load(inputFileName, &width, &height, &componentCount, 4);
    unsigned char *imageDataMinPoolingHost; // Saves Min pooling image
    unsigned char *originalImage;           // Saves the original image on host
    unsigned char *imageDataMinPooling;     // Saves the min pooled image

    printf("Done\r\n");

    size = height * width * 4;

    // Saves Min pooling image
    imageDataMinPoolingHost = (unsigned char *)malloc(size);

    cudaMalloc(&originalImage, size);
    cudaMalloc(&imageDataMinPooling, size);

    cudaMemcpy(originalImage, originalImageHost, size, cudaMemcpyHostToDevice);

    printf("Done\r\n");

    // Process min pooling
    printf("Processing images minimum pooling\r\n");
    // Set up the grid and block dimensions
    dim3 gridDim(width / 2, height / 2);
    dim3 blockDim(2, 2);

    // Launch the kernel
    minPooling<<<gridDim, blockDim>>>(originalImage, imageDataMinPooling, width, height);

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("Error launching kernel: %s\n", cudaGetErrorString(error));
        return;
    }

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    //minPooling<<<threadsPerBlock, numberOfBlocks>>>(originalImage, imageDataMinPooling, width, height);
    //cudaDeviceSynchronize();
    printf("Done\r\n");

    // Writing min pooled images
    printf("Writing min pooling png to disk\r\n");

    cudaMemcpy(imageDataMinPoolingHost, imageDataMinPooling, size, cudaMemcpyDeviceToHost);
    stbi_write_png(fileNameOutMinPooling, width / 2, height / 2, 4, imageDataMinPoolingHost, 4 * (width / 2));
    printf("Done\r\n");

    stbi_image_free(originalImageHost);

    free(imageDataMinPoolingHost);

    cudaFree(originalImage);
    cudaFree(imageDataMinPooling);

    timer_end = clock(); // end the timer
    double time_spent = (double)(timer_end - timer_start) / CLOCKS_PER_SEC;
    printf("\nProgram time: %.3fs\n", time_spent);

    return 0;
}
__global__ void minPooling(unsigned char *originalImage, unsigned char *minPoolingImage, int width, int height)
{
    // Calculate the 2D block index
    int blockY = blockIdx.y;
    int blockX = blockIdx.x;

    // Calculate the starting position of the 2x2 block
    int y = blockY * 2;
    int x = blockX * 2;

    // Calculate the index of the current pixel in the 1D arrays
    int indexOut = blockY * width / 2 * 4 + blockX * 4;

    // For each channel, find the minimum value in the 2x2 block
    for (int c = 0; c < 4; c++)
    {
        unsigned char min = 255;
        for (int dy = 0; dy < 2; dy++)
        {
            for (int dx = 0; dx < 2; dx++)
            {
                // Calculate the index of the current pixel in the 1D array
                int index = (y + dy) * width * 4 + (x + dx) * 4 + c;
                unsigned char value = originalImage[index];
                min = (value < min) ? value : min;
            }
        }
        // Store the minimum value in the result array
        minPoolingImage[indexOut + c] = min;
    }
}

__global__ void maxPooling(unsigned char *originalImage, unsigned char *minPoolingImage, int width, int height)
{
    // Calculate the 2D block index
    int blockY = blockIdx.y;
    int blockX = blockIdx.x;

    // Calculate the starting position of the 2x2 block
    int y = blockY * 2;
    int x = blockX * 2;

    // Calculate the index of the current pixel in the 1D arrays
    int indexOut = blockY * width / 2 * 4 + blockX * 4;

    // For each channel, find the minimum value in the 2x2 block
    for (int c = 0; c < 4; c++)
    {
        unsigned char min = 0;
        for (int dy = 0; dy < 2; dy++)
        {
            for (int dx = 0; dx < 2; dx++)
            {
                // Calculate the index of the current pixel in the 1D array
                int index = (y + dy) * width * 4 + (x + dx) * 4 + c;
                unsigned char value = originalImage[index];
                min = (value > min) ? value : min;
            }
        }
        // Store the minimum value in the result array
        minPoolingImage[indexOut + c] = min;
    }
}
