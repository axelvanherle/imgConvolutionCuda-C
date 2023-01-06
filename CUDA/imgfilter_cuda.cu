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

    cudaStream_t stream[10];

    for (int i = 0; i < 10; i++)
    {
        cudaStreamCreate(&stream[i]);
    }

    printf("Building filepaths\r\n");

    const char *inputFileName[10] =
        {
            "Images/img_0.png",
            "Images/img_1.png",
            "Images/img_2.png",
            "Images/img_3.png",
            "Images/img_4.png",
            "Images/img_5.png",
            "Images/img_6.png",
            "Images/img_7.png",
            "Images/img_8.png",
            "Images/img_9.png",
        };

    // Build output filename
    const char *fileNameOutConvolution[10] =
        {
            "Output_Images/Convolution/OutputConvolution0.png",
            "Output_Images/Convolution/OutputConvolution1.png",
            "Output_Images/Convolution/OutputConvolution2.png",
            "Output_Images/Convolution/OutputConvolution3.png",
            "Output_Images/Convolution/OutputConvolution4.png",
            "Output_Images/Convolution/OutputConvolution5.png",
            "Output_Images/Convolution/OutputConvolution6.png",
            "Output_Images/Convolution/OutputConvolution7.png",
            "Output_Images/Convolution/OutputConvolution8.png",
            "Output_Images/Convolution/OutputConvolution9.png",
        };

    const char *fileNameOutMinPooling[10] =
        {
            "Output_Images/Pooling/OutputMinPooling0.png",
            "Output_Images/Pooling/OutputMinPooling1.png",
            "Output_Images/Pooling/OutputMinPooling2.png",
            "Output_Images/Pooling/OutputMinPooling3.png",
            "Output_Images/Pooling/OutputMinPooling4.png",
            "Output_Images/Pooling/OutputMinPooling5.png",
            "Output_Images/Pooling/OutputMinPooling6.png",
            "Output_Images/Pooling/OutputMinPooling7.png",
            "Output_Images/Pooling/OutputMinPooling8.png",
            "Output_Images/Pooling/OutputMinPooling9.png",
        };

    const char *fileNameOutMaxPooling[10] =
        {
            "Output_Images/Pooling/OutputMaxPooling0.png",
            "Output_Images/Pooling/OutputMaxPooling1.png",
            "Output_Images/Pooling/OutputMaxPooling2.png",
            "Output_Images/Pooling/OutputMaxPooling3.png",
            "Output_Images/Pooling/OutputMaxPooling4.png",
            "Output_Images/Pooling/OutputMaxPooling5.png",
            "Output_Images/Pooling/OutputMaxPooling6.png",
            "Output_Images/Pooling/OutputMaxPooling7.png",
            "Output_Images/Pooling/OutputMaxPooling8.png",
            "Output_Images/Pooling/OutputMaxPooling9.png",
        };

    int width[10], height[10], componentCount[10], size[10];

    unsigned char *originalImageHost[10];
    unsigned char *imageDataConvolutionHost[10]; // Saves output image
    unsigned char *imageDataMinPoolingHost[10];  // Saves Min pooling image
    unsigned char *imageDataMaxPoolingHost[10];  // Saves Max pooling image
    unsigned char *originalImage[10];            // Saves the original image on host
    unsigned char *imageDataGrayscale[10];       // Saves the grayscale image on device
    unsigned char *imageDataConvolution[10];     // Saves the convolved image
    unsigned char *imageDataMinPooling[10];      // Saves the min pooled image
    unsigned char *imageDataMaxPooling[10];      // Saves the max pooled image

    printf("Done\r\n");

    printf("Loading png files\r\n");

    for (int i = 0; i < NUMBER_OF_IMAGES; i++)
    {
        originalImageHost[i] = stbi_load(inputFileName[i], &width[i], &height[i], &componentCount[i], 4);

        size[i] = height[i] * width[i] * 4;

        // Saves output image
        imageDataConvolutionHost[i] = (unsigned char *)malloc(size[i]);

        // Saves Min pooling image
        imageDataMinPoolingHost[i] = (unsigned char *)malloc(size[i]);

        // Saves Max pooling image
        imageDataMaxPoolingHost[i] = (unsigned char *)malloc(size[i]);

        cudaMalloc(&originalImage[i], size[i]);
        cudaMalloc(&imageDataGrayscale[i], size[i]);
        cudaMalloc(&imageDataConvolution[i], size[i]);
        cudaMalloc(&imageDataMinPooling[i], size[i]);
        cudaMalloc(&imageDataMaxPooling[i], size[i]);

        cudaMemcpy(originalImage[i], originalImageHost[i], size[i], cudaMemcpyHostToDevice);
    }

    printf("Done\r\n");

    // Process grayscale
    printf("Processing images grayscale\r\n");
    for (int i = 0; i < NUMBER_OF_IMAGES; i++)
    {
        ConvertImageToGrayCpu<<<numberOfBlocks, threadsPerBlock, i, stream[i]>>>(originalImage[i], imageDataGrayscale[i], width[i], height[i]);
    }
    cudaDeviceSynchronize();
    printf("Done\r\n");

    // Process convolution
    printf("Processing image convolution\r\n");
    for (int i = 0; i < NUMBER_OF_IMAGES; i++)
    {
        convolveImage<<<numberOfBlocks, threadsPerBlock, i, stream[i]>>>(imageDataGrayscale[i], imageDataConvolution[i], width[i], height[i]);
    }
    cudaDeviceSynchronize();
    printf("Done\r\n");

    // Process min pooling
    printf("Processing images minimum pooling\r\n");
    for (int i = 0; i < NUMBER_OF_IMAGES; i++)
    {
        minPooling<<<numberOfBlocks, threadsPerBlock, i, stream[i]>>>(originalImage[i], imageDataMinPooling[i], width[i], height[i]);
    }
    cudaDeviceSynchronize();
    printf("Done\r\n");

    // Process max pooling
    printf("Processing image maximum pooling\r\n");
    for (int i = 0; i < NUMBER_OF_IMAGES; i++)
    {
        maxPooling<<<numberOfBlocks, threadsPerBlock, i, stream[i]>>>(originalImage[i], imageDataMaxPooling[i], width[i], height[i]);
        cudaDeviceSynchronize();
    }
    printf("Done\r\n");

    // Writing Convolved images

    // Write image back to disk
    printf("Writing convolved png to disk\r\n");
    for (int i = 0; i < NUMBER_OF_IMAGES; i++)
    {
        cudaMemcpy(imageDataConvolutionHost[i], imageDataConvolution[i], size[i], cudaMemcpyDeviceToHost);
        stbi_write_png(fileNameOutConvolution[i], width[i] - 2, height[i] - 2, 4, imageDataConvolutionHost[i], 4 * width[i]);
    }
    printf("Done\r\n");

    // Writing min pooled images
    printf("Writing min pooling png to disk\r\n");
    for (int i = 0; i < NUMBER_OF_IMAGES; i++)
    {
        cudaMemcpy(imageDataMinPoolingHost[i], imageDataMinPooling[i], size[i], cudaMemcpyDeviceToHost);
        stbi_write_png(fileNameOutMinPooling[i], width[i] / 2, height[i] / 2, 4, imageDataMinPoolingHost[i], 4 * (width[i] / 2));
    }
    printf("Done\r\n");

    // Writing max pooled images
    printf("Writing max pooling png to disk\r\n");
    for (int i = 0; i < NUMBER_OF_IMAGES; i++)
    {
        cudaMemcpy(imageDataMaxPoolingHost[i], imageDataMaxPooling[i], size[i], cudaMemcpyDeviceToHost);
        stbi_write_png(fileNameOutMaxPooling[i], width[i] / 2, height[i] / 2, 4, imageDataMaxPoolingHost[i], 4 * (width[i] / 2));
    }
    printf("Done\r\n");

    // Free memory and destroy streams
    for (int i = 0; i < NUMBER_OF_IMAGES; i++)
    {
        stbi_image_free(originalImageHost[i]);

        free(imageDataConvolutionHost[i]);
        free(imageDataMinPoolingHost[i]);
        free(imageDataMaxPoolingHost[i]);

        cudaFree(originalImage[i]);
        cudaFree(imageDataConvolution[i]);
        cudaFree(imageDataMinPooling[i]);
        cudaFree(imageDataMaxPooling[i]);

        cudaStreamDestroy(stream[i]);
    }

    timer_end = clock(); // end the timer
    double time_spent = (double)(timer_end - timer_start) / CLOCKS_PER_SEC;
    printf("\nProgram time: %.3fs\n", time_spent);

    return 0;
}

__global__ void ConvertImageToGrayCpu(unsigned char *originalImage, unsigned char *imageDataGrayscale, int width, int height)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    int gridStride = blockDim.x * gridDim.x;
    int totalPixels = width * height * 4;

    for (int x = idx; x < totalPixels; x += gridStride)
    {
        Pixel *ptrPixel = (Pixel *)&imageDataGrayscale[x];
        Pixel *ptrPixelOriginal = (Pixel *)&originalImage[x];
        unsigned char pixelValue = (unsigned char)(ptrPixelOriginal->r * 0.2126f + ptrPixelOriginal->g * 0.7152f + ptrPixelOriginal->b * 0.0722f);
        ptrPixel->r = pixelValue;
        ptrPixel->g = pixelValue;
        ptrPixel->b = pixelValue;
        ptrPixel->a = 255;
    }
}

__global__ void convolveImage(unsigned char *imageDataGrayscale, unsigned char *imageDataConvolution, int width, int height)
{
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    int gridStride = blockDim.x * gridDim.x;
    int totalPixels = width * height * 4;

    int kernel[3][3] =
        {
            {1, 0, -1},
            {1, 0, -1},
            {1, 0, -1}};

    int pixels[3][3] = {0};
    int finalPixel = 0;

    for (int x = idx; x < totalPixels - 2; x += gridStride)
    {
        for (int i = 0; i <= 2; i++)
        {
            Pixel *ptrPixel = (Pixel *)&imageDataGrayscale[x + i * 4];

            pixels[0][i] = ptrPixel->r * kernel[0][i];
        }

        for (int i = 0; i <= 2; i++)
        {
            Pixel *ptrPixel = (Pixel *)&imageDataGrayscale[x + width * 4 + i * 4];

            pixels[1][i] = ptrPixel->r * kernel[1][i];
        }

        for (int i = 0; i <= 2; i++)
        {
            Pixel *ptrPixel = (Pixel *)&imageDataGrayscale[x + (2 * width * 4) + i * 4];

            pixels[2][i] = ptrPixel->r * kernel[2][i];
        }

        finalPixel = (pixels[0][0] + pixels[0][1] + pixels[0][2] + pixels[1][0] + pixels[1][1] + pixels[1][2] + pixels[2][0] + pixels[2][1] + pixels[2][2]) / 9;

        Pixel *ptrPixel = (Pixel *)&imageDataConvolution[x];
        ptrPixel->r = finalPixel;
        ptrPixel->g = finalPixel;
        ptrPixel->b = finalPixel;
        ptrPixel->a = 255;
    }
}

__global__ void minPooling(unsigned char *originalImage, unsigned char *minPoolingImage, int width, int height)
{
    // Declare shared memory for the 2x2 block of pixels
    __shared__ unsigned char block[2][2];

    // Calculate the 2D index of the thread within the grid
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Skip processing pixels that are outside the bounds of the image
    if (x >= width || y >= height)
    {
        return;
    }

    // Load the 2x2 block of pixels into shared memory
    block[threadIdx.y][threadIdx.x] = originalImage[y * width + x];

    // Wait for all threads to finish loading the block
    __syncthreads();

    // Find the minimum value for each channel in the 2x2 block using a loop and a conditional statement
    unsigned char minR = 255, minG = 255, minB = 255, minA = 255;
    for (int dy = 0; dy < 2; dy++)
    {
        for (int dx = 0; dx < 2; dx++)
        {
            int index = dy * 2 + dx;
            minR = (block[index][0] < minR) ? block[index][0] : minR;
            minG = (block[index][1] < minG) ? block[index][1] : minG;
            minB = (block[index][2] < minB) ? block[index][2] : minB;
            minA = (block[index][3] < minA) ? block[index][3] : minA;
        }
    }

    // Store the minimum values in the result image
    minPoolingImage[y * width + x] = minR;
    minPoolingImage[y * width + x + 1] = minG;
    minPoolingImage[y * width + x + 2] = minB;
    minPoolingImage[y * width + x + 3] = 255;
}




__global__ void maxPooling(unsigned char *originalImage, unsigned char *maxPoolingImage, int width, int height)
{
    int counter = 0;

    // Iterate over the image in 2x2 blocks
    for (int y = 0; y < height; y += 2)
    {
        for (int x = 0; x < width; x += 2)
        {
            // For each channel, find the maximum value in the 2x2 block
            for (int c = 0; c < 4; c++)
            {
                Pixel *ptrPixelMaxPooling = (Pixel *)&maxPoolingImage[counter];
                unsigned char max = 0;
                for (int dy = 0; dy < 2; dy++)
                {
                    for (int dx = 0; dx < 2; dx++)
                    {
                        // Calculate the index of the current pixel in the 1D array
                        int index = (y + dy) * width * 4 + (x + dx) * 4 + c;
                        unsigned char value = originalImage[index];
                        max = (value > max) ? value : max;
                    }
                }
                // Store the maximum value in the result array
                ptrPixelMaxPooling->r = max;
                ptrPixelMaxPooling->g = max;
                ptrPixelMaxPooling->b = max;
                ptrPixelMaxPooling->a = max;
                counter++;
            }
        }
    }   
}
