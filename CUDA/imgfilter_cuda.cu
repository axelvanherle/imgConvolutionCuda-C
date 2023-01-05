#include <stdio.h>
#include <stdlib.h> 
#include "cuda.h"
#include "cuda_runtime.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <time.h>

#define INPUT_IMAGE "Images/img_9.png"

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

    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    size_t threadsPerBlock = 256;
    size_t numberOfBlocks = 32 * numberOfSMs;

    cudaStream_t stream0, stream1, stream2, stream3, stream4, stream5, stream6, stream7, stream8, stream9;

    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    cudaStreamCreate(&stream5);
    cudaStreamCreate(&stream6);
    cudaStreamCreate(&stream7);
    cudaStreamCreate(&stream8);
    cudaStreamCreate(&stream9);

    // Open image
    printf("Loading png file\r\n");

    int width, height, componentCount;

    unsigned char *originalImageCPU = stbi_load(INPUT_IMAGE, &width, &height, &componentCount, 4); // Saves original image
    unsigned char *originalImage;
    unsigned char *imageDataGrayscale;            // Saves grayscale image
    unsigned char *imageDataConvolution;          // Saves output image
    unsigned char *imageDataMinPooling;           // Saves Min pooling image
    unsigned char *imageDataMaxPooling;           // Saves Max pooling image

    int size = width * height * 4;

    cudaMallocManaged((unsigned char **)&originalImage, size);
    cudaMallocManaged((unsigned char **)&imageDataGrayscale, size);
    cudaMallocManaged((unsigned char **)&imageDataConvolution, size);
    cudaMallocManaged((unsigned char **)&imageDataMinPooling, size);
    cudaMallocManaged((unsigned char **)&imageDataMaxPooling, size);

    cudaMemPrefetchAsync(originalImage, size, deviceId);
    cudaMemPrefetchAsync(imageDataGrayscale, size, deviceId);
    cudaMemPrefetchAsync(imageDataConvolution, size, deviceId);
    cudaMemPrefetchAsync(imageDataMinPooling, size, deviceId);
    cudaMemPrefetchAsync(imageDataMaxPooling, size, deviceId);

    cudaMemcpy(originalImage, originalImageCPU, size, cudaMemcpyHostToDevice);

	// Build output filename
    const char *fileNameOutConvolution = "Output_Images/Convolution/OutputConvolution.png";
    const char *fileNameOutMinPooling = "Output_Images/Pooling/OutputMinPooling.png";
    const char *fileNameOutMaxPooling = "Output_Images/Pooling/OutputMaxPooling.png";

    if (!originalImage)
    {
        printf("Failed to open Image\r\n");
        stbi_image_free(originalImageCPU);
        cudaFree(originalImage);
		cudaFree(imageDataGrayscale);
		cudaFree(imageDataConvolution);
		cudaFree(imageDataMinPooling);
		cudaFree(imageDataMaxPooling);

        return -1;
    }

    printf("Done\r\n");

    // Validate image sizes
    if (width % 32 || height % 32)
    {
        // NOTE: Leaked memory of "imageDataGrayscale"
        printf("Width and/or Height is not dividable by 32!\r\n");
        stbi_image_free(originalImageCPU);
		cudaFree(originalImage);
		cudaFree(imageDataGrayscale);
		cudaFree(imageDataConvolution);
		cudaFree(imageDataMinPooling);
		cudaFree(imageDataMaxPooling);

        return -1;
    }

    // Process image on cpu
    printf("Processing image grayscale\r\n");
    ConvertImageToGrayCpu<<<numberOfBlocks, threadsPerBlock>>>(originalImage, imageDataGrayscale, width, height);
    cudaDeviceSynchronize();
    printf("Done\r\n");

    // Process image on cpu
    printf("Processing image convolution\r\n");
    convolveImage<<<numberOfBlocks, threadsPerBlock>>>(imageDataGrayscale, imageDataConvolution, width, height);
    cudaDeviceSynchronize();
    printf("Done\r\n");

    // Write image back to disk
    printf("Writing convolved png to disk\r\n");
    stbi_write_png(fileNameOutConvolution, width - 2, height - 2, 4, imageDataConvolution, 4 * width);
    printf("Done\r\n");

    printf("Processing image minimum pooling\r\n");
    minPooling<<<numberOfBlocks, threadsPerBlock>>>(originalImage, imageDataMinPooling, width, height);
    cudaDeviceSynchronize();
    printf("Done\r\n");

    // Write image back to disk
    printf("Writing min pooling png to disk\r\n");
    stbi_write_png(fileNameOutMinPooling, width / 2, height / 2, 4, imageDataMinPooling, 4 * (width / 2));
    printf("Done\r\n");

    printf("Processing image maximum pooling\r\n");
    maxPooling<<<numberOfBlocks, threadsPerBlock>>>(originalImage, imageDataMaxPooling, width, height);
    cudaDeviceSynchronize();
    printf("Done\r\n");

    // Write image back to disk
    printf("Writing max pooling png to disk\r\n");
    stbi_write_png(fileNameOutMaxPooling, width / 2, height / 2, 4, imageDataMaxPooling, 4 * (width / 2));
    printf("Done\r\n");

    stbi_image_free(originalImageCPU);

	cudaFree(originalImage);
	cudaFree(imageDataGrayscale);
	cudaFree(imageDataConvolution);
	cudaFree(imageDataMinPooling);
	cudaFree(imageDataMaxPooling);

    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);
    cudaStreamDestroy(stream5);
    cudaStreamDestroy(stream6);
    cudaStreamDestroy(stream7);
    cudaStreamDestroy(stream8);
    cudaStreamDestroy(stream9);

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
    int counter = 0;

    // Iterate over the image in 2x2 blocks
    for (int y = 0; y < height; y += 2)
    {
        for (int x = 0; x < width; x += 2)
        {
            // For each channel, find the maximum value in the 2x2 block
            for (int c = 0; c < 4; c++)
            {
                Pixel *ptrPixelMinPooling = (Pixel *)&minPoolingImage[counter];
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
                ptrPixelMinPooling->r = min;
                ptrPixelMinPooling->g = min;
                ptrPixelMinPooling->b = min;
                ptrPixelMinPooling->a = min;
                counter++;
            }
        }
    }
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
