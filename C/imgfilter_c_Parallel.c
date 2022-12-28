#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

// The amount of pictures you have.
// Relies on the fact you name your images correct (ie img11.png).
#define NUM_THREADS 9

// Used to store which thread we are in.
int threadNumber = 0;

typedef struct Pixel
{
    unsigned char r, g, b, a;
} Pixel;

// Function to convert the image from colour to grayscale.
void ConvertImageToGrayCpu(unsigned char *imageRGBA, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Pixel *ptrPixel = (Pixel *)&imageRGBA[y * width * 4 + 4 * x];
            unsigned char pixelValue = (unsigned char)(ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f);
            ptrPixel->r = pixelValue;
            ptrPixel->g = pixelValue;
            ptrPixel->b = pixelValue;
            ptrPixel->a = 255;
        }
    }
}

void convolveImage(unsigned char *imageRGBA, unsigned char *imageTest, int width, int height)
{
    int kernel[3][3] =
        {
            {1, 0, -1},
            {1, 0, -1},
            {1, 0, -1}};

    int pixels[3][3] = {0};
    int finalPixel = 0;

    for (int y = 0; y < height - 2; y++)
    {
        for (int x = 0; x < width - 2; x++)
        {
            for (int i = 0; i <= 2; i++)
            {
                Pixel *ptrPixel = (Pixel *)&imageRGBA[(y * width * 4 + 4 * x) + i * 4];

                pixels[0][i] = ptrPixel->r * kernel[0][i];
            }

            for (int i = 0; i <= 2; i++)
            {
                Pixel *ptrPixel = (Pixel *)&imageRGBA[(y * width * 4 + 4 * x) + width * 4 + i * 4];

                pixels[1][i] = ptrPixel->r * kernel[1][i];
            }

            for (int i = 0; i <= 2; i++)
            {
                Pixel *ptrPixel = (Pixel *)&imageRGBA[(y * width * 4 + 4 * x) + (2 * width * 4) + i * 4];

                pixels[2][i] = ptrPixel->r * kernel[2][i];
            }

            finalPixel = (pixels[0][0] + pixels[0][1] + pixels[0][2] + pixels[1][0] + pixels[1][1] + pixels[1][2] + pixels[2][0] + pixels[2][1] + pixels[2][2]) / 9;

            Pixel *ptrPixel = (Pixel *)&imageTest[(y * width * 4 + 4 * x)];
            ptrPixel->r = finalPixel;
            ptrPixel->g = finalPixel;
            ptrPixel->b = finalPixel;
            ptrPixel->a = 255;
        }
    }
}

/*
    This function convolves the image.
*/
void minPooling(unsigned char *originalImage, unsigned char *minPoolingImage, int width, int height)
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

void maxPooling(unsigned char *originalImage, unsigned char *maxPoolingImage, int width, int height)
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

// This function runs all the threads
void *runThreads(void *vargp)
{
    // Get which thread we are in, and do + for the next thread.
    int threadId = threadNumber++;

    // Make a usefull string that we can use to open the correct image.
    char INPUT_IMAGE[32] = "Images/img";
    char result[3];
    sprintf(result, "%d", threadId);
    strcat(INPUT_IMAGE, result);
    strcat(INPUT_IMAGE, ".png");

    // Open image
    int width, height, componentCount;
    printf("Loading %s file...\r\n", INPUT_IMAGE);
    unsigned char *imageData = stbi_load(INPUT_IMAGE, &width, &height, &componentCount, 4);
    if (!imageData)
    {
        printf("Failed to open Image\r\n");
        exit(-1);
    }
    printf(" DONE %s\r\n", INPUT_IMAGE);

    // Validate image sizes
    if (width % 32 || height % 32)
    {
        // NOTE: Leaked memory of "imageData"
        printf("Width and/or Height is not dividable by 32! ( %s )\r\n", INPUT_IMAGE);
        exit(-1);
    }

    // Process image on cpu
    printf("Processing %s...:\r\n", INPUT_IMAGE);
    ConvertImageToGrayCpu(imageData, width, height);
    printf(" DONE %s\r\n", INPUT_IMAGE);

    // Process image on cpu
    printf("Processing image...:\r\n");
    convolveImage(imageData, width, height);
    printf(" DONE \r\n");

    // Build output filename
    char OUTPUT_IMAGE[32] = "Output_Images/gray";
    char result1[3];
    sprintf(result1, "%d", threadId);
    strcat(OUTPUT_IMAGE, result1);
    strcat(OUTPUT_IMAGE, ".png");
    const char *fileNameOut = OUTPUT_IMAGE;

    // Write image back to disk
    printf("Writing %s to disk...\r\n", INPUT_IMAGE);
    stbi_write_png(fileNameOut, width, height, 4, imageData, 4 * width);
    printf("DONE\r\n");

    stbi_image_free(imageData);
    return NULL;
}

int main()
{
    pthread_t tid;

    // Make NUM_THREADS amount of threads.
    for (int i = 0; i <= NUM_THREADS; i++)
    {
        pthread_create(&tid, NULL, runThreads, NULL);
    }

    pthread_exit(NULL);
    return 0;
}