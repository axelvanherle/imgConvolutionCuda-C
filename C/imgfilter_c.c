#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <time.h>

#define INPUT_IMAGE "Images/img9.png"

typedef struct Pixel
{
    unsigned char r, g, b, a;
} Pixel;

void ConvertImageToGrayCpu(unsigned char *originalImage, unsigned char *imageRGBA, int width, int height);
void convolveImage(unsigned char *imageRGBA, unsigned char *imageTest, int width, int height);
void minPooling(unsigned char *originalImage, unsigned char *minPoolingImage, int width, int height);
void maxPooling(unsigned char *originalImage, unsigned char *maxPoolingImage, int width, int height);

int main(int argc, char **argv)
{
    clock_t timer_start, timer_end;
    timer_start = clock();

    // Open image
    int width, height, componentCount;
    printf("Loading png file...\r\n");
    unsigned char *originalImage = stbi_load(INPUT_IMAGE, &width, &height, &componentCount, 4); // Saves original image
    unsigned char *imageData = (unsigned char *)malloc(width * height * 4);                     // Saves grayscale image
    unsigned char *imageDataTest = (unsigned char *)malloc(width * height * 4);                 // Saves output image
    unsigned char *imageDataMinPooling = (unsigned char *)malloc(width * height * 4);           // Saves Min pooling image
    unsigned char *imageDataMaxPooling = (unsigned char *)malloc(width * height * 4);           // Saves Max pooling image

    if (!originalImage)
    {
        printf("Failed to open Image\r\n");
        stbi_image_free(originalImage);
        free(imageData);
        free(imageDataTest);
        free(imageDataMinPooling);
        free(imageDataMaxPooling);
        return -1;
    }

    printf("DONE \r\n");

    // Validate image sizes
    if (width % 32 || height % 32)
    {
        // NOTE: Leaked memory of "imageData"
        printf("Width and/or Height is not dividable by 32!\r\n");
        stbi_image_free(originalImage);
        free(imageData);
        free(imageDataTest);
        free(imageDataMinPooling);
        free(imageDataMaxPooling);
        return -1;
    }

    // Process image on cpu
    printf("Processing image grayscale...:\r\n");
    ConvertImageToGrayCpu(originalImage, imageData, width, height);
    printf("DONE \r\n");

    // Process image on cpu
    printf("Processing image convolution...:\r\n");
    convolveImage(imageData, imageDataTest, width, height);
    printf("DONE \r\n");

    // Build output filename
    const char *fileNameOutConvolution = "OutputConvolution.png";
    const char *fileNameOutMinPooling = "OutputMinPooling.png";
    const char *fileNameOutMaxPooling = "OutputMaxPooling.png";

    // Write image back to disk
    printf("Writing convolved png to disk...\r\n");
    stbi_write_png(fileNameOutConvolution, width - 2, height - 2, 4, imageDataTest, 4 * width);
    printf("DONE\r\n");

    printf("Processing image minimum pooling\r\n");
    minPooling(originalImage, imageDataMinPooling, width, height);
    printf("DONE\r\n");

    // Write image back to disk
    printf("Writing min pooling png to disk...\r\n");
    stbi_write_png(fileNameOutMinPooling, width / 2, height / 2, 4, imageDataMinPooling, 4 * (width / 2));
    printf("DONE\r\n");

    printf("Processing image maximum pooling\r\n");
    maxPooling(originalImage, imageDataMaxPooling, width, height);
    printf("DONE\r\n");

    // Write image back to disk
    printf("Writing max pooling png to disk...\r\n");
    stbi_write_png(fileNameOutMaxPooling, width / 2, height / 2, 4, imageDataMaxPooling, 4 * (width / 2));
    printf("DONE\r\n");

    stbi_image_free(originalImage);
    free(imageData);
    free(imageDataTest);
    free(imageDataMinPooling);
    free(imageDataMaxPooling);

    timer_end = clock(); // end the timer
    double time_spent = (double)(timer_end - timer_start) / CLOCKS_PER_SEC;
    printf("\nProgram time: %.3fs\n", time_spent);

    return 0;
}

void ConvertImageToGrayCpu(unsigned char *originalImage, unsigned char *imageRGBA, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Pixel *ptrPixel = (Pixel *)&imageRGBA[y * width * 4 + 4 * x];
            Pixel *ptrPixelOriginal = (Pixel *)&originalImage[y * width * 4 + 4 * x];
            unsigned char pixelValue = (unsigned char)(ptrPixelOriginal->r * 0.2126f + ptrPixelOriginal->g * 0.7152f + ptrPixelOriginal->b * 0.0722f);
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