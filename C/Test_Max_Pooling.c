#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define INPUT_IMAGE "Images/img9.png"

typedef struct Pixel
{
    unsigned char r, g, b, a;
} Pixel;

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

/*
    This function convolves the image.
*/
void convolveImage(unsigned char *imageRGBA, unsigned char *imageTest, int width, int height)
{
    int kernel[3][3] =
    {
        {1, 0, -1},
        {1, 0, -1},
        {1, 0, -1}
    };

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

/* void minPooling(unsigned char *imageRGBA, unsigned char *imageTest, int width, int height)
{
    int pixels[3][3] = {0};
    int finalPixel = 300;

    for(int y = 0; y < height; y += 3 * 4)
    {
        for(int x = 0; x < width; x += 3 * 4)
        {
            for(int i = 0; i <= 2; i++)
            {
                Pixel *ptrPixel = (Pixel *)&imageRGBA[(y * width * 4 + 4 * x) + i * 4];

                pixels[0][i] = ptrPixel->r;
            }

            for(int i = 0; i <= 2; i++)
            {
                Pixel *ptrPixel = (Pixel *)&imageRGBA[(y * width * 4 + 4 * x) + width * 4 + i * 4]; 

                pixels[1][i] = ptrPixel->r;
            }

            for(int i = 0; i <= 2; i++)
            {
                Pixel *ptrPixel = (Pixel *)&imageRGBA[(y * width * 4 + 4 * x) + (2 * width * 4) + i * 4];

                pixels[2][i] = ptrPixel->r;
            }

            for(int i = 0; i <= 2; i++)
            {
                for(int j = 0; j <= 2; j++)
                {
                    if(pixels[i][j] < finalPixel)
                    {
                        finalPixel = pixels[i][j];
                    }
                }
            }

            Pixel *ptrPixel = (Pixel *)&imageTest[(y * width * 4 + 4 * x)];
            ptrPixel->r = finalPixel;
            ptrPixel->g = finalPixel;
            ptrPixel->b = finalPixel;
            ptrPixel->a = 255;
        }
    }
}
*/

void maxPooling(unsigned char *imageRGBA, unsigned char *imageTest, int width, int height)
{

int i, j;  // Loop variables
    int max;  // Variable to hold the max value in the current pooling window

    // Loop through the picture with a pooling window of size 3x3
    for (i = 0; i < HEIGHT - 2; i += 3) {
        for (j = 0; j < WIDTH - 2; j += 3) {
            max = picture[i][j];  // Initialize max to the current element in the pooling window
            // Find the highest value in the current pooling window
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    if (picture[i+k][j+l] > max) {
                        max = picture[i+k][j+l];
                    }
                }
}
        }
    }
     return 0;
}

int main(int argc, char **argv)
{
    // Open image
    int width, height, componentCount;
    printf("Loading png file...\r\n");
    unsigned char *imageData = (unsigned char *)malloc(width*height*4); // Saves grayscale image
    unsigned char *imageDataTest = (unsigned char *)malloc(width*height*4); // Saves output image
    unsigned char *originalImage = stbi_load(INPUT_IMAGE, &width, &height, &componentCount, 4); // Saves original image
    if (!originalImage)
    {
        printf("Failed to open Image\r\n");
        stbi_image_free(imageData);
        free(imageDataTest);
        free(originalImage);
        return -1;
    }

    printf("DONE \r\n");

    // Validate image sizes
    if (width % 32 || height % 32)
    {
        // NOTE: Leaked memory of "imageData"
        printf("Width and/or Height is not dividable by 32!\r\n");
        stbi_image_free(imageData);
        free(imageDataTest);
        free(originalImage);
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

    // Validate image sizes
    if (width % 3 || height % 3)
    {
        // NOTE: Leaked memory of "imageData"
        printf("Width and/or Height is not dividable by 3!\r\n");
        stbi_image_free(imageData);
        free(imageDataTest);
        free(originalImage);
        return -1;
    }

    printf("Processing image minimum pooling\r\n");
    minPooling(originalImage, imageDataTest, width, height);
    printf("DONE\r\n");

    // Write image back to disk
    printf("Writing min pooling png to disk...\r\n");
    stbi_write_png(fileNameOutMinPooling, width / 3, height / 3, 4, imageDataTest, 4 * width);
    printf("DONE\r\n");

    printf("Processing image maximum pooling\r\n");
    maxPooling(originalImage, imageDataTest, width, height);
    printf("DONE\r\n");

    // Write image back to disk
    printf("Writing max pooling png to disk...\r\n");
    stbi_write_png(fileNameOutMaxPooling, width / 3, height / 3, 4, imageDataTest, 4 * width);
    printf("DONE\r\n");

    stbi_image_free(imageData);
    free(imageDataTest);
    free(originalImage);

    return 0;
}