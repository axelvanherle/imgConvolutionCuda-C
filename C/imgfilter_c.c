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

/*
    This function convolves the image.
*/
void convolveImage(unsigned char *imageRGBA, unsigned char *imageTest, int width, int height)
{
    int kernel[3][3] = {
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

int main(int argc, char **argv)
{
    // Open image
    int width, height, componentCount;
    printf("Loading png file...\r\n");
    unsigned char *imageData = stbi_load(INPUT_IMAGE, &width, &height, &componentCount, 4);
    unsigned char *imageDataTest = (unsigned char *)malloc(width*height*4);
    if (!imageData)
    {
        printf("Failed to open Image\r\n");
        return -1;
    }
    printf(" DONE \r\n");

    // Validate image sizes
    if (width % 32 || height % 32)
    {
        // NOTE: Leaked memory of "imageData"
        printf("Width and/or Height is not dividable by 32!\r\n");
        return -1;
    }

    // Process image on cpu
    printf("Processing image...:\r\n");
    ConvertImageToGrayCpu(imageData, width, height);
    printf(" DONE \r\n");

    // Process image on cpu
    printf("Processing image...:\r\n");
    convolveImage(imageData, imageDataTest, width, height);
    printf(" DONE \r\n");

    // Build output filename
    const char *fileNameOut = "output.png";

    // Write image back to disk
    printf("Writing png to disk...\r\n");
    stbi_write_png(fileNameOut, width - 2, height - 2, 4, imageDataTest, 4 * width);
    printf("DONE\r\n");

    // Validate image sizes
    if (width % 3 || height % 3)
    {
        // NOTE: Leaked memory of "imageData"
        printf("Width and/or Height is not dividable by 3!\r\n");
        stbi_image_free(imageData);
        free(imageDataTest);
        return -1;
    }

    stbi_image_free(imageData);
    free(imageDataTest);

    return 0;
}