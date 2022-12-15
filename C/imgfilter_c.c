#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define INPUT_IMAGE "Images/img5.png"

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

int main(int argc, char** argv)
{
    // Open image
    int width, height, componentCount;
    printf("Loading png file...\r\n");
    unsigned char *imageData = stbi_load(INPUT_IMAGE, &width, &height, &componentCount, 4);
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

    // Build output filename
    const char *fileNameOut = "gray.png";

    // Write image back to disk
    printf("Writing png to disk...\r\n");
    stbi_write_png(fileNameOut, width, height, 4, imageData, 4 * width);
    printf("DONE\r\n");

    stbi_image_free(imageData);
}