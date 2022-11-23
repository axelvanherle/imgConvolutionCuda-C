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

int threadNumber = 0;

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

void *runThreads(void *vargp)
{
    int threadId = threadNumber++;

    char INPUT_IMAGE[32] = "Images/img";
    char result[3];
    sprintf(result, "%d", threadId);
    strcat(INPUT_IMAGE, result);
    strcat(INPUT_IMAGE, ".png");

    // Open image
    int width, height, componentCount;
    printf("Loading %s file...\r\n",INPUT_IMAGE);
    unsigned char *imageData = stbi_load(INPUT_IMAGE, &width, &height, &componentCount, 4);
    if (!imageData)
    {
        printf("Failed to open Image\r\n");
        exit -1;
    }
    printf(" DONE %s\r\n",INPUT_IMAGE);

    // Validate image sizes
    if (width % 32 || height % 32)
    {
        // NOTE: Leaked memory of "imageData"
        printf("Width and/or Height is not dividable by 32! ( %s )\r\n",INPUT_IMAGE);
        exit -1;
    }

    // Process image on cpu
    printf("Processing %s...:\r\n",INPUT_IMAGE);
    ConvertImageToGrayCpu(imageData, width, height);
    printf(" DONE %s\r\n",INPUT_IMAGE);

    // Build output filename
    char OUTPUT_IMAGE[32] = "Output_Images/gray";
    char result1[3];
    sprintf(result1, "%d", threadId);
    strcat(OUTPUT_IMAGE, result1);
    strcat(OUTPUT_IMAGE, ".png");
    const char *fileNameOut = OUTPUT_IMAGE;
    // Write image back to disk
    printf("Writing %s to disk...\r\n",INPUT_IMAGE);
    stbi_write_png(fileNameOut, width, height, 4, imageData, 4 * width);
    printf("DONE\r\n");

    stbi_image_free(imageData);
    return NULL;
}

int main()
{
    pthread_t tid;

    for (int i = 0; i <= NUM_THREADS; i++)
    {
        pthread_create(&tid, NULL, runThreads, NULL);
    }

    pthread_exit(NULL);
}