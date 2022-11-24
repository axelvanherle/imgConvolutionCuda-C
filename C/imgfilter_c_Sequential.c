#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <time.h>


// Hard coded the input files
#define INPUT_IMAGE_0  "Images/img0.png"
#define INPUT_IMAGE_1  "Images/img1.png"
#define INPUT_IMAGE_2  "Images/img2.png"
#define INPUT_IMAGE_3  "Images/img3.png"
#define INPUT_IMAGE_4  "Images/img4.png"
#define INPUT_IMAGE_5  "Images/img5.png"
#define INPUT_IMAGE_6  "Images/img6.png"
#define INPUT_IMAGE_7  "Images/img7.png"
#define INPUT_IMAGE_8  "Images/img8.png"
#define INPUT_IMAGE_9  "Images/img9.png"

// Hard coded the output files
#define OUTPUT_IMAGE_0  "Output_Images/img0.png"
#define OUTPUT_IMAGE_1  "Output_Images/img1.png"
#define OUTPUT_IMAGE_2  "Output_Images/img2.png"
#define OUTPUT_IMAGE_3  "Output_Images/img3.png"
#define OUTPUT_IMAGE_4  "Output_Images/img4.png"
#define OUTPUT_IMAGE_5  "Output_Images/img5.png"
#define OUTPUT_IMAGE_6  "Output_Images/img6.png"
#define OUTPUT_IMAGE_7  "Output_Images/img7.png"
#define OUTPUT_IMAGE_8  "Output_Images/img8.png"
#define OUTPUT_IMAGE_9  "Output_Images/img9.png"

void processImage(char *inputFile, char *outputFile, int imgCounter);           // Opens the image, starts the convertion function and writes the output image
void ConvertImageToGrayCpu(unsigned char *imageRGBA, int width, int height);    // Converts the image to grayscale

typedef struct Pixel
{
    unsigned char r, g, b, a;
} Pixel;

int main()
{
    int imgCounter = 1;
    clock_t timer_start, timer_end;

    timer_start = clock(); // start the timer

    processImage(INPUT_IMAGE_0, OUTPUT_IMAGE_0, imgCounter);
    imgCounter++;

    processImage(INPUT_IMAGE_1, OUTPUT_IMAGE_1, imgCounter);
    imgCounter++;

    processImage(INPUT_IMAGE_2, OUTPUT_IMAGE_2, imgCounter);
    imgCounter++;

    processImage(INPUT_IMAGE_3, OUTPUT_IMAGE_3, imgCounter);
    imgCounter++;

    processImage(INPUT_IMAGE_4, OUTPUT_IMAGE_4, imgCounter);
    imgCounter++;

    processImage(INPUT_IMAGE_5, OUTPUT_IMAGE_5, imgCounter);
    imgCounter++;

    processImage(INPUT_IMAGE_6, OUTPUT_IMAGE_6, imgCounter);
    imgCounter++;

    processImage(INPUT_IMAGE_7, OUTPUT_IMAGE_7, imgCounter);
    imgCounter++;

    processImage(INPUT_IMAGE_8, OUTPUT_IMAGE_8, imgCounter);
    imgCounter++;

    processImage(INPUT_IMAGE_9, OUTPUT_IMAGE_9, imgCounter);

    timer_end = clock(); // end the timer
    double time_spent = (double)(timer_end - timer_start) / CLOCKS_PER_SEC;
    printf("\nProgram time: %.3fs\n", time_spent);

    return 0;
}

void processImage(char *inputFile, char *outputFile, int imgCounter)
{
    // Open image
    int width, height, componentCount;

    printf("Loading png file... [%d / 10]\r\n", imgCounter);
    unsigned char *imageData = stbi_load(inputFile, &width, &height, &componentCount, 4);
    if (!imageData)
    {
        printf("Failed to open Image\r\n");
        exit(-1);
    }
    printf(" DONE \r\n");

    // Validate image sizes
    if (width % 32 || height % 32)
    {
        // NOTE: Leaked memory of "imageData"
        printf("Width and/or Height is not dividable by 32! [%d / 10]\r\n", imgCounter);
        exit(-1);
    }

    // Process image on cpu
    printf("Processing image...: [%d / 10]\r\n", imgCounter);
    ConvertImageToGrayCpu(imageData, width, height);
    printf(" DONE \r\n");

    // Write image back to disk
    printf("Writing to disk... [%d / 10]\r\n", imgCounter);
    stbi_write_png(outputFile, width, height, 4, imageData, 4 * width);
    printf("DONE [%d / 10]\r\n", imgCounter);

    stbi_image_free(imageData);
}

void ConvertImageToGrayCpu(unsigned char *imageRGBA, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Pixel *ptrPixel = (Pixel *)&imageRGBA[y * width * 4 + 4 * x];.
            unsigned char pixelValue = (unsigned char)(ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f);
            ptrPixel->r = pixelValue;
            ptrPixel->g = pixelValue;
            ptrPixel->b = pixelValue;
            ptrPixel->a = 255;
        }
    }
}