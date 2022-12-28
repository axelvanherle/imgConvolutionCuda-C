#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <string.h>
#include <time.h>

typedef struct Pixel
{
    unsigned char r, g, b, a;
} Pixel;

void convertImageToGrayCpu(unsigned char *originalImage, unsigned char *ImageDataGrayscale, int width, int height);
void convolveImage(unsigned char *imageDataGrayscale, unsigned char *imageDataConvolution, int width, int height);
// Add function prototypes for min and max pooling here

int main()
{
    clock_t timer_start, timer_end;
    timer_start = clock(); // Start the timer
    
    for(int i = 0; i < 10; i++)
    {
        char inputImage[32] = "Images/img";
        char outputImageConvolution[32] = "Output_Images/Convolution/img";
        char outputImagePoolingMax[32] = "Output_Images/Pooling/imgMax";
        char outputImagePoolingMin[32] = "Output_Images/Pooling/imgMin";
        char imageNumber[3];
        
        // Make a complete path for the input (inputImage + imageNumber + extention)
        sprintf(imageNumber, "%d", i); // Get the number of the current iteration and convert it to char imageNumber
        strcat(inputImage, imageNumber);
        strcat(inputImage, ".png");

        ////////////////
        // Open Image //
        ////////////////
        // Open the image and allocate memory for each process
        int width, height, componentCount;
        printf("Loading %s.\r\n", inputImage);
        unsigned char *originalImage = stbi_load(inputImage, &width, &height, &componentCount, 4);
        unsigned char *imageDataGrayscale = (unsigned char *)malloc(width * height * 4);
        unsigned char *imageDataConvolution = (unsigned char *)malloc(width * height * 4);
        // Add memory allocation for *imageDataMinPooling here
        // Add memory allocation for *imageDataMaxPooling here

        if(!originalImage)
        {
            printf("Failed to open image!\r\n");
            stbi_image_free(originalImage);
            free(imageDataGrayscale);
            free(imageDataConvolution);
            // Free memory from imageDataMinPooling
            // Free memory from imageDataMaxPooling
            return -1;
        }

        if (width % 32 || height % 32)
        {
            // NOTE: Leaked memory of "imageData"
            printf("Width and/or Height is not dividable by 32!\r\n");
            stbi_image_free(originalImage);
            free(imageDataGrayscale);
            free(imageDataConvolution);
            // Free memory from imageDataMinPooling
            // Free memory from imageDataMaxPooling
            return -1;
        }

        printf("Done\r\n");

        ///////////////////////
        // Convert Grayscale //
        ///////////////////////
        // Convert image to grayscale on CPU
        printf("Processing image grayscale.\r\n");
        convertImageToGrayCpu(originalImage, imageDataGrayscale, width, height);
        printf("Done\r\n");

        ////////////////////
        // Convolve image //
        ////////////////////
        // Make a complete path for the convolved image (outputImage convolution + imageNumber + extention)
        strcat(outputImageConvolution, imageNumber);
        strcat(outputImageConvolution, ".png");

        // Convolve the image on CPU
        printf("Convolving image.\r\n");
        convolveImage(imageDataGrayscale, imageDataConvolution, width, height);
        printf("Done\r\n");

        // Write convolved image to disk
        printf("Writing image to disk\r\n");
        stbi_write_png(outputImageConvolution, width - 2, height - 2, 4, imageDataConvolution, 4 * width);
        printf("Done\r\n");

        /////////////////
        // Min Pooling //
        /////////////////
        // Make a complete path for max pooling
        sprintf(imageNumber, "%d", i); // Get the number of the current iteration and convert it to char imageNumber
        strcat(outputImagePoolingMin, imageNumber);
        strcat(outputImagePoolingMin, ".png");

        printf("DONE\r\n");

        /////////////////
        // Max Pooling //
        /////////////////
        sprintf(imageNumber, "%d", i); // Get the number of the current iteration and convert it to char imageNumber
        strcat(outputImagePoolingMax, imageNumber);
        strcat(outputImagePoolingMax, ".png");

        /////////////////
        // Free memory //
        /////////////////
        stbi_image_free(originalImage);
        free(imageDataGrayscale);
        free(imageDataConvolution);
        // Free memory from imageDataMinPooling
        // Free memory from imageDataMaxPooling

        printf("\r\nNext Image\r\n");
    }

    ///////////////
    // End timer //
    ///////////////
    timer_end = clock(); // end the timer
    double time_spent = (double)(timer_end - timer_start) / CLOCKS_PER_SEC;
    printf("\nProgram time: %.3fs\n", time_spent);

    return 0;
}

/*
 * Converts the input image to grayscale.
 */
void convertImageToGrayCpu(unsigned char *originalImage, unsigned char *imageDataGrayscale, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Pixel *ptrPixel = (Pixel *)&imageDataGrayscale[y * width * 4 + 4 * x];
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
 * Convolves the image
 */
void convolveImage(unsigned char *imageDataGrayscale, unsigned char *imageDataConvolution, int width, int height)
{
    int kernel[3][3] =
        {
            {1, 0, -1},
            {1, 0, -1},
            {1, 0, -1}};

    int pixels[3][3] = {0}; // Stores the temp value of each pixel that has been multiplied by the kernel
    int finalPixel = 0; // Stores the sum of all the calculated pixels in the kernel

    for (int y = 0; y < height - 2; y++)
    {
        for (int x = 0; x < width - 2; x++)
        {
            for (int i = 0; i <= 2; i++)
            {
                Pixel *ptrPixel = (Pixel *)&imageDataGrayscale[(y * width * 4 + 4 * x) + i * 4]; // Gets the top left pixel of the image in the first iteration

                pixels[0][i] = ptrPixel->r * kernel[0][i];
            }

            for (int i = 0; i <= 2; i++)
            {
                Pixel *ptrPixel = (Pixel *)&imageDataGrayscale[(y * width * 4 + 4 * x) + width * 4 + i * 4]; // Gets the first pixel of the second row in the first iteration

                pixels[1][i] = ptrPixel->r * kernel[1][i];
            }

            for (int i = 0; i <= 2; i++)
            {
                Pixel *ptrPixel = (Pixel *)&imageDataGrayscale[(y * width * 4 + 4 * x) + (2 * width * 4) + i * 4]; // Gets the first pixel of the third row in the first iteration

                pixels[2][i] = ptrPixel->r * kernel[2][i];
            }

            finalPixel = (pixels[0][0] + pixels[0][1] + pixels[0][2] + pixels[1][0] + pixels[1][1] + pixels[1][2] + pixels[2][0] + pixels[2][1] + pixels[2][2]) / 9; 

            Pixel *ptrPixel = (Pixel *)&imageDataConvolution[(y * width * 4 + 4 * x)];
            ptrPixel->r = finalPixel;
            ptrPixel->g = finalPixel;
            ptrPixel->b = finalPixel;
            ptrPixel->a = 255;
        }
    }
}