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

void minPooling(unsigned char *originalImage, unsigned char *minPoolingImage, int width, int height)
{
    unsigned char pixelsR[2][2] = {0};
    unsigned char pixelsG[2][2] = {0};
    unsigned char pixelsB[2][2] = {0};

    Pixel *ptrPixel = (Pixel *)&originalImage[1];
    unsigned char finalPixelR = ptrPixel->r;
    unsigned char finalPixelG = ptrPixel->g;
    unsigned char finalPixelB = ptrPixel->b;
    unsigned char finalPixelA = ptrPixel->a;

    int counter = 0;

    for(int y = 0; y < height; y += 2)
    {
        for(int x = 0; x < width; x += 2)
        {
            for(int i = 0; i <= 1; i++)
            {
                Pixel *ptrPixel = (Pixel *)&originalImage[(y * width * 4 + 4 * x) + i * 4];

                pixelsR[0][i] = ptrPixel->r;
                pixelsG[0][i] = ptrPixel->g;
                pixelsB[0][i] = ptrPixel->b;
            }

            for(int i = 0; i <= 1; i++)
            {
                Pixel *ptrPixel = (Pixel *)&originalImage[(y * width * 4 + 4 * x) + width * 4 + i * 4]; 

                pixelsR[1][i] = ptrPixel->r;
                pixelsG[1][i] = ptrPixel->g;
                pixelsB[1][i] = ptrPixel->b;
            }

            finalPixelA = ptrPixel->a;

            for(int i = 0; i <= 1; i++)
            {
                for(int j = 0; j <= 1; j++)
                {
                    if(pixelsR[i][j] < finalPixelR)
                    {
                        finalPixelR = pixelsR[i][j];
                    }
                }
            }

            for(int i = 0; i <= 1; i++)
            {
                for(int j = 0; j <= 1; j++)
                {
                    if(pixelsG[i][j] < finalPixelG)
                    {
                        finalPixelG = pixelsG[i][j];
                    }
                }
            }

            for(int i = 0; i <= 1; i++)
            {
                for(int j = 0; j <= 1; j++)
                {
                    if(pixelsB[i][j] < finalPixelB)
                    {
                        finalPixelB = pixelsB[i][j];
                    }
                }
            }

            Pixel *ptrPixelMinPooling = (Pixel *)&minPoolingImage[counter];
            ptrPixelMinPooling->r = finalPixelR;
            ptrPixelMinPooling->g = finalPixelG;
            ptrPixelMinPooling->b = finalPixelB;
            ptrPixelMinPooling->a = 255;

            counter++;
        }
    }
}
/*
void maxPooling(unsigned char *imageRGBA, unsigned char *imageTest, int width, int height)
{
    for (int y = 0; y < height; y += 2)
    {
        for (int x = 0; x < width; x += 2)
        {
            // For each channel, find the maximum value in the 2x2 block
            for (int c = 0; c < 4; c++)
            {
                unsigned char max = 0;
                for (int dy = 0; dy < 2; dy++)
                {
                    for (int dx = 0; dx < 2; dx++)
                    {
                        //unsigned char value = imageRGBA[y + dy][x + dx][c];
                        if (value > max)
                        {
                            max = value;
                        }
                    }
                }
                // Store the maximum value in the result array
                //imageTest[y / 2][x / 2][c] = max;
            }
        }
    }
}
*/
int main(int argc, char **argv)
{
    // Open image
    int width, height, componentCount;
    printf("Loading png file...\r\n");
    unsigned char *originalImage = stbi_load(INPUT_IMAGE, &width, &height, &componentCount, 4); // Saves original image
    unsigned char *imageData = (unsigned char *)malloc(width * height * 4);                     // Saves grayscale image
    unsigned char *imageDataTest = (unsigned char *)malloc(width * height * 4);                 // Saves output image
    unsigned char *imageDataMinPooling = (unsigned char *)malloc(width * height * 4);           // Saves Min pooling image
    if (!originalImage)
    {
        printf("Failed to open Image\r\n");
        stbi_image_free(originalImage);
        free(imageData);
        free(imageDataTest);
        free(imageDataMinPooling);
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
        stbi_image_free(originalImage);
        free(imageData);
        free(imageDataTest);
        free(imageDataMinPooling);
        return -1;
    }

    printf("Processing image minimum pooling\r\n");
    minPooling(originalImage, imageDataMinPooling, width, height);
    printf("DONE\r\n");

    // Write image back to disk
    printf("Writing min pooling png to disk...\r\n");
    stbi_write_png(fileNameOutMinPooling, width, height, 4, imageDataMinPooling, 4 * width);
    printf("DONE\r\n");
    /*
    printf("Processing image maximum pooling\r\n");
    maxPooling(originalImage, imageDataTest, width, height);
    printf("DONE\r\n");

    // Write image back to disk
    printf("Writing max pooling png to disk...\r\n");
    stbi_write_png(fileNameOutMaxPooling, width / 3, height / 3, 4, imageDataTest, 4 * width);
    printf("DONE\r\n");
    */
    stbi_image_free(originalImage);
    free(imageData);
    free(imageDataTest);
    free(imageDataMinPooling);

    return 0;
}