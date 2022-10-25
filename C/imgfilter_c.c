#include <stdio.h>
#include <stdlib.h>

#define INPUT_FILE "Images/test.bmp"
#define OUTPUT_FILE "Images/Output.bmp"

// Function opens the image.
FILE *openBMP(void);
// Funtion opens the target image.
FILE *openTargetBMP(void);
// Put the first 54 bytes of inputBMP in the header and writes away the header in targetBMP.
void readHeader(FILE *inputBMP, unsigned char *header, FILE *targetBMP);
// Function calculates the height of the image.
void calcHeight(unsigned char *header, signed int *height);
// Function calculates the width of the image.
void calcWidth(unsigned char *header, signed int *width);
// Calculates the number of pixels and stores it in numberOfPixels.
void calcPixels(signed int *height, signed int *width, signed int *numberOfPixels);
// Prints the pixels from whatever char array (assuming its a bmp image thats divideable by 4 without header) we input.
void printPixels(unsigned char *imagePixels, signed int *numberOfPixels);
// Releases all memory we used on the heap.
void cleanup(unsigned char *header, signed int *height, signed int *width, signed int *numberOfPixels, unsigned char *originalPixels, unsigned char *editedPixels, FILE *inputBMP, FILE *targetBMP);

int main()
{
    FILE *inputBMP = openBMP();        // Opens the BMP file.
    FILE *targetBMP = openTargetBMP(); // Opens the BMP output file.
    unsigned char *header = (unsigned char *)malloc(54 * sizeof(unsigned char));
    signed int *height = (signed int *)malloc(sizeof(signed int));
    signed int *width = (signed int *)malloc(sizeof(signed int));
    signed int *numberOfPixels = (signed int *)malloc(sizeof(signed int));

    readHeader(inputBMP, header, targetBMP); // Reads the header.
    calcHeight(header, height);              // Calculates height BMP file.
    calcWidth(header, width);                // Calculates width BMP file
    if (*width % 4 != 0 && *height % 4 != 0)
    {
        printf("Incompatible Image\n");
        fclose(inputBMP);
        fclose(targetBMP);
        free(width);
        free(height);
        exit(-1);
    }
    calcPixels(height, width, numberOfPixels); // Calculates the number of pixels.
    unsigned char *originalPixels = (unsigned char *)malloc(*numberOfPixels * 3);
    unsigned char *editedPixels = (unsigned char *)malloc(*numberOfPixels * 3);
    fread(originalPixels, 1, *numberOfPixels * 3, inputBMP);
    printPixels(originalPixels, numberOfPixels);
    /*
     *   insert filter
     */

    cleanup(header, height, width, numberOfPixels, originalPixels, editedPixels, inputBMP, targetBMP);

    return 0;
}

FILE *openBMP() // Function opens the image.
{
    FILE *inputBMP = fopen(INPUT_FILE, "rb");

    if (inputBMP == NULL)
    {
        printf("%s\n", "Error: Unable to open the file!\n");
        exit(-1);
    }

    return inputBMP;
}

FILE *openTargetBMP() // Funtion opens the target image.
{
    FILE *targetBMP = fopen(OUTPUT_FILE, "wb");

    if (targetBMP == NULL)
    {
        printf("%s\n", "Error: Unable to create the file!\n");
        exit(-1);
    }

    return targetBMP;
}

void readHeader(FILE *inputBMP, unsigned char *header, FILE *targetBMP)
{
    fread(header, 1, 54, inputBMP); // Put the first 54 bytes of inputBMP in the header.

    fwrite(header, 1, 54, targetBMP); // Writes away the header in targetBMP.
}

void calcHeight(unsigned char *header, signed int *height) // Function calculates the height of the image.
{
    *height = header[25] << 24 | header[24] << 16 | header[23] << 8 | header[22]; // Result: height = (8 bits header[21]) (8 bits header[20]) (8 bits header[19]) (8 bits header[18]).
    printf("\nheight: %dpx\n", *height);
}

void calcWidth(unsigned char *header, signed int *width) // Function calculates the width of the image
{
    *width = header[21] << 24 | header[20] << 16 | header[19] << 8 | header[18]; // Result: width = (8 bits header[21]) (8 bits header[20]) (8 bits header[19]) (8 bits header[18]).
    printf("width: %dpx\n", *width);
}

void calcPixels(signed int *height, signed int *width, signed int *numberOfPixels)
{
    *numberOfPixels = *height * *width;
    printf("Total number of pixels: %dpx\n", *numberOfPixels);
}

void printPixels(unsigned char *imagePixels, signed int *numberOfPixels)
{
    for (int i = 0; i < *numberOfPixels * 3; i++)
    {
        if (i % 3 == 0)
        {
            printf("\n");
        }
        printf(" %x", imagePixels[i]);
    }
}

void cleanup(unsigned char *header, signed int *height, signed int *width, signed int *numberOfPixels, unsigned char *originalPixels, unsigned char *editedPixels, FILE *inputBMP, FILE *targetBMP)
{
    free(header);
    free(height);
    free(width);
    free(numberOfPixels);
    free(originalPixels);
    free(editedPixels);

    fclose(inputBMP);
    fclose(targetBMP);
}
