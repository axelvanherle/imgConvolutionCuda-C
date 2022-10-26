#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define INPUT_FILE "Images/image8.bmp"
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
// Writes the edited pixels in the new file.
void writeNewPixels(unsigned char *editedPixels, signed int *numberOfPixels, FILE *targetBMP);
// Prints the info of the file we are manipulating.
void printFileInfo(void);
// Releases all memory we used on the heap.
void cleanup(unsigned char *header, signed int *height, signed int *width, signed int *numberOfPixels, unsigned char *originalPixels, unsigned char *editedPixels, FILE *inputBMP, FILE *targetBMP);

int main()
{
    printFileInfo();

    FILE *inputBMP = openBMP();        // Opens the BMP file.
    FILE *targetBMP = openTargetBMP(); // Opens the BMP output file.

    // Variable Declaration.
    unsigned char *header = (unsigned char *)malloc(54 * sizeof(unsigned char)); // Used to save the header from the BMP image.
    signed int *height = (signed int *)malloc(sizeof(signed int));               // Used to save the height of the original image.
    signed int *width = (signed int *)malloc(sizeof(signed int));                // Used to save the height of the original image.
    signed int *numberOfPixels = (signed int *)malloc(sizeof(signed int));       // Used to save the total number of pixels from the original image.

    readHeader(inputBMP, header, targetBMP); // Reads the header.
    calcHeight(header, height);              // Calculates height BMP file.
    calcWidth(header, width);                // Calculates width BMP file
    // This checks if the pixels from the BMP image are divisible by 4. If they arent, the image handeling gets difficult.
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

    unsigned char *originalPixels = (unsigned char *)malloc(*numberOfPixels * 3); // Used to save the "pixels" from the original image.
    unsigned char *editedPixels = (unsigned char *)malloc(*numberOfPixels * 3);   // Used to save the "pixels" from the original image that we edited before convection.

    fread(originalPixels, 1, (*numberOfPixels * 3), inputBMP); // Reads the pixels from the original image into *originalPixels.
    // printPixels(originalPixels, numberOfPixels); // Prints the pixels from the original image.

    // Insert the filter here.

    writeNewPixels(originalPixels, numberOfPixels, targetBMP);                                         // Writes the pixels to the output file.
    cleanup(header, height, width, numberOfPixels, originalPixels, editedPixels, inputBMP, targetBMP); // Memory cleanup.

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
    printf("image characteristics: \n");
    for (int i = 0; i < *numberOfPixels * 3; i++)
    {
        if (i % 16 == 0)
        {
            printf("\n");
        }
        printf(" %x", imagePixels[i]);
    }
}

void writeNewPixels(unsigned char *editedPixels, signed int *numberOfPixels, FILE *targetBMP)
{
    uint8_t offsetHeader = 54; // Header takes first 54 bytes in the new file
    fseek(targetBMP, offsetHeader, SEEK_SET);
    fwrite(editedPixels, 1, (3 * (*numberOfPixels)), targetBMP);
}

void printFileInfo(void)
{
    printf("input file: %s\n", INPUT_FILE);
    printf("output file: %s\n", OUTPUT_FILE);
}

void cleanup(unsigned char *header, signed int *height, signed int *width, signed int *numberOfPixels, unsigned char *originalPixels, unsigned char *editedPixels, FILE *inputBMP, FILE *targetBMP)
{
    // Goodbye!
    free(header);
    free(height);
    free(width);
    free(numberOfPixels);
    free(originalPixels);
    free(editedPixels);

    fclose(inputBMP);
    fclose(targetBMP);
}
