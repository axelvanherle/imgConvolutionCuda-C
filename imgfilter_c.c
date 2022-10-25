#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_FILE "Images/testImage.bmp"
#define OUTPUT_FILE "Images/Output.bmp"

FILE *openBMP() //Functie opent de afbeelding
{  
    FILE *inputBMP = fopen(INPUT_FILE, "rb");

    if(inputBMP == NULL)
    {
        printf("%s\n", "Error: Unable to open the file!\n");
        exit(-1);
    }

    return inputBMP;
}

FILE *openTargetBMP() //Functie opent de target afbeelding
{
    FILE *targetBMP = fopen(OUTPUT_FILE, "wb");

    if(targetBMP == NULL)
    {
        printf("%s\n", "Error: Unable to create the file!\n");
        exit(-1);
    }
    
    return targetBMP;
}

void readHeader(FILE *inputBMP, unsigned char *header, FILE *targetBMP)
{
    fread(header, 1, 54, inputBMP); //Zet de eerste 54 bites van inputBMP in de header.

    fwrite(header, 1, 54, targetBMP); //Schrijft de header weg in targetBMP
}

void calcHeight(unsigned char *header, signed int *hoogte) //Functie berekend de hoogte van de afbeelding
{
    *hoogte = header[25] << 24 | header[24] << 16 | header[23] << 8 | header[22]; //Resultaat: hoogte = (8 bits header[21]) (8 bits header[20]) (8 bits header[19]) (8 bits header[18])
    printf("\nHoogte: %dpx\n", *hoogte);
}

void calcWidth(unsigned char *header, signed int *breedte) //Functie berekend de breedte van de afbeelding
{
    *breedte = header[21] << 24 | header[20] << 16 | header[19] << 8 | header[18]; //Resultaat: breedte = (8 bits header[21]) (8 bits header[20]) (8 bits header[19]) (8 bits header[18])
    printf("Breedte: %dpx\n", *breedte);
}

void cleanup(unsigned char *header, signed int *hoogte, signed int *breedte, FILE *inputBMP, FILE *targetBMP)
{
    free(header);
    free(hoogte);
    free(breedte);

    fclose(inputBMP);
    fclose(targetBMP);
}

int main()
{
    unsigned char *header = (unsigned char *) malloc(54 * sizeof(unsigned char));
    signed int *hoogte = (signed int *) malloc(sizeof(signed int));
    signed int *breedte = (signed int *) malloc(sizeof(signed int));

    FILE *inputBMP = openBMP(); //Opent BMP file
    FILE *targetBMP = openTargetBMP(); //Opent de BMP Target file

    readHeader(inputBMP, header, targetBMP); //Leest de header
    calcHeight(header, hoogte); //Berekend hoogte BMP file
    calcWidth(header, breedte); //Berekend breedte BMP file

    if(*breedte % 4 != 0 && *hoogte % 4 != 0)
    {
        printf("Incompatible Image\n");
        exit(-1);
    }

    cleanup(header, hoogte, breedte, inputBMP, targetBMP);

    return 0;
}

