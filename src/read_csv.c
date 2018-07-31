/*
Author: Manohar Mukku
Date: 18.07.2018
Desc: Read .csv file into a 2D array
GitHub: https://github.com/manoharmukku/multilayer-perceptron-in-c
*/

#include "read_csv.h"

void read_csv(char* filename, int rows, int cols, double** data) {
    // Open file and perform sanity check
    FILE* fp = fopen(filename, "r");
    if (NULL == fp) {
        printf("Error opening %s file. Make sure you mentioned the file path correctly\n", filename);
        exit(0);
    }

    // Create memory to read a line/row from the file
    char* line = (char*)malloc(MAX_LINE_SIZE * sizeof(char));

    // Read the file line by line and save it in the matrix 'data'
    int i, j;
    for (i = 0; fgets(line, MAX_LINE_SIZE, fp) && i < rows; i++) {
        char* tok = strtok(line, ",");
        for (j = 0; tok && *tok; j++) {
            data[i][j] = atof(tok);
            tok = strtok(NULL, ",\n");
        }
    }

    // Free the allocated memory in Heap for line
    free(line);

    // Close the file
    fclose(fp);
}
