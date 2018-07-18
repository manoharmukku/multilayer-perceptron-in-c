/*
Author: Manohar Mukku
Date: 18.07.2018
Desc: Read .csv into a 2D array
GitHub: https://github.com/manoharmukku/multilayer-perceptron-in-c
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_SIZE 1048576 // 2^20

void read_csv(char* filename, int rows, int cols, double** data) {
    // Open file and perform sanity check
    FILE* fp = fopen(filename, "r");
    if (NULL == fp) {
        printf("Error opening %s file\n", filename);
        exit(0);
    }

    // Create memory to read a line/row from the file
    char* line = (char*)malloc(MAX_LINE_SIZE * sizeof(char));

    // Read the file and save it in data
    int i, j;
    for (i = 0; fgets(line, MAX_LINE_SIZE, fp) && i < rows; i++) {
        char* tok;
        for (j = 0, tok = strtok(line, "\t"); tok && *tok; j++, tok = strtok(NULL, "\t\n")) {
            data[i][j] = atof(tok);
        }
    }

    // Free the memory
    free(line);

    // Close the file
    fclose(fp);
}
