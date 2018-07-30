/*
Author: Manohar Mukku
Date: 18.07.2018
Desc: Read .csv file into a 2D array
GitHub: https://github.com/manoharmukku/multilayer-perceptron-in-c
Courtesy of amirmasoudabdol on GitHub: https://gist.github.com/amirmasoudabdol/f1efda29760b97f16e0e
*/

#include "read_csv.h"

void read_csv(char* filename, int rows, int cols, double** data) {
    // Open file and perform sanity check
    FILE* fp = fopen(filename, "r");
    if (NULL == fp) {
        printf("Error opening %s file\nMake sure you mentioned the path correctly\n", filename);
        exit(0);
    }

    // Create memory to read a line/row from the file
    char* line = (char*)malloc(MAX_LINE_SIZE * sizeof(char));

    // Read the file and save it in the matrix 'data'
    int i, j;
    for (i = 0; fgets(line, MAX_LINE_SIZE, fp) && i < rows; i++) {
        char* tok;
        for (j = 0, tok = strtok(line, "\t"); tok && *tok; j++, tok = strtok(NULL, "\t\n")) {
            data[i][j] = atof(tok);
        }
    }

    // Free the allocated memory in Heap for line
    free(line);

    // Close the file
    fclose(fp);
}
