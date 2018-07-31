/*
Author: Manohar Mukku
Date: 31.07.2018
Desc: Write a matrix to a specified .csv file
GitHub: https://github.com/manoharmukku/multilayer-perceptron-in-c
*/

#include "write_csv.h"

void write_csv(char* filename, int rows, int cols, double** data) {
    FILE* fp = fopen(filename, "w");
    if (NULL == fp) {
        printf("Cannot create/open file %s. Make sure you have permission to create/open a file in the directory\n", filename);
        exit(0);
    }

    // Create a header in the file with the output layer node numbers
    int i;
    for (i = 1; i <= cols-1; i++)
        fprintf(fp, "Node %d output,", i);
    fprintf(fp, "Node %d output\n", cols);

    // Dump the matrix into the file element by element
    int j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j <= cols-2; j++) {
            fprintf(fp, "%lf,", data[i][j]);
        }
        fprintf(fp, "%lf\n", data[i][cols-1]);
    }

    // Close the file
    fclose(fp);
}