/*
Author: Manohar Mukku
Date: 18.07.2018
Desc: MLP Classifier
GitHub: https://github.com/manoharmukku/multilayer-perceptron-in-c
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int n_hidden;
    int* hidden_layer_size;
    int hidden_activation_function;
    float regularization_parameter;
    int n_iterations_max;
    int momentum;
    int output_layer_size;
    int output_activation_function;
} training_parameters;

typedef struct {
    char* filename;
    int sample_size;
    int feature_size;
} data_parameters;

void MLPClassifier(training_parameters* train_params, data_parameters* data_params) {
    int rows = data_params->sample_size;
    int cols = data_params->feature_size;

    // Create 2D array memory for the dataset
    double** data = (double**)malloc(rows * sizeof(double*));
    int i;
    for (i = 0; i < rows; i++)
        data[i] = (double*)malloc(cols * sizeof(double));

    // Read the dataset from the csv into the 2D array
    read_csv(data_params->filename, rows, cols, data);

    

    // Free the allocated memory
    for (i = 0; i < rows; i++)
        free(data[i]);
    free(data);
}