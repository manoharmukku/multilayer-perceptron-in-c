/*
Author: Manohar Mukku
Date: 20.07.2018
Desc: MLP Classifier
GitHub: https://github.com/manoharmukku/multilayer-perceptron-in-c
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct {
    int n_hidden;
    int* hidden_layers_size;
    int* hidden_activation_functions;
    double learning_rate;
    int n_iterations_max;
    int momentum;
    int output_layer_size;
    int output_activation_function;
    int sample_size;
    int feature_size;
    double** data;
} parameters;

void initialize_weights(int n_layers, int* layer_sizes, double*** weight) {
    srand(time(0));

    // epsilon = sqrt(6/(layer_size[i] + layer_size[i+1])) used for random initialization
    double* epsilon = (double*)calloc(n_layers-1, sizeof(double));
    int i;
    for (i = 0; i < n_layers-1; i++)
        epsilon[i] = sqrt(6.0 / (layer_sizes[i] + layer_sizes[i+1]));

    // Random initialization between [-epsilon[i], epsilon[i]] for weight[i]
    int j, k;
    for (i = 0; i < n_layers-1; i++)
        for (j = 0; j < layer_sizes[i]+1; j++)
            for (k = 0; k < layer_sizes[i+1]; k++)
                weight[i][j][k] = -epsilon[i] + ((double)rand() / ((double)RAND_MAX / (2.0 * epsilon[i])));

    // Free the memory allocated in Heap for epsilon array
    free(epsilon);
}

void MLPClassifier(parameters* param) {
    int n_layers = param->n_hidden + 2;

    // Save the sizes of layers in an array
    int** layer_sizes = (int*)calloc(n_layers, sizeof(int));

    layer_sizes[0] = param->feature_size - 1;
    layer_sizes[n_layers-1] = param->output_layer_size;

    int i;
    for (i = 1; i < n_layers-1 ; i++)
        layer_sizes[i] = param->hidden_layers_size[i-1];

    // Create memory for the weight matrices between layers
    // weight is a pointer to the array of 2D arrays between the layers
    double*** weight = (double***)calloc(n_layers - 1, sizeof(double**));

    // Each 2D array between two layers i and i+1 is of size ((layer_size[i]+1) x layer_size[i+1])
    // The weight matrix includes weights for the bias terms too
    for (i = 0; i < n_layers-1; i++)
        weight[i] = (double**)calloc(layer_sizes[i]+1, sizeof(double*));

    int j;
    for (i = 0; i < n_layers-1; i++)
        for (j = 0; j < layer_sizes[i]+1; j++)
            weight[i][j] = (double*)calloc(layer_sizes[i+1], sizeof(double));

    // Create memory for arrays of inputs to the layers
    double** layer_inputs = (double**)calloc(n_layers, sizeof(double*));

    for (i = 0; i < n_layers; i++)
        layer_inputs[i] = (double*)calloc(layer_sizes[i], sizeof(double));

    // Create memory for arrays of outputs from the layers
    double** layer_outputs = (double**)calloc(n_layers, sizeof(double*));

    for (i = 0; i < n_layers; i++)
        layer_outputs[i] = (double*)calloc(layer_sizes[i]+1, sizeof(double));

    // Initialize the weights
    initialize_weights(n_layers, layer_sizes, weight);

    

    // Free the memory allocated in Heap
    for (i = 0; i < n_layers; i++)
        free(layer_outputs[i]);

    free(layer_outputs);

    for (i = 0; i < n_layers; i++)
        free(layer_inputs[i]);

    free(layer_inputs);

    for (i = 0; i < n_layers - 1; i++)
        for (j = 0; j < layer_sizes[i]+1; j++)
            free(weight[i][j]);

    for (i = 0; i < n_layers - 1; i++)
        free(weight[i]);

    free(weight);

    free(layer_sizes);
}
