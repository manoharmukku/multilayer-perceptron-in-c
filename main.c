/*
Author: Manohar Mukku
Date: 18.07.2018
Desc: Multilayer Perceptron implementation in C
GitHub: https://github.com/manoharmukku/multilayer-perceptron-in-c
*/

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int n_hidden;
    int* hidden_layer_sizes;
    int hidden_activation_function;
    float regularization_parameter;
    int n_iterations_max;
    int momentum;
    int output_layer_size;
    int output_activation_function;
    int sample_size;
    int feature_size;
    double** data;
} parameters;

int main(int argc, char** argv) {
    /*
    argv[0]: Executable file name Ex: a.out
    argv[1]: Number of hidden layers Ex: 3
    argv[2]: Size of each hidden layer separated by comma Ex: 4,5,5
    argv[3]: Hidden activation function (identity - 1, sigmoid - 2, tanh - 3, relu - 4)
    argv[4]: Alpha (L2 Regularization parameter value)
    argv[5]: Maximum number of iterations
    argv[6]: Momentum for gradient descent update
    argv[7]: Number of units in output layer
    argv[8]: Output activation function (identity - 1, sigmoid - 2, tanh - 3, relu - 4)
    argv[9]: Name of the csv file containing the dataset
    argv[10]: Number of rows or samples in the dataset
    argv[11]: Number of features including the output variable in the dataset
    */

    // Sanity check of command line arguments
    if (argc <= 1) {
        printf("Usage: ./a.out 'No. of hidden layers' 'Size of each hidden layer separated by comma' 'Hidden activation'\n
            'Alpha' 'Max iterations' 'Momentum' 'Size of output layer' 'Output activation' 'Filename' 'Rows' 'Cols'\n");
        exit(0);
    }

    // Create memory for training parameters struct
    parameters* params = (parameters*)malloc(sizeof(parameters));

    // Number of hidden layers
    params->n_hidden = atoi(argv[1]);
    // Sanity check of number of hidden layers
    if (params->n_hidden < 0) {
        printf("Error: Number of hidden layers should be >= 0\n");
        exit(0);
    }

    // Size of each hidden layer
    params->hidden_layer_sizes = (int*)malloc(params->n_hidden * sizeof(int));
    int i;
    char* tok;
    for (i = 0, tok = strtok(argv[2], ","); tok = strtok(NULL, ",") && i < params->n_hidden; i++) {
        params->hidden_layer_sizes[i] = atoi(tok);
        // Sanity check of size of hidden layer
        if (params->hidden_layer_sizes[i] <= 0) {
            printf("Error: Hidden layer sizes should be positive\n");
            exit(0);
        }
    }

    // Hidden activation function
    params->hidden_activation_function;
    switch (argv[3]) {
        case "identity":
            params->hidden_activation_function = 1;
            break;
        case "sigmoid":
            params->hidden_activation_function = 2;
            break;
        case "tanh":
            params->hidden_activation_function = 3;
            break;
        case "relu":
            params->hidden_activation_function = 4;
            break;
        default:
            printf("Error: Invalid value for hidden activation function\n");
            printf("Input either identity or sigmoid or tanh or relu for hidden activation function\n");
            exit(0);
            break;
    }

    // L2 Regularization parameter
    params->regularization_parameter = atoi(argv[4]);

    // Max. number of iterations
    params->n_iterations_max = atoi(argv[5]);
    if (params->n_iterations_max <= 0) {
        printf("Max. number of iterations value should be positive\n");
        exit(0);
    }

    // Momentum
    params->momentum = atoi(argv[6]);

    // Output layer size
    params->output_layer_size = atoi(argv[7]);
    if (params->output_layer_size <= 0) {
        printf("Output layer size should be positive\n");
        exit(0);
    }

    // Output activation function
    params->output_activation_function;
    switch (argv[8]) {
        case "identity":
            params->output_activation_function = 1;
            break;
        case "sigmoid":
            params->output_activation_function = 2;
            break;
        case "tanh":
            params->output_activation_function = 3;
            break;
        case "relu":
            params->output_activation_function = 4;
            break;
        default:
            printf("Error: Invalid value for output activation function\n");
            printf("Input either identity or sigmoid or tanh or relu for output activation function\n");
            exit(0);
            break;
    }

    // Get the parameters of the dataset
    char* filename = argv[9];
    params->sample_size = atoi(argv[10]);
    params->feature_size = atoi(argv[11]);

    // Create 2D array memory for the dataset
    params->data = (double**)malloc(params->sample_size * sizeof(double*));
    int i;
    for (i = 0; i < params->sample_size; i++)
        params->data[i] = (double*)malloc(params->feature_size * sizeof(double));

    // Read the dataset from the csv into the 2D array
    read_csv(filename, params->sample_size, params->feature_size, data);

    

    // Free the allocated memory
    for (i = 0; i < params->sample_size; i++)
        free(params->data[i]);
    free(params->data);
    free(params->hidden_layer_sizes);
    free(params);

    return 0;
}
