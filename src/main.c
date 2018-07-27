/*
Author: Manohar Mukku
Date: 18.07.2018
Desc: Multilayer Perceptron implementation in C
GitHub: https://github.com/manoharmukku/multilayer-perceptron-in-c
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mlp_trainer.h"

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

int main(int argc, char** argv) {
    /*
    argv[0]: Executable file name Ex: a.out
    argv[1]: Number of hidden layers Ex: 3
    argv[2]: Size of each hidden layer separated by comma Ex: 4,5,5
    argv[3]: Hidden activation functions (identity - 1, sigmoid - 2, tanh - 3, relu - 4, softmax - 5)
    argv[4]: Alpha (L2 Regularization parameter value)
    argv[5]: Maximum number of iterations
    argv[6]: Momentum for gradient descent update
    argv[7]: Number of units in output layer
    argv[8]: Output activation function (identity - 1, sigmoid - 2, tanh - 3, relu - 4, softmax - 5)
    argv[9]: Name of the csv file containing the dataset
    argv[10]: Number of rows or samples in the dataset
    argv[11]: Number of features including the output variable in the dataset
    */

    // Sanity check of command line arguments
    if (argc <= 1) {
        printf("Usage: ./a.out 'No. of hidden layers' 'Size of each hidden layer separated by comma' 'Hidden activations separated by comma'\n
            'Alpha' 'Max iterations' 'Momentum' 'Size of output layer' 'Output activation' 'Filename' 'Rows' 'Cols'\n");
        exit(0);
    }

    // Create memory for training parameters struct
    parameters* param = (parameters*)malloc(sizeof(parameters));

    // Number of hidden layers
    param->n_hidden = atoi(argv[1]);
    // Sanity check of number of hidden layers
    if (param->n_hidden < 0) {
        printf("Error: Number of hidden layers should be >= 0\n");
        exit(0);
    }

    // Size of each hidden layer
    param->hidden_layers_size = (int*)malloc(param->n_hidden * sizeof(int));
    int i;
    char* tok;
    for (i = 0, tok = strtok(argv[2], ","); tok = strtok(NULL, ",") && i < param->n_hidden; i++) {
        param->hidden_layers_size[i] = atoi(tok);
        // Sanity check of size of hidden layer
        if (param->hidden_layers_size[i] <= 0) {
            printf("Error: Hidden layer sizes should be positive\n");
            exit(0);
        }
    }

    // Hidden activation functions - Activation functions for each hidden layer
    param->hidden_activation_functions = (int*)malloc(n_hidden * sizeof(int));
    for (i = 0, tok = strtok(argv[3], ","); tok = strtok(NULL, ",") && i < n_hidden; i++) {
        switch (tok) {
            case "identity":
                param->hidden_activation_functions[i] = 1;
                break;
            case "sigmoid":
                param->hidden_activation_functions[i] = 2;
                break;
            case "tanh":
                param->hidden_activation_functions[i] = 3;
                break;
            case "relu":
                param->hidden_activation_functions[i] = 4;
                break;
            case "softmax":
                param->hidden_activation_functions[i] = 5;
                break;
            default:
                printf("Error: Invalid value for hidden activation function\n");
                printf("Input either identity or sigmoid or tanh or relu or softmax for hidden activation function\n");
                exit(0);
                break;
        }
    }

    // L2 Regularization parameter
    param->learning_rate = atoi(argv[4]);

    // Max. number of iterations
    param->n_iterations_max = atoi(argv[5]);
    if (param->n_iterations_max <= 0) {
        printf("Max. number of iterations value should be positive\n");
        exit(0);
    }

    // Momentum
    param->momentum = atoi(argv[6]);

    // Output layer size
    param->output_layer_size = atoi(argv[7]);
    if (param->output_layer_size <= 0) {
        printf("Output layer size should be positive\n");
        exit(0);
    }

    // Output activation function
    switch (argv[8]) {
        case "identity":
            param->output_activation_function = 1;
            break;
        case "sigmoid":
            param->output_activation_function = 2;
            break;
        case "tanh":
            param->output_activation_function = 3;
            break;
        case "relu":
            param->output_activation_function = 4;
            break;
        case "softmax":
            param->output_activation_function = 5;
            break;
        default:
            printf("Error: Invalid value for output activation function\n");
            printf("Input either identity or sigmoid or tanh or relu or softmax for output activation function\n");
            exit(0);
            break;
    }

    // Get the parameters of the dataset
    char* filename = argv[9];
    param->sample_size = atoi(argv[10]);
    // Feature size = Number of input features + 1 output feature
    param->feature_size = atoi(argv[11]);

    // Create 2D array memory for the dataset
    param->data = (double**)malloc(param->sample_size * sizeof(double*));
    int i;
    for (i = 0; i < param->sample_size; i++)
        param->data[i] = (double*)malloc(param->feature_size * sizeof(double));

    // Read the dataset from the csv into the 2D array
    read_csv(filename, param->sample_size, param->feature_size, param->data);

    

    // Free the memory allocated in Heap
    for (i = 0; i < param->sample_size; i++)
        free(param->data[i]);
    free(param->data);
    free(param->hidden_activation_functions);
    free(param->hidden_layers_size);
    free(param);

    return 0;
}
