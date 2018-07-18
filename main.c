/*
Author: Manohar Mukku
Date: 18.07.2018
Desc: Multilayer Perceptron implementation in C
GitHub: https://github.com/manoharmukku/multilayer-perceptron-in-c
*/

#include <stdio.h>
#include <stdlib.h>

/*
Hidden Layer sizes: # of hidden layers, Size of each hidden layer separated by space (int)
Hidden activation function: identity, sigmoid, tanh, relu (char*)
Alpha (L2 Regularization parameter value): Float value (float)
Maximum number of iterations: > 0 (int)
Momentum for gradient descent update: Integer value (int)
Number of units in output layer: # of units in output layer (int)
Output activation: identity, sigmoid, tanh, relu (char*)
Filename: Name of the csv file containing the dataset
Rows: Number of rows or samples in the dataset
Cols: Number of features including the output variable in the dataset
*/

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

int main(int argc, char** argv) {
    // Sanity check of command line arguments
    if (argc < 1) {
        printf("Usage: 'No. of hidden layers' 'Size of each hidden layer separated by space' 'Hidden activation'\n
            'Alpha' 'Max iterations' 'Momentum' 'Size of output layer' 'Output activation' 'Filename' 'Rows' 'Cols'\n");
        exit(0);
    }

    // Create memory for training parameters struct
    training_parameters* train_params = (training_parameters*)malloc(sizeof(training_parameters));

    // Number of hidden layers
    train_params->n_hidden = atoi(argv[1]);
    // Sanity check of number of hidden layers
    if (train_params->n_hidden < 0) {
        printf("Error: Number of hidden layers should be >= 0\n");
        exit(0);
    }

    // Size of each hidden layer
    train_params->hidden_layer_size = (int*)malloc(train_params->n_hidden * sizeof(int));
    int i;
    for (i = 0; i < train_params->n_hidden; i++) {
        train_params->hidden_layer_size[i] = atoi(argv[i+2]);

        // Sanity check of size of hidden layer
        if (train_params->hidden_layer_size[i] <= 0) {
            printf("Error: Hidden layer sizes should be positive\n");
            exit(0);
        }
    }

    // Hidden activation function
    train_params->hidden_activation_function;
    switch (argv[train_params->n_hidden + 2]) {
        case "identity":
            train_params->hidden_activation_function = 1;
            break;
        case "sigmoid":
            train_params->hidden_activation_function = 2;
            break;
        case "tanh":
            train_params->hidden_activation_function = 3;
            break;
        case "relu":
            train_params->hidden_activation_function = 4;
            break;
        default:
            printf("Error: Invalid value for hidden activation function\n");
            printf("Input either identity or sigmoid or tanh or relu for hidden activation function\n");
            exit(0);
            break;
    }

    // L2 Regularization parameter
    train_params->regularization_parameter = atoi(argv[train_params->n_hidden + 3]);

    // Max. number of iterations
    train_params->n_iterations_max = atoi(argv[train_params->n_hidden + 4]);
    if (train_params->n_iterations_max <= 0) {
        printf("Max. number of iterations value should be positive\n");
        exit(0);
    }

    // Momentum
    train_params->momentum = atoi(argv[train_params->n_hidden + 5]);

    // Output layer size
    train_params->output_layer_size = atoi(argv[train_params->n_hidden + 6]);
    if (train_params->output_layer_size <= 0) {
        printf("Output layer size should be positive\n");
        exit(0);
    }

    // Output activation function
    train_params->output_activation_function;
    switch (argv[train_params->n_hidden + 7]) {
        case "identity":
            train_params->output_activation_function = 1;
            break;
        case "sigmoid":
            train_params->output_activation_function = 2;
            break;
        case "tanh":
            train_params->output_activation_function = 3;
            break;
        case "relu":
            train_params->output_activation_function = 4;
            break;
        default:
            printf("Error: Invalid value for output activation function\n");
            printf("Input either identity or sigmoid or tanh or relu for output activation function\n");
            exit(0);
            break;
    }

    // Create memory for the data parameters struct
    data_parameters* data_params = (data_parameters*)malloc(sizeof(data_parameters));

    // Get the parameters of the dataset
    data_params->filename = argv[train_params->n_hidden + 8];
    data_params->sample_size = atoi(argv[train_params->n_hidden + 9]);
    data_params->feature_size = atoi(argv[train_params->n_hidden + 10]);

    return 0;
}
