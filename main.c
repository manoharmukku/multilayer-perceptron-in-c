/*
Author: Manohar Mukku
Date: 18.07.2018
Desc: Multilayer Perceptron implementation in C
GitHub: https://github.com/manoharmukku/multilayer-perceptron-in-c
*/

#include "main.h"

int main(int argc, char** argv) {
    /*
    argv[0]: Executable file name Ex: a.out
    argv[1]: Number of hidden layers Ex: 3
    argv[2]: Size of each hidden layer separated by comma Ex: 4,5,5
    argv[3]: Hidden activation functions (identity - 1, sigmoid - 2, tanh - 3, relu - 4, softmax - 5)
    argv[4]: Alpha (L2 Regularization parameter value)
    argv[5]: Maximum number of iterations
    argv[7]: Number of units in output layer
    argv[8]: Output activation function (identity - 1, sigmoid - 2, tanh - 3, relu - 4, softmax - 5)
    argv[9]: Name of the csv file containing the dataset
    argv[10]: Number of rows or samples in the dataset
    argv[11]: Number of features including the output variable in the dataset
    */

    // Sanity check of command line arguments
    if (argc <= 1) {
        printf("Usage: ./MLP \'No. of hidden layers\' \'Size of each hidden layer separated by comma (no space in-between)\'\n \
            \'Hidden activations separated by comma (no space in-between)\' \'Size of output layer\' \'Output activation\' \n \
            \'Learning rate\' \'Max iterations\' \'Filename\' \'Rows\' \'Cols\'\n");
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
    for (i = 0, tok = strtok(argv[2], ","); (tok = strtok(NULL, ",")) && (i < param->n_hidden); i++) {
        param->hidden_layers_size[i] = atoi(tok);
        // Sanity check of size of hidden layer
        if (param->hidden_layers_size[i] <= 0) {
            printf("Error: Hidden layer sizes should be positive\n");
            exit(0);
        }
    }

    // Hidden activation functions - Activation functions for each hidden layer
    param->hidden_activation_functions = (int*)malloc(param->n_hidden * sizeof(int));
    for (i = 0, tok = strtok(argv[3], ","); (tok = strtok(NULL, ",")) && (i < param->n_hidden); i++) {
        if (strcmp(tok, "identity") == 0) {
            param->hidden_activation_functions[i] = 1;
        }
        else if (strcmp(tok, "sigmoid") == 0) {
            param->hidden_activation_functions[i] = 2;
        }
        else if (strcmp(tok, "tanh") == 0) {
            param->hidden_activation_functions[i] = 3;
        }
        else if (strcmp(tok, "relu") == 0) {
            param->hidden_activation_functions[i] = 4;
        }
        else if (strcmp(tok, "softmax") == 0) {
            param->hidden_activation_functions[i] = 5;
        }
        else {
            printf("Error: Invalid value for hidden activation function\n");
            printf("Input either identity or sigmoid or tanh or relu or softmax for hidden activation function\n");
            exit(0);
        }
    }

    // Output layer size
    param->output_layer_size = atoi(argv[4]);
    if (param->output_layer_size <= 0) {
        printf("Output layer size should be positive\n");
        exit(0);
    }

    // Output activation function
    if (strcmp(argv[5], "identity") == 0) {
        param->output_activation_function = 1;
    }
    else if (strcmp(argv[5], "sigmoid") == 0) {
        param->output_activation_function = 2;
    }
    else if (strcmp(argv[5], "tanh") == 0) {
        param->output_activation_function = 3;
    }
    else if (strcmp(argv[5], "relu") == 0) {
        param->output_activation_function = 4;
    }
    else if (strcmp(argv[5], "softmax") == 0) {
        param->output_activation_function = 5;
    }
    else {
        printf("Error: Invalid value for output activation function\n");
        printf("Input either identity or sigmoid or tanh or relu or softmax for output activation function\n");
        exit(0);
    }

    // L2 Regularization parameter
    param->learning_rate = atoi(argv[6]);

    // Max. number of iterations
    param->n_iterations_max = atoi(argv[7]);
    if (param->n_iterations_max <= 0) {
        printf("Max. number of iterations value should be positive\n");
        exit(0);
    }

    // Momentum
    //param->momentum = atoi(argv[6]);

    // Get the parameters of the dataset
    char* filename = argv[8];
    param->sample_size = atoi(argv[9]);
    // Feature size = Number of input features + 1 output feature
    param->feature_size = atoi(argv[10]);

    // Create 2D array memory for the dataset
    param->data = (double**)malloc(param->sample_size * sizeof(double*));
    for (i = 0; i < param->sample_size; i++)
        param->data[i] = (double*)malloc(param->feature_size * sizeof(double));

    // Read the dataset from the csv into the 2D array
    read_csv(filename, param->sample_size, param->feature_size, param->data);

    // Train the neural network
    mlp_trainer(param);

    // Free the memory allocated in Heap
    for (i = 0; i < param->feature_size; i++)
        free(weight[0][i]);

    int j;
    for (i = 1; i < param->n_hidden; i++)
        for (j = 0; j < param->hidden_layers_size[i-1]+1; j++)
            free(weight[i][j]);

    for (i = 0; i < param->output_layer_size+1; i++)
        free(weight[param->n_hidden][i]);

    for (i = 0; i < param->n_hidden+1; i++)
        free(weight[i]);

    free(weight);

    for (i = 0; i < param->sample_size; i++)
        free(param->data[i]);

    free(param->data);
    free(param->hidden_activation_functions);
    free(param->hidden_layers_size);
    free(param);

    return 0;
}
