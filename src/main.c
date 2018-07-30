/*
Author: Manohar Mukku
Date: 18.07.2018
Desc: Multilayer Perceptron implementation in C
GitHub: https://github.com/manoharmukku/multilayer-perceptron-in-c
*/

#include "mlp_trainer.h"
#include "mlp_classifier.h"
#include "read_csv.h"

int main(int argc, char** argv) {
    /*
    argv[0]: Executable file name Ex: a.out
    argv[1]: Number of hidden layers Ex: 3
    argv[2]: Size of each hidden layer separated by comma Ex: 4,5,5
    argv[3]: Hidden activation functions (identity - 1, sigmoid - 2, tanh - 3, relu - 4, softmax - 5)
    argv[4]: Alpha (L2 Regularization parameter value)
    argv[5]: Maximum number of iterations
    argv[6]: Number of units in output layer
    argv[7]: Output activation function (identity - 1, sigmoid - 2, tanh - 3, relu - 4, softmax - 5)
    argv[8]: Name of the csv file containing the train dataset
    argv[9]: Number of rows or samples in the train dataset
    argv[10]: Number of features including the output variable in the train dataset
    argv[11]: Name of the csv file containing the test dataset
    argv[12]: Number of rows or samples in the test dataset
    argv[13]: Number of features including the output variable in the test dataset
    */

    // Sanity check of command line arguments
    if (argc <= 1) {
        printf("Usage: ./MLP \'No. of hidden layers\' \'Size of each hidden layer separated by comma (no space in-between)\'\n \
            \'Hidden activations separated by comma (no space in-between)\' \'Size of output layer\' \'Output activation\' \n \
            \'Learning rate\' \'Max iterations\' \'Train_filename\' \'Train_rows\' \'Train_cols\' \'Test_filename\' \'Test_rows\' \'Test_cols\'\n");
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
    for (i = 0, tok = strtok(argv[3], ","); (i < param->n_hidden); i++) {
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

        tok = strtok(NULL, ",");
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

    // Get the parameters of the train dataset
    char* train_filename = argv[8];
    param->train_sample_size = atoi(argv[9]);
    // Feature size = Number of input features + 1 output feature
    param->feature_size = atoi(argv[10]);

    // Create 2D array memory for the dataset
    param->data_train = (double**)malloc(param->train_sample_size * sizeof(double*));
    for (i = 0; i < param->train_sample_size; i++)
        param->data_train[i] = (double*)malloc(param->feature_size * sizeof(double));

    // Read the train dataset from the csv into the 2D array
    read_csv(train_filename, param->train_sample_size, param->feature_size, param->data_train);

    // Get the parameters of the test dataset
    char* test_filename = argv[11];
    param->test_sample_size = atoi(argv[12]);
    // Feature size = Number of input features + 1 output feature
    param->feature_size = atoi(argv[13]);

    // Create 2D array memory for the dataset
    param->data_test = (double**)malloc(param->test_sample_size * sizeof(double*));
    for (i = 0; i < param->test_sample_size; i++)
        param->data_test[i] = (double*)malloc(param->feature_size * sizeof(double));

    // Read the test dataset from the csv into the 2D array
    read_csv(test_filename, param->test_sample_size, param->feature_size, param->data_test);

    // Total number of layers
    int n_layers = param->n_hidden + 2;

    // Save the sizes of layers in an array
    int* layer_sizes = (int*)calloc(n_layers, sizeof(int));

    layer_sizes[0] = param->feature_size - 1;
    layer_sizes[n_layers-1] = param->output_layer_size;

    for (i = 1; i < n_layers-1 ; i++)
        layer_sizes[i] = param->hidden_layers_size[i-1];

    // Create memory for the weight matrices between layers
    // weight is a pointer to the array of 2D arrays between the layers
    // Global variable
    weight = (double***)calloc(n_layers - 1, sizeof(double**));

    // Each 2D array between two layers i and i+1 is of size ((layer_size[i]+1) x layer_size[i+1])
    // The weight matrix includes weights for the bias terms too
    for (i = 0; i < n_layers-1; i++)
        weight[i] = (double**)calloc(layer_sizes[i]+1, sizeof(double*));

    int j;
    for (i = 0; i < n_layers-1; i++)
        for (j = 0; j < layer_sizes[i]+1; j++)
            weight[i][j] = (double*)calloc(layer_sizes[i+1], sizeof(double));

    // Train the neural network on the train data
    mlp_trainer(param, layer_sizes);

    // Classify the test data using the trained parameter weights
    mlp_classifier(param, layer_sizes);

    // Free the memory allocated in Heap
    for (i = 0; i < n_layers-1; i++)
        for (j = 0; j < layer_sizes[i]+1; j++)
            free(weight[i][j]);

    for (i = 0; i < n_layers-1; i++)
        free(weight[i]);

    free(weight);

    free(layer_sizes);

    for (i = 0; i < param->train_sample_size; i++)
        free(param->data_train[i]);

    for (i = 0; i < param->test_sample_size; i++)
        free(param->data_test[i]);

    free(param->data_train);
    free(param->data_test);
    free(param->hidden_activation_functions);
    free(param->hidden_layers_size);
    free(param);

    return 0;
}
