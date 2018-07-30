/*
Author: Manohar Mukku
Date: 21.07.2018
Desc: Feedforward propagation
GitHub: https://github.com/manoharmukku/multilayer-perceptron-in-c
*/

#include "forward_propagation.h"

#define max(x, y) (x > y ? x : y)

void mat_mul(double* a, double** b, double* result, int n, int p) {
    // matrix a of size 1 x n (array)
    // matrix b of size n x p
    // matrix result of size 1 x p (array)
    // result = a * b
    int j, k;
    for (j = 0; j < p; j++) {
        result[j] = 0.0;
        for (k = 0; k < n; k++)
            result[j] += (a[k] * b[k][j]);
    }
}

void identity(int n, double* input, double* output) {
    output[0] = 1; // Bias term

    int i;
    for (i = 0; i < n; i++) 
        output[i+1] = input[i]; // Identity function
}

void sigmoid(int n, double* input, double* output) {
    output[0] = 1; // Bias term

    int i;
    for (i = 0; i < n; i++) 
        output[i+1] = 1.0 / (1.0 + exp(-input[i])); // Sigmoid function
}

void tan_h(int n, double* input, double* output) {
    output[0] = 1; // Bias term

    int i;
    for (i = 0; i < n; i++) 
        output[i+1] = tanh(input[i]); // tanh function
}

void relu(int n, double* input, double* output) {
    output[0] = 1; // Bias term

    int i;
    for (i = 0; i < n; i++) 
        output[i+1] = max(0.0, input[i]); // ReLU function
}

void softmax(int n, double* input, double* output) {
    output[0] = 1; // Bias term

    int i;
    double sum = 0.0;
    for (i = 0; i < n; i++)
        sum += exp(input[i]);

    for (i = 0; i < n; i++) 
        output[i+1] = exp(input[i]) / sum; // Softmax function
}

void forward_propagation(parameters* param, int training_example, int n_layers, int* layer_sizes, double** layer_inputs, double** layer_outputs) {
    // Fill the input layer's input and output (both are equal) from data matrix with the given training example
    int i;
    layer_outputs[0][0] = 1; // Bias term of input layer
    for (i = 0; i < param->feature_size-1; i++)
        layer_outputs[0][i+1] = layer_inputs[0][i] = param->data_train[training_example][i];

    // Perform forward propagation for each hidden layer
    // Calculate input and output of each hidden layer
    for (i = 1; i < n_layers-1; i++) {
        // Compute layer_inputs[i]
        mat_mul(layer_outputs[i-1], param->weight[i-1], layer_inputs[i], layer_sizes[i-1]+1, layer_sizes[i]);

        // Compute layer_outputs[i]
        // Activation functions (identity - 1, sigmoid - 2, tanh - 3, relu - 4, softmax - 5)
        switch (param->hidden_activation_functions[i-1]) {
            case 1: // identity
                identity(layer_sizes[i], layer_inputs[i], layer_outputs[i]);
                break;
            case 2: // sigmoid
                sigmoid(layer_sizes[i], layer_inputs[i], layer_outputs[i]);
                break;
            case 3: // tanh
                tan_h(layer_sizes[i], layer_inputs[i], layer_outputs[i]);
                break;
            case 4: // relu
                relu(layer_sizes[i], layer_inputs[i], layer_outputs[i]);
                break;
            case 5: // softmax
                softmax(layer_sizes[i], layer_inputs[i], layer_outputs[i]);
                break;
            default:
                printf("Forward propagation: Invalid hidden activation function\n");
                exit(0);
                break;
        }
    }

    // Fill the output layers's input and output
    mat_mul(layer_outputs[n_layers-2], param->weight[n_layers-2], layer_inputs[n_layers-1], layer_sizes[n_layers-2]+1, layer_sizes[n_layers-1]);

    // Activation functions (identity - 1, sigmoid - 2, tanh - 3, relu - 4, softmax - 5)
    switch (param->output_activation_function) {
        case 1: // identity
            identity(layer_sizes[n_layers-1], layer_inputs[n_layers-1], layer_outputs[n_layers-1]);
            break;
        case 2: // sigmoid
            sigmoid(layer_sizes[n_layers-1], layer_inputs[n_layers-1], layer_outputs[n_layers-1]);
            break;
        case 3: // tanh
            tan_h(layer_sizes[n_layers-1], layer_inputs[n_layers-1], layer_outputs[n_layers-1]);
            break;
        case 4: // relu
            relu(layer_sizes[n_layers-1], layer_inputs[n_layers-1], layer_outputs[n_layers-1]);
            break;
        case 5: // softmax
            softmax(layer_sizes[n_layers-1], layer_inputs[n_layers-1], layer_outputs[n_layers-1]);
            break;
        default:
            printf("Forward propagation: Invalid hidden activation function\n");
            exit(0);
            break;
    }
}
